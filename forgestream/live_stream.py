"""GeminiLiveStream -- WebSocket bridge between AudioSource and Orchestrator."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any
from uuid import uuid4

from .audio.source import AudioSource
from .config import ForgeStreamConfig
from .emotion.buffer import AudioRingBuffer
from .emotion.extractor import EmotionExtractor
from .events.schema import Event, EventType
from .post_meeting import PostMeetingSynthesis
from .gemini.context import ContextBuilder
from .gemini.extraction import ClaimExtractor
from .gemini.prompt_tuner import PromptParams
from .graph.materializer import GraphMaterializer
from .orchestrator import Orchestrator
from .synthesis.requirements import RequirementDetector

logger = logging.getLogger(__name__)

MODE_INSTRUCTIONS = {
    "extract": (
        "You are an ECEF knowledge extractor in a meeting. "
        "For each substantive claim, emit a JSON object on its own line: "
        '{"text": "...", "speaker": "Speaker 1/2", "confidence": 0.0-1.0, '
        '"tone_markers": [], "topic_keywords": [], "is_requirement": false, '
        '"is_question": false}. '
        "Focus on what the expert needs built. Extract requirements, "
        "constraints, tech preferences. Do NOT summarize. "
        "Extract EVERY claim. Emit in real-time."
    ),
    "collaborative": (
        "You are an ECEF knowledge extractor in a design discussion. "
        "For each substantive claim, emit a JSON object on its own line: "
        '{"text": "...", "speaker": "Speaker 1/2", "confidence": 0.0-1.0, '
        '"tone_markers": [], "topic_keywords": [], "is_requirement": false, '
        '"is_question": false}. '
        "Focus on architectural decisions, trade-offs, agreements "
        "and disagreements. Emit in real-time."
    ),
    "knowledge": (
        "You are an ECEF knowledge extractor doing expertise capture. "
        "For each substantive claim, emit a JSON object on its own line: "
        '{"text": "...", "speaker": "Speaker 1/2", "confidence": 0.0-1.0, '
        '"tone_markers": [], "topic_keywords": [], "is_requirement": false, '
        '"is_question": false}. '
        "Focus on domain knowledge, mental models, heuristics, "
        "and tacit knowledge. Emit in real-time."
    ),
}


class GeminiLiveStream:
    """Manages the Gemini Live API WebSocket session.

    Connects an AudioSource to the Orchestrator:
    AudioSource -> PCM chunks -> Gemini -> claims -> Orchestrator -> EventBus -> TUI
    """

    def __init__(
        self,
        config: ForgeStreamConfig,
        orchestrator: Orchestrator,
        audio_source: AudioSource,
        mode: str = "extract",
    ) -> None:
        self.config = config
        self.orchestrator = orchestrator
        self.audio_source = audio_source
        self.mode = mode

        self._session = None
        self._session_cm = None
        self._active = False
        self._tasks: list[asyncio.Task] = []

        self.branch_id = uuid4()
        self.extractor = ClaimExtractor(
            session_id=orchestrator.session_id,
            branch_id=self.branch_id,
        )
        self.context_builder = ContextBuilder()
        self.req_detector = RequirementDetector()
        self.materializer = GraphMaterializer()

        # Proof obligation pipeline (always active)
        self.proof_detector = orchestrator.attach_proof_detector()

        # Load persisted evaluator weights (GRPO loop continuity)
        pms = PostMeetingSynthesis(config=config, data_dir=config.data_dir)
        saved_weights = pms.load_weights()
        orchestrator.evaluator.weights = saved_weights
        self._meeting_count = pms.load_meeting_count()
        logger.info(
            "Loaded weights (meeting %d): %s",
            self._meeting_count,
            {k: round(v, 3) for k, v in saved_weights.items()},
        )

        # Load persisted rapport weights (GRPO loop continuity)
        self._rapport_weights = pms.load_rapport_weights(self._meeting_count)
        logger.info(
            "Loaded rapport weights: %s",
            {k: round(v, 3) for k, v in self._rapport_weights.items()},
        )

        # Load persisted prompt params (PromptTuner GRPO loop)
        from pathlib import Path as _Path
        _prompt_params_path = str(_Path(config.data_dir) / "prompt_params.json")
        try:
            self._prompt_params = PromptParams.load(_prompt_params_path)
            logger.info("Loaded prompt params: %s", self._prompt_params.to_dict())
        except (FileNotFoundError, OSError):
            self._prompt_params = PromptParams()
            logger.info("Using default prompt params")

        # Load persisted config overrides (ConfigTuner GRPO loop)
        from .config import load_config_overrides
        config = load_config_overrides(data_dir=config.data_dir, config=config)
        self.config = config
        logger.info("Applied config overrides from %s/config_overrides.json", config.data_dir)

        # Load persisted trust region state
        from .governor.trust_region import TrustRegion as _TrustRegion
        from pathlib import Path as _TRPath
        _tr_path = _TRPath(config.data_dir) / "trust_region.json"
        self._trust_region = _TrustRegion.load(_tr_path)
        self._trust_region_path = _tr_path
        logger.info("Loaded trust region: epsilon=%.3f", self._trust_region.epsilon)

        # Load Sentinel meta-GRPO params (optional — tunes ForgeStream GRPO behavior)
        try:
            import json as _json
            _sentinel_state = _TRPath.home() / ".claude" / "state" / "sentinel"
            # Find project-specific meta params
            _registry_path = _sentinel_state / "registry.json"
            if _registry_path.exists():
                _registry = _json.loads(_registry_path.read_text())
                for _rpath in _registry:
                    if os.getcwd().startswith(_rpath):
                        _pid = _rpath.rsplit("/", 1)[-1]
                        import hashlib
                        _pid = f"{_pid}-{hashlib.md5(_rpath.encode()).hexdigest()[:6]}"
                        _meta_path = _sentinel_state / "projects" / _pid / "forgestream_meta_params.json"
                        if _meta_path.exists():
                            _meta = _json.loads(_meta_path.read_text())
                            logger.info("Loaded Sentinel meta-GRPO params: iteration=%d", _meta.get("meta_iteration", 0))
                        break
        except Exception:
            pass  # Sentinel meta-params are optional

        # Emotion pipeline (parallel to claim extraction)
        if config.emotion_enabled:
            self.audio_buffer: AudioRingBuffer | None = AudioRingBuffer(
                capacity_seconds=config.emotion_buffer_seconds,
            )
            # Speaker diarization (optional)
            diarizer = None
            if config.diarization_enabled and config.huggingface_token:
                from .emotion.diarizer import SpeakerDiarizer
                if SpeakerDiarizer.is_available():
                    diarizer = SpeakerDiarizer(
                        huggingface_token=config.huggingface_token,
                    )
            # SenseVoice categorical emotion classifier (optional)
            classifier = None
            if config.emotion_ml_enabled:
                from .emotion.classifier import SenseVoiceClassifier
                if SenseVoiceClassifier.is_available():
                    classifier = SenseVoiceClassifier(
                        model_name=config.sensevoice_model,
                    )
                    logger.info("SenseVoiceClassifier enabled: %s", config.sensevoice_model)
                else:
                    logger.info(
                        "emotion_ml_enabled=True but funasr not installed — "
                        "SenseVoice classifier disabled"
                    )
            self.emotion_extractor: EmotionExtractor | None = EmotionExtractor(
                orchestrator=orchestrator,
                audio_buffer=self.audio_buffer,
                branch_id=self.branch_id,
                window_seconds=config.emotion_window_seconds,
                stride_seconds=config.emotion_stride_seconds,
                diarizer=diarizer,
                classifier=classifier,
            )
            self._emotion_queue: asyncio.Queue[tuple[bytes, int]] = asyncio.Queue()
            self.emotion_correlator = orchestrator.attach_emotion_correlator()
            self.dynamics_engine = orchestrator.attach_dynamics_engine()
            if config.rapport_enabled:
                self.rapport_engine = orchestrator.attach_rapport_engine(
                    meeting_count=self._meeting_count,
                    damping_factor=config.rapport_damping_factor,
                    runpod_endpoint=config.runpod_crqa_endpoint,
                    runpod_timeout=config.runpod_timeout_seconds,
                    rapport_weights=self._rapport_weights,
                )
            else:
                self.rapport_engine = None
        else:
            self.audio_buffer = None
            self.emotion_extractor = None
            self._emotion_queue = None
            self.emotion_correlator = None
            self.dynamics_engine = None
            self.rapport_engine = None

        # Sentinel bridge — forwards events to the Sentinel daemon if running
        self._sentinel_forwarder = None
        try:
            from .sentinel_bridge import SentinelForwarder
            forwarder = SentinelForwarder(project_path=os.getcwd())
            forwarder.subscribe(orchestrator.event_bus)
            self._sentinel_forwarder = forwarder
            logger.info("Sentinel bridge: attached to EventBus")
        except Exception as exc:
            logger.debug("Sentinel bridge: not available (%s)", exc)

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def _parse_jsonl(self, text: str) -> list[dict[str, Any]]:
        """Parse JSONL text, skipping invalid lines."""
        claims = []
        for line in text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("```"):
                continue
            try:
                claims.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return claims

    async def connect(self) -> None:
        """Establish WebSocket connection to Gemini Live API."""
        try:
            from google import genai

            # Live API requires API key auth, NOT Vertex AI
            if self.config.gemini_api_key:
                client = genai.Client(api_key=self.config.gemini_api_key)
            else:
                client = genai.Client(
                    vertexai=self.config.gemini_use_vertex,
                    project=self.config.gemini_project,
                    location=self.config.gemini_location,
                )

            from .gemini.prompt_tuner import PromptTuner as _PromptTuner
            base_instruction = MODE_INSTRUCTIONS.get(self.mode, MODE_INSTRUCTIONS["extract"])
            system_instruction = _PromptTuner().apply_params(base_instruction, self._prompt_params)

            from google.genai import types

            self._session_cm = client.aio.live.connect(
                model=self.config.gemini_model,
                config=types.LiveConnectConfig(
                    response_modalities=[types.Modality.AUDIO],
                    system_instruction=system_instruction,
                    temperature=0.0,
                ),
            )
            self._session = await self._session_cm.__aenter__()
            self._active = True
            logger.info("Connected to Gemini Live API")
        except ImportError:
            raise RuntimeError(
                "google-genai not installed. pip install google-genai"
            )

    async def start(self) -> None:
        """Start streaming audio and receiving claims."""
        await self.connect()
        await self.audio_source.start()
        self._tasks = [
            asyncio.create_task(self._send_loop()),
            asyncio.create_task(self._receive_loop()),
            asyncio.create_task(self._context_injection_loop()),
        ]
        if self.emotion_extractor is not None:
            self._tasks.append(
                asyncio.create_task(self._emotion_extraction_loop())
            )

    async def stop(
        self,
        run_post_meeting: bool = True,
        human_score: float | None = None,
    ) -> dict | None:
        """Stop all loops, disconnect, and optionally run post-meeting synthesis.

        Args:
            run_post_meeting: Whether to run post-meeting synthesis pipeline.
            human_score: Optional human quality score (0.0-1.0) passed to GRPO
                weight tuning. If None, auto-score is computed from events.

        Returns the post-meeting result dict if run_post_meeting is True.
        """
        self._active = False
        await self.audio_source.stop()
        for task in self._tasks:
            task.cancel()
        if self._session_cm:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._session = None
            self._session_cm = None

        if run_post_meeting:
            return await self._run_post_meeting(human_score=human_score)
        return None

    async def _run_post_meeting(self, human_score: float | None = None) -> dict:
        """Run the full post-meeting synthesis pipeline.

        Args:
            human_score: Optional human quality score forwarded to pms.run().
        """
        pms = PostMeetingSynthesis(config=self.config, data_dir=self.config.data_dir)
        events = self.orchestrator._event_buffer
        meeting_name = self.config.meeting_name or "unnamed"
        result = await pms.run(
            events,
            meeting_name=meeting_name,
            human_score=human_score,
            audio_buffer=self.audio_buffer,
        )
        logger.info(
            "Post-meeting synthesis complete: E(π)=%.3f, meeting_count=%d",
            result["e_meso"], result["meeting_count"],
        )

        # Persist trust region state
        self._trust_region.save(str(self._trust_region_path))

        # Notify Sentinel daemon of meeting end
        if self._sentinel_forwarder:
            try:
                await self._sentinel_forwarder.on_meeting_end(result, meeting_name)
                await self._sentinel_forwarder.close()
            except Exception as exc:
                logger.debug("Sentinel bridge: meeting-end notification failed (%s)", exc)

        return result

    async def _send_loop(self) -> None:
        """Send audio chunks to Gemini and tee into emotion pipeline."""
        try:
            async for chunk in self.audio_source.chunks():
                if not self._active:
                    break
                # Tee into emotion pipeline
                if self.audio_buffer is not None:
                    chunk_idx = self.audio_buffer.write_chunk(chunk)
                    self._emotion_queue.put_nowait((chunk, chunk_idx))
                # Send to Gemini via realtime input API
                if self._session:
                    from google.genai import types as _types
                    await self._session.send_realtime_input(
                        audio=_types.Blob(
                            data=chunk,
                            mime_type="audio/pcm;rate=16000",
                        ),
                    )
        except asyncio.CancelledError:
            pass

    async def _receive_loop(self) -> None:
        """Receive and process claims from Gemini.

        With AUDIO-only modality, Gemini responds with speech. We extract
        claims from transcriptions, buffering incremental updates until
        a transcription is marked finished or a turn completes.
        """
        if not self._session:
            return

        # Buffer incremental transcriptions until finished
        input_buffer = ""
        output_buffer = ""

        try:
            async for response in self._session.receive():
                if not self._active:
                    break

                # Direct text (LiveServerMessage.text property)
                try:
                    if response.text:
                        await self._process_text_as_claims(response.text)
                except (AttributeError, ValueError):
                    pass

                # Transcriptions from server_content
                sc = response.server_content
                if not sc:
                    continue

                # Buffer input transcription (user speech)
                if sc.input_transcription and sc.input_transcription.text:
                    input_buffer += sc.input_transcription.text
                    # Emit when finished flag is set
                    if getattr(sc.input_transcription, "finished", False):
                        if input_buffer.strip():
                            await self._process_text_as_claims(input_buffer.strip())
                        input_buffer = ""

                # Buffer output transcription (model speech)
                if sc.output_transcription and sc.output_transcription.text:
                    output_buffer += sc.output_transcription.text
                    if getattr(sc.output_transcription, "finished", False):
                        if output_buffer.strip():
                            await self._process_text_as_claims(output_buffer.strip())
                        output_buffer = ""

                # Turn complete — flush any remaining buffer
                if sc.turn_complete:
                    if input_buffer.strip():
                        await self._process_text_as_claims(input_buffer.strip())
                        input_buffer = ""
                    if output_buffer.strip():
                        await self._process_text_as_claims(output_buffer.strip())
                        output_buffer = ""

            # Flush on exit
            if input_buffer.strip():
                await self._process_text_as_claims(input_buffer.strip())
            if output_buffer.strip():
                await self._process_text_as_claims(output_buffer.strip())
        except asyncio.CancelledError:
            # Flush on cancel
            if input_buffer.strip():
                await self._process_text_as_claims(input_buffer.strip())
            if output_buffer.strip():
                await self._process_text_as_claims(output_buffer.strip())

    async def _process_text_as_claims(self, text: str) -> None:
        """Convert text (JSONL or raw transcription) into claim events."""
        # Try JSONL first (structured claims from TEXT modality)
        claims = self._parse_jsonl(text)
        if claims:
            for claim_data in claims:
                event = self.extractor.parse_claim(claim_data)
                await self.orchestrator.process_event(event)
                await self._check_requirement(event)
        else:
            # Raw transcription — create claim from buffered text
            event = self.extractor.parse_claim({
                "text": text,
                "speaker": "Speaker 1",
                "confidence": 0.7,
                "tone_markers": [],
                "topic_keywords": [],
            })
            await self.orchestrator.process_event(event)
            await self._check_requirement(event)
            logger.info("Claim from transcription: %s", text[:80])

    async def _check_requirement(self, event: Event) -> None:
        """Check if a claim event contains a requirement and emit suggestion."""
        req = self.req_detector.check(event)
        if req:
            suggestion = Event(
                event_type=EventType.SUGGESTION,
                session_id=self.orchestrator.session_id,
                branch_id=self.branch_id,
                author="synthesis",
                evaluator=0.0,
                payload={
                    "text": f"Scaffold: {req['description'][:60]}",
                    "priority": 0.7,
                },
            )
            await self.orchestrator.process_event(suggestion)

    async def _emotion_extraction_loop(self) -> None:
        """Process audio chunks through the EmotionExtractor."""
        try:
            while self._active:
                chunk, chunk_idx = await self._emotion_queue.get()
                await self.emotion_extractor.process_chunk(chunk, chunk_idx)
        except asyncio.CancelledError:
            pass

    async def _context_injection_loop(self) -> None:
        """Inject knowledge graph summary every 10 minutes."""
        try:
            while self._active:
                await asyncio.sleep(600)
                if self._session and self._active:
                    events = self.orchestrator._event_buffer
                    graph = self.materializer.materialize(events)
                    summary = self.context_builder.build_injection(graph, [])
                    await self._session.send({"text": summary})
                    logger.info("Context injection sent")
        except asyncio.CancelledError:
            pass
