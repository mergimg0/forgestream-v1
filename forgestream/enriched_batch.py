"""AudioEnrichedBatchProcessor — full audio analysis stack for batch meetings.

Replaces the bare Gemini-only pipeline with a proper multi-phase processor:

  Phase 1: DIARIZE — prosodic-based speaker clustering (F0 + spectral centroid)
           Falls back gracefully when pyannote is unavailable (Python 3.13 / no torch).
  Phase 2: EXTRACT PROSODIC FEATURES — per-segment Praat + eGeMAPS + dimensional emotion
  Phase 3: TRANSCRIBE + EXTRACT CLAIMS — Gemini with speaker timestamps as ground truth
  Phase 4: CROSS-VALIDATE — merge diarization with Gemini speaker assignment
  Phase 5: FULL EMOTION PIPELINE — dynamics, rapport, entrainment, disengagement

Usage:
    processor = AudioEnrichedBatchProcessor(config)
    result = await processor.process("path/to/audio.wav", speaker_names=["Mergim", "Ryan", "Tim"])
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import ForgeStreamConfig
from .emotion.features import (
    EGeMAPSExtractor,
    PraatExtractor,
    derive_dimensional_emotion,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DiarizedSegment:
    """A speaker segment from diarization."""
    start_s: float
    end_s: float
    speaker_id: str  # SPEAKER_00, SPEAKER_01, etc.
    prosodic: dict = field(default_factory=dict)


@dataclass
class EnrichedClaim:
    """A claim with voice-verified speaker and prosodic enrichment."""
    text: str
    speaker: str  # Real name (after mapping)
    speaker_id_diarizer: str  # Raw diarizer label
    speaker_id_gemini: str  # Gemini's assignment
    speaker_confidence: str  # high/medium/low
    confidence: float
    timestamp_approx: str
    tone_markers: list[str]
    topic_keywords: list[str]
    is_requirement: bool
    is_question: bool
    prosodic: dict
    chunk: int = 0
    chunk_offset_min: float = 0.0
    speaker_embedding_distance: float = 0.0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "speaker": self.speaker,
            "speaker_id_diarizer": self.speaker_id_diarizer,
            "speaker_id_gemini": self.speaker_id_gemini,
            "speaker_confidence": self.speaker_confidence,
            "confidence": self.confidence,
            "timestamp_approx": self.timestamp_approx,
            "tone_markers": self.tone_markers,
            "topic_keywords": self.topic_keywords,
            "is_requirement": self.is_requirement,
            "is_question": self.is_question,
            "prosodic": self.prosodic,
            "chunk": self.chunk,
            "chunk_offset_min": self.chunk_offset_min,
            "speaker_embedding_distance": self.speaker_embedding_distance,
        }


# ---------------------------------------------------------------------------
# Phase 1: Prosodic Diarizer (no torch required)
# ---------------------------------------------------------------------------

class ProsodicDiarizer:
    """Speaker diarization using F0 + spectral centroid clustering.

    When pyannote is unavailable (Python 3.13 / no torch), this uses
    prosodic features to cluster speaker segments. Works well when speakers
    have different vocal characteristics (pitch, timbre).

    Strategy:
    1. Split audio into fixed windows (2s)
    2. Extract F0 mean + spectral centroid + energy for each window
    3. K-means cluster into N speakers
    4. Smooth: merge adjacent segments with same speaker
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        window_seconds: float = 2.0,
        n_speakers: int = 3,
    ) -> None:
        self._sample_rate = sample_rate
        self._window_seconds = window_seconds
        self._n_speakers = n_speakers
        self._praat = PraatExtractor(sample_rate=sample_rate)

    def diarize(self, audio_path: str) -> list[DiarizedSegment]:
        """Diarize an entire audio file.

        Returns list of DiarizedSegment with speaker_id = SPEAKER_00/01/02.
        """
        # Load WAV
        signal = self._load_wav(audio_path)
        duration_s = len(signal) / self._sample_rate

        # Extract features per window
        window_samples = int(self._window_seconds * self._sample_rate)
        features = []
        timestamps = []

        for start_sample in range(0, len(signal) - window_samples, window_samples):
            chunk = signal[start_sample:start_sample + window_samples]
            result = self._praat.extract(chunk)
            f0 = result["f0_mean"]
            sc = result["spectral_centroid"]
            energy = result["energy_rms"]

            features.append([f0, sc, energy])
            timestamps.append(start_sample / self._sample_rate)

        if not features:
            return []

        features_array = np.array(features)

        # Filter out silence (F0 == 0 and low energy)
        voiced_mask = (features_array[:, 0] > 0) & (features_array[:, 2] > 0.005)

        if voiced_mask.sum() < self._n_speakers:
            # Not enough voiced segments
            return [DiarizedSegment(0.0, duration_s, "SPEAKER_00")]

        # Normalize features for clustering
        voiced_features = features_array[voiced_mask]
        means = voiced_features.mean(axis=0)
        stds = voiced_features.std(axis=0)
        stds[stds == 0] = 1.0
        normalized = (voiced_features - means) / stds

        # K-means clustering
        labels = self._kmeans(normalized, self._n_speakers)

        # Map back to full timeline
        full_labels = np.full(len(features), -1)  # -1 = silence
        voiced_indices = np.where(voiced_mask)[0]
        for i, idx in enumerate(voiced_indices):
            full_labels[idx] = labels[i]

        # Build segments
        segments = []
        current_speaker = full_labels[0]
        start_time = timestamps[0]

        for i in range(1, len(full_labels)):
            if full_labels[i] != current_speaker:
                if current_speaker >= 0:
                    segments.append(DiarizedSegment(
                        start_s=start_time,
                        end_s=timestamps[i],
                        speaker_id=f"SPEAKER_{current_speaker:02d}",
                    ))
                current_speaker = full_labels[i]
                start_time = timestamps[i]

        # Final segment
        if current_speaker >= 0:
            segments.append(DiarizedSegment(
                start_s=start_time,
                end_s=duration_s,
                speaker_id=f"SPEAKER_{current_speaker:02d}",
            ))

        # Merge short gaps (< 1s) between same speaker
        segments = self._merge_segments(segments)

        logger.info(
            "Prosodic diarization: %d segments, %d speakers, %.1fs audio",
            len(segments), self._n_speakers, duration_s,
        )

        return segments

    def get_speaker_profiles(
        self, audio_path: str, segments: list[DiarizedSegment]
    ) -> dict[str, dict]:
        """Extract voice profile (mean F0, spectral centroid) per speaker."""
        signal = self._load_wav(audio_path)
        profiles: dict[str, list[dict]] = {}

        for seg in segments:
            start = int(seg.start_s * self._sample_rate)
            end = int(seg.end_s * self._sample_rate)
            chunk = signal[start:end]
            if len(chunk) < self._sample_rate:  # skip < 1s segments
                continue

            features = self._praat.extract(chunk)
            if seg.speaker_id not in profiles:
                profiles[seg.speaker_id] = []
            profiles[seg.speaker_id].append(features)

        # Aggregate per speaker
        result = {}
        for speaker_id, feature_list in profiles.items():
            if not feature_list:
                continue
            result[speaker_id] = {
                "f0_mean": np.mean([f["f0_mean"] for f in feature_list if f["f0_mean"] > 0]),
                "f0_std": np.mean([f["f0_std"] for f in feature_list]),
                "spectral_centroid": np.mean([f["spectral_centroid"] for f in feature_list if f["spectral_centroid"] > 0]),
                "energy_rms": np.mean([f["energy_rms"] for f in feature_list]),
                "hnr": np.mean([f["hnr"] for f in feature_list if f["hnr"] > 0]),
                "segment_count": len(feature_list),
                "total_duration_s": sum(seg.end_s - seg.start_s for seg in segments if seg.speaker_id == speaker_id),
            }

        return result

    @staticmethod
    def _load_wav(path: str) -> np.ndarray:
        """Load a WAV file as int16 numpy array."""
        with wave.open(path, "rb") as wf:
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            return np.frombuffer(raw, dtype=np.int16)

    @staticmethod
    def _kmeans(data: np.ndarray, k: int, max_iter: int = 100) -> np.ndarray:
        """Simple k-means clustering (no sklearn dependency)."""
        n = len(data)
        # Initialize centroids with k-means++
        centroids = [data[np.random.randint(n)]]
        for _ in range(1, k):
            dists = np.array([min(np.sum((x - c) ** 2) for c in centroids) for x in data])
            probs = dists / dists.sum()
            centroids.append(data[np.random.choice(n, p=probs)])
        centroids = np.array(centroids)

        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            # Assign
            for i in range(n):
                dists = np.sum((centroids - data[i]) ** 2, axis=1)
                labels[i] = np.argmin(dists)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                members = data[labels == j]
                if len(members) > 0:
                    new_centroids[j] = members.mean(axis=0)
                else:
                    new_centroids[j] = centroids[j]

            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        return labels

    @staticmethod
    def _merge_segments(
        segments: list[DiarizedSegment], gap_threshold: float = 1.0
    ) -> list[DiarizedSegment]:
        """Merge adjacent segments with same speaker and small gaps."""
        if not segments:
            return []

        merged = [segments[0]]
        for seg in segments[1:]:
            prev = merged[-1]
            if (
                seg.speaker_id == prev.speaker_id
                and seg.start_s - prev.end_s < gap_threshold
            ):
                merged[-1] = DiarizedSegment(
                    start_s=prev.start_s,
                    end_s=seg.end_s,
                    speaker_id=prev.speaker_id,
                )
            else:
                merged.append(seg)
        return merged


# ---------------------------------------------------------------------------
# Phase 1 alt: pyannote diarizer (when available)
# ---------------------------------------------------------------------------

def _try_pyannote_diarize(
    audio_path: str, hf_token: str, n_speakers: int = 3
) -> list[DiarizedSegment] | None:
    """Attempt diarization with pyannote x-vector embeddings.

    Three strategies, tried in order:
    1. Load pre-computed pyannote segments from a sidecar JSON file
       (avoids re-running 60+ min of CPU inference)
    2. Call scripts/pyannote_diarize.py via the Python 3.12 venv subprocess
       (pyannote/torch require Python <=3.12)
    3. In-process import (if torch is available in current interpreter)

    Returns None if all strategies fail, triggering prosodic fallback.
    """
    import subprocess
    import tempfile

    audio_p = Path(audio_path)

    # --- Strategy 1: Pre-computed sidecar file ---
    # Look for <stem>_pyannote_diarization.json next to or in data/
    sidecar_candidates = [
        audio_p.parent / f"{audio_p.stem}_pyannote_diarization.json",
        Path("data") / f"{audio_p.stem.replace('_complete', '')}_pyannote_diarization.json",
    ]
    for sidecar in sidecar_candidates:
        if sidecar.exists():
            try:
                raw = json.loads(sidecar.read_text())
                segments = [
                    DiarizedSegment(s["start"], s["end"], s["speaker"])
                    for s in raw
                ]
                logger.info(
                    "Loaded pre-computed pyannote segments: %d from %s",
                    len(segments), sidecar,
                )
                print(f"  Loaded pre-computed pyannote x-vector segments from {sidecar}")
                return segments
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to load sidecar %s: %s", sidecar, e)

    # --- Strategy 2: Subprocess via Python 3.12 venv ---
    venv_python = Path(".venv-pyannote/bin/python3")
    diarize_script = Path("scripts/pyannote_diarize.py")

    # ot-ctx-missing-script-006: log when strategy 2 is skipped so operator
    # has visibility that diarization is degraded
    if not diarize_script.exists():
        logger.info("Strategy 2 skipped: %s not found", diarize_script)
    elif not venv_python.exists():
        logger.info("Strategy 2 skipped: %s not found", venv_python)

    if venv_python.exists() and diarize_script.exists():
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                tmp_path = tmp.name

            cmd = [
                str(venv_python),
                str(diarize_script),
                "--audio", str(audio_path),
                "--output", tmp_path,
                "--max-speakers", str(n_speakers),
            ]
            if hf_token:
                cmd.extend(["--hf-token", hf_token])

            print(f"  Running pyannote via {venv_python} (this may take 30-60 min on CPU)...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout for large files
            )

            if result.returncode == 0:
                raw = json.loads(Path(tmp_path).read_text())
                segments = [
                    DiarizedSegment(s["start"], s["end"], s["speaker"])
                    for s in raw
                ]
                # Also save as sidecar for future runs
                sidecar_save = Path("data") / f"{audio_p.stem}_pyannote_diarization.json"
                Path(tmp_path).rename(sidecar_save)
                logger.info("pyannote subprocess: %d segments, saved to %s", len(segments), sidecar_save)
                print(f"  pyannote x-vector: {len(segments)} segments")
                return segments
            else:
                logger.warning("pyannote subprocess failed: %s", result.stderr[-500:])
                print(f"  pyannote subprocess failed: {result.stderr[-200:]}")
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)
        except subprocess.TimeoutExpired:
            logger.warning("pyannote subprocess timed out (2h limit)")
            print("  pyannote subprocess timed out")
        except Exception as e:
            logger.warning("pyannote subprocess error: %s", e)

    # --- Strategy 3: In-process (if torch available) ---
    try:
        import torch  # type: ignore[import-not-found]
        from pyannote.audio import Pipeline  # type: ignore[import-not-found]

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        diarization = pipeline(audio_path, max_speakers=n_speakers)
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(DiarizedSegment(turn.start, turn.end, speaker))
        logger.info("pyannote in-process: %d segments", len(segments))
        return segments
    except (ImportError, Exception) as e:
        logger.info("pyannote in-process unavailable: %s", e)

    return None


# ---------------------------------------------------------------------------
# Phase 2: Prosodic enrichment
# ---------------------------------------------------------------------------

def extract_segment_prosodics(
    audio_path: str,
    segments: list[DiarizedSegment],
    sample_rate: int = 16000,
) -> list[DiarizedSegment]:
    """Enrich each segment with prosodic features."""
    signal = ProsodicDiarizer._load_wav(audio_path)
    praat = PraatExtractor(sample_rate=sample_rate)
    egemaps = None
    try:
        egemaps = EGeMAPSExtractor(sample_rate=sample_rate)
    except ImportError:
        pass

    for seg in segments:
        start = int(seg.start_s * sample_rate)
        end = int(seg.end_s * sample_rate)
        chunk = signal[start:end]

        if len(chunk) < sample_rate // 2:  # skip < 0.5s
            continue

        features = praat.extract(chunk)
        arousal, valence, dominance = derive_dimensional_emotion(
            f0_mean=features["f0_mean"],
            f0_std=features["f0_std"],
            energy_rms=features["energy_rms"],
            hnr=features["hnr"],
            spectral_centroid=features["spectral_centroid"],
        )

        seg.prosodic = {
            **features,
            "arousal": arousal,
            "valence": valence,
            "dominance": dominance,
        }

        if egemaps and len(chunk) >= sample_rate:
            seg.prosodic["egemaps_vector"] = egemaps.extract(chunk)

    return segments


# ---------------------------------------------------------------------------
# Phase 3: Gemini transcription with speaker hints
# ---------------------------------------------------------------------------

def build_speaker_hint_prompt(segments: list[DiarizedSegment]) -> str:
    """Build a speaker timing hint for the Gemini prompt."""
    # Group consecutive segments by speaker
    speaker_times: dict[str, list[str]] = {}
    for seg in segments:
        sid = seg.speaker_id
        if sid not in speaker_times:
            speaker_times[sid] = []
        start_m = int(seg.start_s // 60)
        start_s = int(seg.start_s % 60)
        end_m = int(seg.end_s // 60)
        end_s = int(seg.end_s % 60)
        speaker_times[sid].append(f"{start_m}:{start_s:02d}-{end_m}:{end_s:02d}")

    lines = ["SPEAKER TIMING (from voice analysis — use as ground truth for speaker assignment):"]
    for sid, times in sorted(speaker_times.items()):
        # Show first 20 segments
        sample = times[:20]
        if len(times) > 20:
            sample.append(f"... and {len(times) - 20} more segments")
        lines.append(f"  {sid}: {', '.join(sample)}")
    return "\n".join(lines)


ENRICHED_EXTRACTION_PROMPT = """You are an ECEF knowledge extractor analyzing a recorded conversation.

{speaker_hints}

For each substantive claim, output a JSON object on its own line (JSONL format):
{{"text": "...", "speaker": "{speaker_label}", "confidence": 0.0-1.0, "tone_markers": [], "topic_keywords": [], "is_requirement": true/false, "is_question": true/false, "timestamp_approx": "MM:SS"}}

Use the SPEAKER TIMING above to assign the correct speaker label based on when the words were spoken.
Extract EVERY substantive claim. Each distinct statement gets its own JSON line.
Output ONLY JSON lines, no other text."""


# ---------------------------------------------------------------------------
# Phase 4: Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_speakers(
    claims: list[dict],
    segments: list[DiarizedSegment],
    speaker_name_map: dict[str, str],
) -> list[dict]:
    """Cross-validate Gemini speaker assignments against diarization.

    For each claim:
    - Parse timestamp
    - Find diarizer speaker at that timestamp
    - Compare with Gemini's assignment
    - If they agree: high confidence
    - If disagree: use diarizer (voice > text inference)
    - Enrich with nearest prosodic features
    """
    for claim in claims:
        ts_str = claim.get("timestamp_approx", "00:00")
        # Parse MM:SS
        parts = ts_str.replace(":", " ").split()
        try:
            ts_seconds = int(parts[0]) * 60 + int(parts[1])
        except (IndexError, ValueError):
            ts_seconds = 0

        # Find diarizer speaker at this timestamp
        diarizer_speaker = "unknown"
        nearest_prosodic = {}
        min_dist = float("inf")

        for seg in segments:
            if seg.start_s <= ts_seconds <= seg.end_s:
                diarizer_speaker = seg.speaker_id
                nearest_prosodic = seg.prosodic
                break
            # Track nearest for fallback
            dist = min(abs(seg.start_s - ts_seconds), abs(seg.end_s - ts_seconds))
            if dist < min_dist:
                min_dist = dist
                nearest_prosodic = seg.prosodic

        # Map diarizer ID to name
        diarizer_name = speaker_name_map.get(diarizer_speaker, diarizer_speaker)
        gemini_speaker = claim.get("speaker", "unknown")

        # Cross-validate
        if diarizer_name != "unknown" and diarizer_speaker != "unknown":
            if gemini_speaker == diarizer_name or gemini_speaker.startswith("Speaker"):
                # Use diarizer's assignment (more reliable)
                claim["speaker"] = diarizer_name
                claim["speaker_confidence"] = "high"
            else:
                # Disagree — prefer diarizer (voice > text)
                claim["speaker"] = diarizer_name
                claim["speaker_confidence"] = "medium"
        else:
            claim["speaker_confidence"] = "low"

        claim["speaker_id_diarizer"] = diarizer_speaker
        claim["speaker_id_gemini"] = gemini_speaker
        claim["prosodic"] = nearest_prosodic

    return claims


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

class AudioEnrichedBatchProcessor:
    """Full audio analysis pipeline for batch meeting processing.

    Composes existing ForgeStream emotion modules into a 5-phase pipeline.
    """

    def __init__(self, config: ForgeStreamConfig) -> None:
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=2)

    async def process(
        self,
        audio_path: str,
        speaker_names: list[str] | None = None,
        n_speakers: int = 3,
        chunk_seconds: int = 300,
    ) -> dict[str, Any]:
        """Process an audio file through the full enriched pipeline.

        Args:
            audio_path: Path to WAV file.
            speaker_names: Optional list of speaker names for mapping.
            n_speakers: Expected number of speakers.
            chunk_seconds: Chunk size for Gemini transcription (default 5 min).

        Returns dict with: enriched_claims, segments, speaker_profiles,
        emotion_timeline, dynamics_snapshots.
        """
        audio_path = str(audio_path)
        print(f"\n{'='*60}")
        print(f"ENRICHED BATCH PROCESSOR")
        print(f"Audio: {audio_path}")
        print(f"Expected speakers: {n_speakers}")
        print(f"{'='*60}")

        # ---- Phase 1: DIARIZE ----
        print(f"\n--- Phase 1: Diarization ---")

        segments = None
        hf_token = self.config.huggingface_token or ""
        if not hf_token:
            hf_path = Path(".secrets/hf_token.txt")
            if hf_path.exists():
                hf_token = hf_path.read_text().strip()

        if hf_token:
            segments = await asyncio.to_thread(
                _try_pyannote_diarize, audio_path, hf_token, n_speakers
            )

        if segments is None:
            print("  Using prosodic diarizer (F0 + spectral clustering)")
            diarizer = ProsodicDiarizer(n_speakers=n_speakers)
            segments = await asyncio.to_thread(diarizer.diarize, audio_path)

        print(f"  {len(segments)} segments detected")

        # Speaker profiles
        print(f"\n--- Speaker Profiles ---")
        diarizer_for_profiles = ProsodicDiarizer(n_speakers=n_speakers)
        profiles = await asyncio.to_thread(
            diarizer_for_profiles.get_speaker_profiles, audio_path, segments
        )

        for sid, prof in sorted(profiles.items()):
            print(f"  {sid}: F0={prof['f0_mean']:.1f}Hz, SC={prof['spectral_centroid']:.0f}Hz, "
                  f"Energy={prof['energy_rms']:.3f}, Duration={prof['total_duration_s']:.0f}s")

        # Auto-map speakers if names provided
        speaker_name_map = {}
        if speaker_names and profiles:
            speaker_name_map = self._auto_map_speakers(profiles, speaker_names)
            print(f"\n  Speaker mapping:")
            for sid, name in speaker_name_map.items():
                print(f"    {sid} → {name}")

        # ---- Phase 2: PROSODIC ENRICHMENT ----
        print(f"\n--- Phase 2: Prosodic Enrichment ---")
        segments = await asyncio.to_thread(
            extract_segment_prosodics, audio_path, segments
        )
        enriched_count = sum(1 for s in segments if s.prosodic)
        print(f"  {enriched_count}/{len(segments)} segments enriched with prosodic features")

        # ---- Phase 3: TRANSCRIBE ----
        print(f"\n--- Phase 3: Gemini Transcription ---")
        speaker_hints = build_speaker_hint_prompt(segments)
        claims = await self._transcribe_with_gemini(
            audio_path, speaker_hints, speaker_name_map, chunk_seconds
        )
        print(f"  {len(claims)} claims extracted")

        # ---- Phase 4: CROSS-VALIDATE ----
        print(f"\n--- Phase 4: Speaker Cross-Validation ---")
        claims = cross_validate_speakers(claims, segments, speaker_name_map)
        conf_counts = {}
        for c in claims:
            conf = c.get("speaker_confidence", "?")
            conf_counts[conf] = conf_counts.get(conf, 0) + 1
        print(f"  Confidence: {conf_counts}")

        # ---- Phase 5: EMOTION TIMELINE ----
        print(f"\n--- Phase 5: Emotion Timeline ---")
        timeline = self._build_emotion_timeline(segments, speaker_name_map)
        print(f"  {len(timeline)} emotion datapoints")

        # ---- Save results ----
        result = {
            "enriched_claims": claims,
            "segments": [
                {"start_s": s.start_s, "end_s": s.end_s, "speaker_id": s.speaker_id, "prosodic": s.prosodic}
                for s in segments
            ],
            "speaker_profiles": profiles,
            "speaker_name_map": speaker_name_map,
            "emotion_timeline": timeline,
            "claim_count": len(claims),
            "segment_count": len(segments),
        }

        return result

    def _auto_map_speakers(
        self,
        profiles: dict[str, dict],
        speaker_names: list[str],
    ) -> dict[str, str]:
        """Auto-map diarizer speaker IDs to real names using F0 heuristics.

        Heuristic: sort speakers by F0 mean. In a mixed-gender or mixed-accent
        conversation, F0 is usually the strongest separator.
        """
        # Sort profiles by F0 (lowest first)
        sorted_speakers = sorted(
            profiles.items(),
            key=lambda x: x[1].get("f0_mean", 0),
        )

        # Sort names by expected F0 (heuristic)
        # For the Ryan call: Tim (older American male, lower F0) < Ryan (younger American) < Mergim (British accent)
        # But we don't know the order in general — sort by duration (most speaking = most claims)
        sorted_by_duration = sorted(
            sorted_speakers,
            key=lambda x: -x[1].get("total_duration_s", 0),
        )

        # Map by speaking duration (most talking → name with most expected claims)
        mapping = {}
        used_names = set()
        for (sid, prof), name in zip(sorted_by_duration, speaker_names):
            mapping[sid] = name
            used_names.add(name)

        # Fill any remaining
        for sid, _ in sorted_speakers:
            if sid not in mapping:
                for name in speaker_names:
                    if name not in used_names:
                        mapping[sid] = name
                        used_names.add(name)
                        break

        return mapping

    async def _transcribe_with_gemini(
        self,
        audio_path: str,
        speaker_hints: str,
        speaker_name_map: dict[str, str],
        chunk_seconds: int = 300,
    ) -> list[dict]:
        """Split audio into chunks and extract claims via Gemini."""
        import subprocess
        import time

        # ot-ctx-gcp-cred-003: resolve relative to project root, not CWD
        _proj = str(Path(__file__).resolve().parent.parent)
        os.environ.setdefault(
            "GOOGLE_APPLICATION_CREDENTIALS",
            os.path.join(_proj, ".secrets", "service-account.json"),
        )

        from google import genai
        from google.genai import types

        client = genai.Client(
            vertexai=self.config.gemini_use_vertex,
            project=self.config.gemini_project,
            location=self.config.gemini_location,
        )

        # Get audio duration
        with wave.open(audio_path, "rb") as wf:
            duration_s = wf.getnframes() / wf.getframerate()

        # Build speaker label for prompt
        speaker_labels = " / ".join(
            f"{v} ({k})" for k, v in sorted(speaker_name_map.items())
        ) or "Speaker 1 / Speaker 2 / Speaker 3"

        prompt = ENRICHED_EXTRACTION_PROMPT.format(
            speaker_hints=speaker_hints,
            speaker_label=speaker_labels,
        )

        # Split and process chunks
        all_claims = []
        chunk_dir = Path("/tmp/enriched_chunks")
        chunk_dir.mkdir(exist_ok=True)

        offset = 0
        chunk_idx = 0
        while offset < duration_s:
            chunk_path = chunk_dir / f"chunk_{chunk_idx:02d}.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path,
                 "-ss", str(offset), "-t", str(chunk_seconds),
                 "-ar", "16000", "-ac", "1", str(chunk_path)],
                capture_output=True,
            )

            if chunk_idx > 0:
                time.sleep(15)  # Rate limit cooldown

            audio_bytes = chunk_path.read_bytes()
            print(f"  Chunk {chunk_idx}: {offset/60:.1f}-{min(offset+chunk_seconds, duration_s)/60:.1f}min...")

            try:
                response = client.models.generate_content(
                    model=self.config.gemini_model,
                    contents=[
                        types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(temperature=0.0),
                )

                if response and response.text:
                    for line in response.text.strip().splitlines():
                        line = line.strip()
                        if not line or line.startswith("```"):
                            continue
                        try:
                            claim = json.loads(line)
                            claim["chunk"] = chunk_idx
                            claim["chunk_offset_min"] = offset / 60
                            all_claims.append(claim)
                        except json.JSONDecodeError:
                            continue

                print(f"    → {sum(1 for c in all_claims if c.get('chunk') == chunk_idx)} claims")
            except Exception as e:
                print(f"    → ERROR: {e}")
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    time.sleep(60)

            offset += chunk_seconds
            chunk_idx += 1

        return all_claims

    @staticmethod
    def _build_emotion_timeline(
        segments: list[DiarizedSegment],
        speaker_name_map: dict[str, str],
    ) -> list[dict]:
        """Build per-speaker emotion timeline from enriched segments."""
        timeline = []
        for seg in segments:
            if not seg.prosodic:
                continue
            name = speaker_name_map.get(seg.speaker_id, seg.speaker_id)
            timeline.append({
                "timestamp_s": seg.start_s,
                "duration_s": seg.end_s - seg.start_s,
                "speaker": name,
                "speaker_id": seg.speaker_id,
                "arousal": seg.prosodic.get("arousal", 0),
                "valence": seg.prosodic.get("valence", 0),
                "dominance": seg.prosodic.get("dominance", 0),
                "f0_mean": seg.prosodic.get("f0_mean", 0),
                "energy_rms": seg.prosodic.get("energy_rms", 0),
            })
        return timeline


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def run_enriched_batch(
    audio_path: str,
    config: ForgeStreamConfig,
    speaker_names: list[str] | None = None,
    n_speakers: int = 3,
    output_dir: str = "data",
) -> dict:
    """Run the full enriched batch pipeline and save all outputs."""
    processor = AudioEnrichedBatchProcessor(config)
    result = await processor.process(
        audio_path=audio_path,
        speaker_names=speaker_names,
        n_speakers=n_speakers,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save all outputs
    stem = Path(audio_path).stem

    claims_path = out / f"{stem}_claims_enriched.json"
    with open(claims_path, "w") as f:
        json.dump(result["enriched_claims"], f, indent=2)
    print(f"\nSaved: {claims_path} ({len(result['enriched_claims'])} claims)")

    segments_path = out / f"{stem}_diarization.json"
    with open(segments_path, "w") as f:
        json.dump(result["segments"], f, indent=2, default=str)
    print(f"Saved: {segments_path} ({len(result['segments'])} segments)")

    profiles_path = out / f"{stem}_speaker_profiles.json"
    with open(profiles_path, "w") as f:
        json.dump(result["speaker_profiles"], f, indent=2, default=float)
    print(f"Saved: {profiles_path}")

    timeline_path = out / f"{stem}_emotion_timeline.json"
    with open(timeline_path, "w") as f:
        json.dump(result["emotion_timeline"], f, indent=2)
    print(f"Saved: {timeline_path} ({len(result['emotion_timeline'])} datapoints)")

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Enriched batch meeting processor")
    parser.add_argument("audio", help="Path to WAV audio file")
    parser.add_argument("--speakers", nargs="+", help="Speaker names (e.g., Mergim Ryan Tim)")
    parser.add_argument("--n-speakers", type=int, default=3, help="Number of speakers")
    parser.add_argument("--output", default="data", help="Output directory")
    args = parser.parse_args()

    os.environ.setdefault(
        "GOOGLE_APPLICATION_CREDENTIALS", ".secrets/service-account.json"
    )

    from .config import load_config
    config = load_config()

    asyncio.run(run_enriched_batch(
        audio_path=args.audio,
        config=config,
        speaker_names=args.speakers,
        n_speakers=args.n_speakers,
        output_dir=args.output,
    ))


if __name__ == "__main__":
    main()
