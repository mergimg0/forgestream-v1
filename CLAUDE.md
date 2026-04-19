# ForgeStream — CLAUDE.md

## Architecture
- Append-only ECEF event log is the SINGLE coordination mechanism. All components read/write events.
- SOS Governor is a POST-WRITE OBSERVER (immune system), not a pre-write gate. Never reject valid data.
- Evaluator function is pluggable — user has a future custom RL equation planned. Keep the interface open.
- Orchestrator is thin: validate → write PostgreSQL → sync Firestore → publish EventBus. No business logic.
- SynthesisEngine subscribes to EventBus and emits derived events (requirements, contradictions, seeds). It ignores self-authored events (`author="synthesis_engine"`) to prevent infinite loops.

## Gemini Live API
- Does NOT work via Vertex AI. Use GCP API key + raw WebSocket.
- Model: `gemini-2.5-flash-native-audio-latest` (as of April 2026; old `gemini-3.1-flash-live-preview` was deprecated — Google renamed Live API models to `*-native-audio-*` scheme)
- Override via env: `FORGESTREAM_GEMINI_MODEL=<model-name>`
- Endpoint: `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={API_KEY}`
- Response modalities MUST include AUDIO (TEXT-only causes 1011 internal error)
- Send audio via `realtimeInput.audio` with base64 + `audio/pcm;rate=16000`
- Send 2s of silence (zero-byte PCM) after speech to trigger VAD end-of-speech
- SSL: use `ssl.create_default_context(cafile=certifi.where())` on macOS
- Setup message uses `setup.model` and `setup.generationConfig`, NOT `config`
- Use `send_realtime_input()` with SDK, or `realtimeInput` JSON key with raw WebSocket
- Transcriptions arrive as `serverContent.inputTranscription` (user) and `serverContent.outputTranscription` (model)

## Gemini Batch API (Vertex AI)
- Works via service account at `.secrets/service-account.json`
- Set `GOOGLE_APPLICATION_CREDENTIALS=.secrets/service-account.json`
- 15s cooldown between audio file API calls to avoid 429 rate limits
- Retry with 60s/120s/180s backoff on RESOURCE_EXHAUSTED
- m4a files: use `audio/mp4` mime type (not `audio/m4a`)

## Testing
- `python3 -m pytest -q` — full suite (needs Docker PostgreSQL running)
- Without DB: `python3 -m pytest -q --ignore=tests/events/test_store.py --ignore=tests/events/test_subscribe.py -k "not writes_to_store and not milestone_a and not full_pipeline_with"`
- PostgreSQL: Docker container `continuous-claude-postgres`, database `forgestream`
- Always run full suite before committing — regressions compound fast across 500+ tests

## Textual TUI
- `RichLog.markup` defaults to False. Always pass `markup=True` in FeedPanel constructor.
- EventBus.subscribe() deduplicates — textual can call on_mount multiple times.
- Background processes buffer stdout. Use `flush=True` or `python3 -u`.
- TUI panels receive events via `on_event_received()` called from app's `_on_event()` EventBus handler.
- The TUI app is the entry point in runner mode. Orchestrator tasks run inside textual's async loop via `run_worker()`.

## Firestore
- Append-only security rules deployed: `allow create: if true; allow update, delete: if false;`
- Dashboard reads from Firestore via `create_live_app(config)` — see `dashboard/launcher.py`
- Firebase project is the same GCP project: `forgestream-ai`
- `firebase-admin` initializes with ApplicationDefault credentials. Check `get_app()` first to avoid re-initialization errors.

## Key Paths
- Design spec: `~/projects/proofforge/docs/superpowers/specs/2026-03-27-forgestream-design.md`
- Milestone B spec: `~/projects/proofforge/docs/superpowers/specs/2026-03-27-milestone-b-full-autonomy-design.md`
- Handoff: `docs/handoff-next-session.md`
- API key (Live): in GCP Console, restricted to generativelanguage.googleapis.com
- Service account (Vertex): `.secrets/service-account.json` (gitignored)
- GCP project: `forgestream-ai`, region: `europe-west2`
- Credits: £739 GenAI, valid until 2027-03-11

## Branch Detection
- Jaccard distance threshold: 0.9 (in `synthesis/branches.py`)
- Lower values create excessive branches from natural conversation drift
- Batch mode needs baseline accumulation before branch detection starts — currently every claim triggers drift check against growing centroid

## Trust Region
- Initial epsilon is 0.525 (not 0.3) because sigmoid(0)=0.5, so competence_multiplier starts at 1.75
- This means initial spawn limits are 3 research / 4 scaffold agents, not 2/2
- Trust region state persisted via TrustRegion.save()/load() to data/trust_region.json
- record_meeting_result() accepts optional rapport_trend param for trust boost

## Audio Sources
- All three sources yield identical 16KB PCM chunks (16kHz mono, 0.5s per chunk)
- FileReplaySource handles m4a/mp3 via pydub fallback (requires ffmpeg: `brew install ffmpeg`)
- MicrophoneSource uses sounddevice callback → queue → async generator pattern
- SystemAudioSource extends MicrophoneSource, auto-detects BlackHole device
- BlackHole installed but needs Mac restart to activate

## Emotion Pipeline
- `forgestream/emotion/` — AudioRingBuffer, EmotionExtractor, PraatExtractor, EGeMAPSExtractor, EmotionCorrelator, GroupDynamicsEngine, RapportEngine, DisengagementDetector, CRQAComputeRouter, EmotionCorpus
- EmotionExtractor runs in a ThreadPoolExecutor (CPU-bound Parselmouth/openSMILE off event loop)
- All emotion data flows as ECEF events: PROSODIC_FEATURE, EMOTION_STATE, ENTRAINMENT_SNAPSHOT, RAPPORT_SCORE, PROOF_OBLIGATION
- EventBus wiring: use `orchestrator.attach_*()` pattern (attach_emotion_correlator, attach_dynamics_engine, attach_rapport_engine, attach_proof_detector, attach_contradiction_resolver)
- Emotion is opt-out: `config.emotion_enabled=True` by default. Diarization is opt-in: `config.diarization_enabled=False`
- openSMILE eGeMAPS returns 88 features. Parselmouth gives F0/jitter/shimmer/HNR. Both accept int16 numpy arrays at 16kHz.

## Rapport Tracking
- 4-component model from Tickle-Degnen (1990): attentiveness, positivity, coordination, symmetry
- Weights interpolate via sigmoid based on meeting_count (early=positivity-heavy, established=coordination-heavy)
- Coordination uses CRQA %DET via RunPod GPU endpoint (circuit breaker: 3-fail open, 5-cycle skip, auto-recovery)
- Disengagement detection: energy↓ + F0 flattening + one-sided → damping factor 0.3 applied to rapport composite
- CRITICAL: convergence ≠ rapport. Asymmetry (who converges to whom) is the strongest signal. One-way convergence = power dynamic, not rapport.
- RunPod CRQA endpoint: `https://yvbcstitpla5i6-8000.proxy.runpod.net` (RTX A4000). SSH: `<podHostId>@ssh.runpod.io` with -tt flag required.

## GRPO Pattern (Reused Everywhere)
- Same algorithm in 5 places: evaluator weights, tone adjustments, rapport weights, config params, prompt params
- Pattern: generate 10 Gaussian perturbations → score each retroactively → blend best 70/30 with current → normalize
- WeightTuner.tune() and tune_multi_objective() in governor/improvement.py
- All GRPO results persisted to data/ as JSON. Loaded at meeting startup via PostMeetingSynthesis.

## PostMeetingSynthesis Pipeline
- PostMeetingSynthesis.run() is THE integration point. Currently calls (in order):
  1. generate_report → save_report
  2. tune_weights → save_weights (evaluator GRPO)
  3. save corpus (audio WAV + feature index)
  4. tune tone adjustments
  5. tune rapport weights
  6. update user profile
  7. performance analysis + config tuning + arch report
  8. export proof obligations
- GeminiLiveStream.stop(run_post_meeting=True, human_score=None) invokes the full pipeline
- GeminiLiveStream.__init__ loads persisted weights and meeting_count at startup (GRPO loop continuity)

## Optional Dependencies
- `emotion`: opensmile, praat-parselmouth, numpy (always install for emotion pipeline)
- `emotion-ml`: funasr, torch (SenseVoice — plan written but not yet implemented)
- `diarization`: pyannote.audio, torch, torchaudio (opt-in, requires HF token)
- Lazy loading pattern: check availability in __init__, load model on first use, log if unavailable

## Secrets
- `.secrets/service-account.json` — GCP Vertex AI
- `.secrets/hf_token.txt` — HuggingFace (pyannote speaker diarization)
- `.secrets/runpod_api_key.txt` — RunPod API
- `.secrets/runpod_endpoint.txt` — Live CRQA endpoint URL
- All gitignored. Never commit.

## Dashboard
- 10 D3.js panels: knowledge graph, evaluator trajectory, meeting timeline, emotion timeline, entrainment heatmap, rapport trajectory, branch tree, seed garden, SOS convergence, contradictions, proof queue
- All panels auto-refresh every 5 seconds via polling
- D3 v7 from CDN, no build step. Each panel is a JS class with fetch() → render() → update()
- API endpoints at /api/*. Static files mounted at /static via FastAPI StaticFiles

## Common Pitfalls
- psycopg3 dynamic SQL: use `psycopg.sql.SQL()` for query composition, not f-strings (Pyright LiteralString)
- `cur.description` can be None — always guard before iterating
- Tethered mobile connections drop SSL to `oauth2.googleapis.com` — use WiFi for Google API calls
- Gemini claim extraction is non-deterministic — same audio gives different claim counts each run
- `pip` doesn't exist on this machine — use `python3 -m pip`
