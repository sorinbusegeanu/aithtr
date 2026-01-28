# Agentcast.md

Stricter “agents as specialists” for a fully-automated **text → episode.mp4** pipeline (daily ~5 minutes), with **human-in-the-loop critique**.

## 0) Shared interfaces (contracts)

All agents:
- **Input**: JSON (validated against a schema)
- **Output**: JSON (validated) + referenced artifacts (stored by Artifact Store)
- **No side effects** except via MCP tools

Canonical artifacts:
- `screenplay.json` (locked when approved)
- `scene_plan.json` (per-scene staging)
- `cast_bible.json` (persistent)
- `performance/*.wav` + `performance/*.mp4`
- `timeline.json` (edit decision list)
- `episode.mp4` + `preview.mp4`
- `critic_report.json`

## 1) Orchestrator Agent (Supervisor)

**Goal**: Run the episode build graph deterministically; enforce schemas; manage retries; gate with human critique.

**Inputs**
- `episode_seed` (theme, mood, constraints, date)
- `series_bible_id` (points to persistent memory)
- `runtime_config` (tool endpoints, budgets)

**Outputs**
- `episode_manifest.json` (IDs/paths for every artifact; status per step)

**Responsibilities**
- DAG execution (state machine)
- Human checkpoints:
  - Gate A: screenplay approval
  - Gate B: preview approval
- Automatic QC gating:
  - duration bounds, missing lines, missing assets, audio peak/loudness checks
- Caching: if an input hash matches, reuse artifacts

**MCP tools required**
- `memory.*`
- `artifact.*`
- `qc.*`
- `render.*`

---

## 2) Showrunner Agent (High-level episode intent)

**Goal**: Decide episode “why/what”: premise, emotional arc, constraints, and target duration.

**Inputs**
- `yesterday_summary` (from memory)
- `series_bible`
- optional `prompt_seed`

**Outputs**
- `episode_brief.json`:
  - `premise`
  - `beats[]` (time budget per beat)
  - `tone` (funny/sad, intensity)
  - `cast_constraints` (who must appear, max cast per scene)

**Hard constraints**
- 5:00 ± 15s target total duration
- max scenes (e.g., 3–6), max cast per scene (e.g., 2–3)

**MCP tools**
- `memory.retrieve`
- `memory.store`

---

## 3) Writer Agent (Dialogue + stage directions)

**Goal**: Produce a screenplay that is renderable and timed.

**Inputs**
- `episode_brief.json`
- `series_bible` + `cast_bible`
- `style_guide` (format constraints)

**Outputs**
- `screenplay_draft.json`:
  - scenes[]:
    - `scene_id`
    - `setting_prompt` (short)
    - `characters[]`
    - `lines[]`: `{speaker, text, emotion, pause_ms_after, sfx_tag?}`
- `risk_flags.json` (if needed: profanity, sensitive topics)

**Rules**
- Turn-based speaking by default (no overlap)
- Each line ≤ N seconds (configurable)
- Minimize “camera choreography” (leave to Director)

**MCP tools**
- `memory.retrieve`
- optional `qc.text_policy` (local rules)

---

## 4) Dramaturg Agent (Structure + continuity)

**Goal**: Enforce arc, continuity, and pacing; generate revision notes.

**Inputs**
- `screenplay_draft.json`
- `episode_brief.json`
- `series_bible` + prior episodes summaries

**Outputs**
- `screenplay_revision_notes.json`:
  - required edits (timing, continuity, clarity)
  - suggested cuts/additions to hit duration

**MCP tools**
- `memory.retrieve`

---

## 5) Human Critic Gate A (Screenplay approval)

**Goal**: Approve or request minimal targeted changes before expensive generation.

**Inputs**
- `screenplay_draft.json`
- `dramaturg_notes.json`

**Outputs**
- `screenplay_locked.json` (approved) **or**
- `critique_notes.json` (requested edits)

**Storage**
- Critique is written to memory for learning.

**MCP tools**
- `memory.store`
- `artifact.annotate`

---

## 6) Casting Director Agent (Cast selection + voice/visual binding)

**Goal**: Map roles → stable character identities (voice + avatar), preserving consistency across episodes.

**Inputs**
- `screenplay_locked.json`
- `cast_bible.json` (persistent)
- available assets list

**Outputs**
- `cast_plan.json`:
  - role → `character_id`
  - `voice_id`
  - `avatar_id`
  - default `emotion_map` (emotion → TTS style params)
- updated `cast_bible.json` (if new character is introduced)

**MCP tools**
- `asset.list` / `asset.get`
- `memory.retrieve` / `memory.store`

---

## 7) Scene Designer Agent (Background + props)

**Goal**: Produce consistent scene visuals from a bounded library (fast) or generator (optional).

**Inputs**
- `screenplay_locked.json`
- `episode_brief.json`
- `asset_catalog`

**Outputs**
- `scene_assets.json`:
  - per scene:
    - `background_asset_id`
    - `props[]` (optional overlays)
    - `layout_hints` (safe zones for subtitles/actors)

**MCP tools**
- `asset.search`
- optional `imagegen.generate_background` (if enabled)

---

## 8) Director Agent (Blocking + shot plan)

**Goal**: Decide who is on screen, where, and when (layout + timing).

**Inputs**
- `screenplay_locked.json`
- `cast_plan.json`
- `scene_assets.json`

**Outputs**
- `scene_plan.json`:
  - per scene:
    - stage layout (positions/scale)
    - entrance/exit timing per character
    - reaction shots / idle moments
    - subtitle placement

**MCP tools**
- `qc.plan_validate` (e.g., no overlaps if disallowed)

---

## 9) Voice Actor Agent (per character)

One agent instance per character.

**Goal**: Generate voice lines deterministically (TTS), producing one WAV per line.

**Inputs**
- `character_bundle.json`:
  - `character_id`, `voice_id`
  - assigned lines: `{line_id, text, emotion, timing_hint}`
  - `emotion_map`

**Outputs**
- `performance_audio_manifest.json`:
  - `line_id → wav_uri` (+ measured duration)

**MCP tools**
- `tts.synthesize`
- `artifact.put`

---

## 10) Performance Animator Agent (per character)

One agent instance per character.

**Goal**: Convert each line WAV → talking-head clip with subtle motion.

**Inputs**
- `avatar_id`
- `line_id → wav_uri`
- style params (still / subtle / expressive)

**Outputs**
- `performance_video_manifest.json`:
  - `line_id → clip_uri` (mp4/webm, alpha if supported)

**MCP tools**
- `lipsync.render_clip`
- optional `enhance.face_restore`

---

## 11) Foley & Music Agent

**Goal**: Select/generate SFX/music cues from a local library.

**Inputs**
- `screenplay_locked.json` (sfx tags)
- `episode_brief.json`

**Outputs**
- `audio_cues.json` (time-aligned events)
- referenced audio assets

**MCP tools**
- `asset.search_audio`
- `audio.trim` / `audio.mix` (optional)

---

## 12) Editor Agent (Timeline / EDL)

**Goal**: Convert screenplay + scene_plan + performances into a final edit decision list.

**Inputs**
- `screenplay_locked.json`
- `scene_plan.json`
- `performance_audio_manifest.json`
- `performance_video_manifest.json`
- `audio_cues.json`

**Outputs**
- `timeline.json`:
  - ordered scenes with absolute times
  - layers (background, actors, props, captions)
  - audio tracks (dialogue + SFX + music)
  - transitions

**MCP tools**
- `qc.timeline_validate`

---

## 13) Renderer Agent (Final + preview)

**Goal**: Render `timeline.json` into MP4.

**Inputs**
- `timeline.json`
- referenced assets

**Outputs**
- `preview.mp4` (low-res fast encode)
- `episode.mp4` (final encode)

**MCP tools**
- `render.preview`
- `render.final`

---

## 14) QC Agent (Automated quality gates)

**Goal**: Detect obvious failures before human review.

**Inputs**
- `preview.mp4` / `episode.mp4`
- `timeline.json`

**Outputs**
- `qc_report.json`:
  - duration, silence gaps, clipping, missing captions, black frames, etc.

**MCP tools**
- `qc.audio`
- `qc.video`
- `qc.subtitles`

---

## 15) Human Critic Gate B (Preview approval)

**Goal**: Approve final or request specific revisions.

**Inputs**
- `preview.mp4`
- `qc_report.json`

**Outputs**
- `approval.json` or `revision_request.json`

**Learning**
- Store critique + outcomes to memory for improving script, pacing, and casting.

**MCP tools**
- `memory.store`

---

## 16) Learning / Memory Curator Agent

**Goal**: Convert outcomes into training data and retrieval memories.

**Inputs**
- approved `episode_manifest.json`
- critiques + qc reports + engagement metrics (if available)

**Outputs**
- Memory updates:
  - `series_bible` updates (rules, recurring jokes, character traits)
  - embeddings for retrieval
  - labeled failure cases (“too long”, “flat pacing”, “voice mismatch”)

**MCP tools**
- `memory.store`
- `memory.index`
