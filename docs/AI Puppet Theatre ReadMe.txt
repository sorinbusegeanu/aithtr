**AI Puppet Theatre** is a fully scriptable pipeline that generates animated theatre scenes from text.
It converts multi-character dialogue into voiced, lip-synced avatar performances on a static stage.

**Overview**

* Each character is an AI “actor” with a voice and avatar.
* A script (or LLM) produces dialogue lines.
* Audio is synthesized per character.
* Talking-head video is generated per line.
* Clips are stitched into a final scene video.

**Pipeline**

```
Dialogue (script or LLM)
→ Text-to-Speech (per character)
→ Avatar video generation (lip sync + idle motion)
→ Scene compositor (background + cuts)
→ MP4 output
```

**Features**

* 100% automated, no manual animation.
* Deterministic (same script = same video).
* Supports multiple characters and scenes.
* Fixed camera, minimal motion (face + upper body).
* Suitable for episodic YouTube content.

**Use cases**

* AI-generated plays
* Improvised character dialogues
* Educational or comedic skits
* Narrative experiments with agent characters

**Design constraints**

* Static backgrounds
* Limited gestures
* Turn-based speaking
* Audio-driven animation only

**Output**

* One rendered MP4 per scene or episode.
