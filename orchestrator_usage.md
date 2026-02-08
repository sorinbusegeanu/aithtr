# Orchestrator Usage Guide

This document describes the current orchestrator behavior, CLI parameters, and practical workflows for:
- test/inference runs
- dataset generation for modules
- training handoff
- memory/RAG usage

## 1) Which orchestrator entrypoint to use

Use the full orchestrator here:

`python -m src.orchestrator.run_orchestrator ...`

Do not use `src/orchestrator/cli.py` for production pipeline runs. `src/orchestrator/cli.py` calls a scaffold pass-through (`src/orchestrator/pipeline.py`).

## 2) What the orchestrator does

`src/orchestrator/run_orchestrator.py` runs an iterative ARC pipeline:

1. Load tasks from `--batch-file`.
2. Run Describer to produce facts.
3. Run Teacher Ruler (and optionally Student/Enhanced Ruler).
4. Apply + verify candidates.
5. Compute canonical comparator-based `feature_vec` and `verifier_score`.
6. Run deterministic critic scoring and CSA analysis.
7. Select best/seed candidates and iterate.
8. Emit task outputs, metrics, optional traces, and optional training buffers.
9. Optionally write/read memory (SQLite store + retrieval bundles).

Defaults keep behavior close to single pass:
- `--loop-iters 1`
- `--candidate-k 1`
- `--edit-k 0`

## 3) CLI parameters

## 3.1 Required run arguments

- `--batch-file` Input JSONL batch.
- `--out-jsonl` Output per-task result JSONL.
- `--describer-checkpoint` Describer adapter/checkpoint.
- `--ruler-checkpoint` Teacher Ruler adapter/checkpoint.

## 3.2 Optional model arguments

- `--enhanced-ruler-checkpoint` Student (enhanced ruler) checkpoint.
- `--critic-checkpoint` Critic checkpoint metadata path.

## 3.3 Iteration and search controls

- `--loop-iters` Number of loop iterations.
- `--candidate-k` Draft count in initial iteration.
- `--edit-k` Candidates in edit iterations.
- `--edit-top-m` Number of seeds retained for edit selection.
- `--iters` Alias for `--loop-iters` (batch mode).
- `--K-drafts` Alias for `--candidate-k` (batch mode).

## 3.4 Memory/RAG controls

- `--memory-k` Retrieval size for generic memory summary.
- `--memory-k-episodes` Episodes for critic/ruler retrieval.
- `--memory-k-cases` Cases for critic retrieval.
- `--memory-k-solved-examples` Solved examples for ruler conditioning.
- `--memory-k-repair-cases` Repair cases for ruler conditioning.
- `--memory-max-bytes` Max serialized payload size for retrieved bundles.
- `--train-namespace` Memory namespace isolation for training data generation.

## 3.5 Output controls

- `--metrics-json` Optional metrics JSON file.
- `--trace-jsonl` Optional per-task trace JSONL.
- `--critic-buffer-jsonl` Optional critic training buffer JSONL.
- `--ruler-student-buffer-jsonl` Optional enhanced-ruler transition buffer JSONL.
- `--out-dir` Batch output directory (trace bundle + dataset builders).

## 3.6 Batch/dataset mode switches

- `--critic-batch-run` Run and write `out-dir/traces/*.json` plus `out-dir/index.jsonl`.
- `--critic-build-ranking-ds` Build ranking dataset from saved traces.
- `--critic-directive-sweep` Build directive preference dataset from saved traces.
- `--critic-build-blame-ds` Build blame-locus dataset from saved traces.
- `--failed-only` Restrict dataset builds to failed tasks.
- `--M-seeds` Seeds per iteration for directive preference builder.
- `--J-edits` Edit candidates considered per seed for preference builder.
- `--delta-threshold` Minimum score delta for preference pairs.

## 3.7 General controls

- `--max-tasks` Cap task count.
- `--fail-fast` Stop after first failure.
- `--seed` Global seed.
- `--log-level` Log level.
- `--log-file` Optional log path.

## 4) Inference / test usage

Minimal single-pass run:

```bash
python -m src.orchestrator.run_orchestrator \
  --batch-file data/batch.jsonl \
  --out-jsonl out/tasks.jsonl \
  --describer-checkpoint <describer_ckpt> \
  --ruler-checkpoint <ruler_ckpt>
```

Iterative teacher+student run with traces/buffers:

```bash
python -m src.orchestrator.run_orchestrator \
  --batch-file data/batch.jsonl \
  --out-jsonl out/tasks.jsonl \
  --describer-checkpoint <describer_ckpt> \
  --ruler-checkpoint <teacher_ckpt> \
  --enhanced-ruler-checkpoint <student_ckpt> \
  --loop-iters 3 --candidate-k 4 --edit-k 4 --edit-top-m 2 \
  --trace-jsonl out/trace.jsonl \
  --critic-buffer-jsonl out/critic_pairs.jsonl \
  --ruler-student-buffer-jsonl out/student_transitions.jsonl \
  --seed 0
```

## 5) Dataset generation workflows

## 5.1 Pass 1: batch traces

```bash
python -m src.orchestrator.run_orchestrator \
  --batch-file data/curriculum_shard.jsonl \
  --out-jsonl out/run_tasks.jsonl \
  --describer-checkpoint <describer_ckpt> \
  --ruler-checkpoint <teacher_ckpt> \
  --enhanced-ruler-checkpoint <student_ckpt> \
  --critic-batch-run --out-dir out/batch \
  --loop-iters 3 --candidate-k 4 --edit-k 4 --edit-top-m 2 \
  --max-tasks 10000 --seed 0
```

Outputs include:
- `out/batch/traces/task_<id>.json`
- `out/batch/index.jsonl`

## 5.2 Critic datasets from traces

Ranking pairs:

```bash
python -m src.orchestrator.run_orchestrator \
  --batch-file data/curriculum_shard.jsonl \
  --out-jsonl out/tmp.jsonl \
  --describer-checkpoint <describer_ckpt> \
  --ruler-checkpoint <teacher_ckpt> \
  --critic-build-ranking-ds --failed-only --out-dir out/batch
```

Blame locus:

```bash
python -m src.orchestrator.run_orchestrator \
  --batch-file data/curriculum_shard.jsonl \
  --out-jsonl out/tmp.jsonl \
  --describer-checkpoint <describer_ckpt> \
  --ruler-checkpoint <teacher_ckpt> \
  --critic-build-blame-ds --failed-only --out-dir out/batch
```

Directive preferences:

```bash
python -m src.orchestrator.run_orchestrator \
  --batch-file data/curriculum_shard.jsonl \
  --out-jsonl out/tmp.jsonl \
  --describer-checkpoint <describer_ckpt> \
  --ruler-checkpoint <teacher_ckpt> \
  --critic-directive-sweep --failed-only \
  --M-seeds 2 --J-edits 4 --delta-threshold 1e-6 \
  --out-dir out/batch
```

Current files produced:
- `out/batch/critic_ranking_pairs.jsonl`
- `out/batch/critic_blame_locus.jsonl`
- `out/batch/critic_directive_prefs.jsonl`

## 5.3 Describer/Ruler training data

The orchestrator itself does not export full describer/ruler supervised datasets directly. Typical flow:

1. Use synthetic builders under `src/utils/` for module-specific train/val files.
2. Use orchestrator traces/buffers as additional preference/transition data.

Relevant utilities:
- `src/utils/build_describer_synth_dataset_v2.py`
- `src/utils/build_ruler_synth_dataset.py`
- `src/utils/generate_test_pipeline_synth_dataset.py`

## 6) Training modules from generated data

The orchestrator generates data; training is done with module CLIs.

## 6.1 Describer training

Entry point:

`python -m src.describer.cli train ...`

Backed by `src/describer/train.py`.

## 6.2 Ruler training

Entry points:

- `python -m src.ruler.cli train ...`
- `python -m src.ruler.cli eval ...`
- `python -m src.ruler.cli infer ...`

Backed by `src/ruler/training.py`.

## 6.3 Critic training

There is no full critic trainer wired in `src/critic/cli.py` yet. Current critic side includes:
- deterministic critic pipeline runtime (`src/critic/pipeline.py`)
- deterministic scoring engine (`src/critic/engine.py`)

Train critic offline from generated JSONL datasets using your training script/notebook or future repo trainer.

## 7) Memory behavior

Memory is enabled when `ARC_V3_MEMORY_DIR` is set.

## 7.1 Storage

`src/utils/memory.py` manages SQLite + blob references:
- `episodes`
- `attempts`
- `gold_solutions`
- `fragments`
- `cases` (repair case store keyed by namespace + ops_signature + error_type)

## 7.2 Namespace isolation

Use namespace to isolate train/eval memory:
- env: `ARC_V3_MEMORY_NAMESPACE=<name>`
- or CLI: `--train-namespace <name>`

## 7.3 Retrieval usage in orchestrator

- Ruler path gets memory bundles (`solved_examples`, `repair_cases`) via `retrieve_for_ruler`.
- Critic path gets retrieval features/bundles via `retrieve_for_critic`.
- Payloads are bounded by `--memory-max-bytes`.

## 7.4 Case-store updates

Case stats are updated only on strict improvements (`delta > threshold` in orchestrator logic) and tied to comparator-scored transitions.

## 8) Labels and determinism

- Verifier + canonical comparator are the authority for labels.
- Pairwise labels are generated only within `(task_id, iter_idx)`.
- Ties are dropped in dataset builders.
- Rows carry comparator/versioned feature data for reproducibility.

## 9) Practical notes

- For full pipeline work, always call `src.orchestrator.run_orchestrator`.
- Keep teacher active even when student is enabled.
- Use `--trace-jsonl` or `--out-dir` for post-run auditing and dataset generation.
- Start with small `--max-tasks` smoke runs before large batch jobs.
