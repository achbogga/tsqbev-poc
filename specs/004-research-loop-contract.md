# Spec 004: Research Loop Contract

## Goal

Enable a bounded local experiment loop that borrows the strongest transferable mechanics from
Andrej Karpathy's public `autoresearch` repo without turning `tsqbev-poc` into an unbounded
autonomous system.

## Standing Research Policy

The loop must be run with first-principles skepticism:

- re-question material design decisions before extending them
- assume the current bottleneck diagnosis may be wrong until supported by local evidence
- move quickly with bounded experiments, not with vague open-ended exploration
- refresh the design space with primary papers, official repos, and official weights when
  progress stalls, especially for KD, teacher-student design, pretrained backbones, query design,
  multimodal fusion, and deployment tradeoffs
- if a public dense-BEV upstream stack is materially better supported than the current custom
  sparse-query line, treat that reset as the main search direction

The loop must treat KD as a menu of candidate mechanisms, not a single checkbox. Candidate
directions include logits, feature, lightweight `1x1` alignment, dense teacher outputs such as
heatmaps or BEV maps, relational, online, mutual, self-distillation, and teacher-anchor transfer.

For the current TSQBEV student, the bounded loop should prefer KD interventions in this order
unless local evidence strongly contradicts it:

1. ranking-critical outputs such as quality-aware class scores, heatmaps, or objectness targets
2. lightweight teacher feature alignment through BEV or multiscale image features
3. dense teacher maps such as segmentation / occupancy style supervision
4. relational KD
5. online, mutual, or self-distillation

## ROI And Token-Burn Rule

Every active direction must record:

- the bottleneck being targeted
- the expected lift
- the integration cost
- the evidence already available
- the stopping condition

Every active direction must also maintain a lightweight `token_burn_score`:

- `expected_roi`: `1-5`
- `integration_cost`: `1-5`
- `uncertainty`: `1-5`
- `evidence_gain`: `1-5`
- `token_burn_score = integration_cost + uncertainty - expected_roi - evidence_gain`

Boundary:

- `<= -2`: continue aggressively
- `-1` to `2`: continue, but force a checkpoint after the next bounded result
- `>= 3`: stop and reassess before spending more time or compute

The loop must stop and reassess when the active branch becomes a rabbit hole, including repeated
bounded failures without new evidence or a clear mismatch between the measured bottleneck and the
current intervention.

The loop must also operate from durable local memory:

- build a pre-run research brief from the exact catalog and evidence index
- write a pre-run first-principles checkpoint before launching the invocation
- sync new run artifacts back into the local memory stack
- publish a PI-facing report after each completed invocation

## Active Scope

- dataset: `nuScenes v1.0-mini`
- train split: `mini_train`
- validation split: `mini_val`
- experiment count per invocation: at most `7`
- fixed comparable train budget per recipe: `max_train_steps = 960`
- loop shape:
  - baseline or carry-over incumbent recheck
  - dense-BEV reset baselines when available
  - paired teacher-off versus teacher-on comparison when a teacher cache is available
  - bounded exploration
  - bounded exploitation derived from the current incumbent
  - when warranted by the current bottleneck, one explicit augmentation branch and one explicit
    KD/ranking branch may be added without opening the loop further

## Allowed Recipe Changes

- dense-BEV architecture choices across LiDAR, camera, fusion, detection, and lane/map heads
- compact pretrained image-backbone family
- pretrained image-backbone freeze policy
- optional cache-backed external teacher provider
- teacher usage mode: `off`, `KD-only`, `teacher-anchor`
- teacher-anchor selection policy inside the fixed seed budget
- batch size
- gradient accumulation
- learning rate
- optimizer schedule
- gradient clip norm
- detection loss mode and hard-negative budget
- best-checkpoint selection policy
- bounded score-threshold / top-k calibration
- label-safe augmentation mode
- teacher-region objectness / ranking supervision derived from cached teacher boxes and scores

## Required Per-Run Metadata

Every run must record:

- `run_id`
- recipe name
- parent recipe, if any
- hypothesis
- mutation reason
- targeted bottleneck
- token-burn score
- git SHA
- UTC timestamp
- exact canonical CLI command
- device and package environment
- train/val sample counts
- train, export, eval, and source-mix durations

## Required Outputs

- append-only `results.jsonl`
- human-readable `results.tsv`
- `summary.json`
- per-run `manifest.json`
- selected checkpoint path
- selected epoch and best epoch
- synthetic forward latency measurement for each completed recipe
- official `mini_val` export/eval for each completed recipe
- calibration summary when a run sweeps threshold / `top_k`
- per-run source-mix diagnostics from the selected sparse query bank
- per-run prediction-geometry diagnostics from the exported `mini_val` result JSON
- per-run root-cause verdict
- `artifacts/memory/sync_manifest.json`
- `artifacts/memory/brief.json`
- `docs/reports/current.md`
- supervisor-side `first_principles_checkpoint.json` for each continuous invocation

## Decision Semantics

Each record must distinguish:

- `interim_decision`: `advance`, `reject`, or `crash`
- `final_decision`: `promote`, `discard`, or `crash`

Exactly one record may end a completed invocation with `final_decision = promote`.

## Selection Policy

- primary metric: official `mini_val` `NDS`
- secondary metric: official `mini_val` `mAP`
- tertiary metric: validation loss

The loop may not keep a recipe purely because it has a lower surrogate loss if its official
`mini_val` metrics are worse.

The loop may not promote a recipe whose exported predictions are still geometrically implausible,
even if `NDS`, `mAP`, or validation loss improve.

The loop may not introduce geometry-changing augmentations unless the affected dataset loader also
updates the corresponding camera intrinsics and supervision transforms correctly.

If a run keeps a best checkpoint that materially outperforms its last checkpoint, the selected
checkpoint and selected-epoch metrics are the ones that count for promotion.

## Teacher Comparison Rule

If a teacher cache is available for the invocation, the loop must produce at least one valid paired
comparison on the same mini setup:

- teacher-off student baseline
- teacher-on `KD-only` variant with the same architecture
- teacher-on `teacher-anchor` variant when compatible, treated as the preferred geometry exploit

Teacher-anchor runs must keep the teacher role narrow:

- cached teacher detections provide the primary LiDAR/object anchors
- routing must switch to anchor-first selection
- teacher distillation may be disabled so the run isolates anchor quality from extra KD pressure

Teacher-lift claims are invalid if teacher targets are not present in the collated training batch
or if the comparison mixes different backbones or incompatible parent recipes.

## Scale-Gate Output

Every completed invocation must emit a machine-readable `scale_gate_verdict` in `summary.json`.

This verdict must at minimum report:

- source-mix stability status
- geometry sanity status
- mini generalization status
- teacher-lift status
- efficiency status
- repeatability status
- overall `authorized: true|false`

The loop may promote a local incumbent without authorizing larger-scale spend.

## Forbidden

- full `v1.0-trainval` search through the loop
- silent mutation of data contracts or metric contracts
- deleting failed experiments from the ledger
- open-ended or recursive loop execution

## References

- Andrej Karpathy `autoresearch`: <https://github.com/karpathy/autoresearch>
- baseline `program.md`: <https://raw.githubusercontent.com/karpathy/autoresearch/master/program.md>
- nuScenes official mini split support: <https://github.com/nutonomy/nuscenes-devkit>
