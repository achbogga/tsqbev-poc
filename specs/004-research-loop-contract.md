# Spec 004: Research Loop Contract

## Goal

Enable a bounded local experiment loop that borrows the strongest transferable mechanics from
Andrej Karpathy's public `autoresearch` repo without turning `tsqbev-poc` into an unbounded
autonomous system.

## Active Scope

- dataset: `nuScenes v1.0-mini`
- train split: `mini_train`
- validation split: `mini_val`
- experiment count per invocation: at most `5`
- fixed comparable train budget per recipe: `max_train_steps = 960`
- loop shape:
  - baseline or carry-over incumbent recheck
  - paired teacher-off versus teacher-on comparison when a teacher cache is available
  - bounded exploration
  - bounded exploitation derived from the current incumbent

## Allowed Recipe Changes

- sparse query-budget allocation across seed sources
- compact pretrained image-backbone family
- pretrained image-backbone freeze policy
- optional cache-backed external teacher provider
- teacher usage mode: `off`, `KD-only`, `KD + teacher-ref-seed`
- batch size
- gradient accumulation
- learning rate

## Required Per-Run Metadata

Every run must record:

- `run_id`
- recipe name
- parent recipe, if any
- hypothesis
- mutation reason
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
- synthetic forward latency measurement for each completed recipe
- official `mini_val` export/eval for each completed recipe
- per-run source-mix diagnostics from the selected sparse query bank

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

## Teacher Comparison Rule

If a teacher cache is available for the invocation, the loop must produce at least one valid paired
comparison on the same mini setup:

- teacher-off student baseline
- teacher-on `KD-only` variant with the same architecture
- teacher-on `teacher-ref-seed` variant when compatible

Teacher-lift claims are invalid if teacher targets are not present in the collated training batch
or if the comparison mixes different backbones or incompatible parent recipes.

## Scale-Gate Output

Every completed invocation must emit a machine-readable `scale_gate_verdict` in `summary.json`.

This verdict must at minimum report:

- source-mix stability status
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
