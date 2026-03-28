# Spec 004: Research Loop Contract

## Goal

Enable a bounded local experiment loop that improves the public baseline without turning the repo into an unbounded autonomous system.

## Active Scope

- dataset: `nuScenes v1.0-mini`
- train split: `mini_train`
- validation split: `mini_val`
- experiment count per invocation: at most `3`

## Allowed Recipe Changes

- sparse query-budget allocation across seed sources
- compact pretrained image-backbone family
- pretrained image-backbone freeze policy
- optional cache-backed external teacher provider
- batch size
- gradient accumulation
- learning rate

## Required Outputs

- append-only `results.jsonl`
- `summary.json`
- selected checkpoint path
- synthetic forward latency measurement for each completed recipe
- official `mini_val` export/eval for each completed recipe
- per-recipe source-mix diagnostics from the selected sparse query bank

## Selection Policy

- primary metric: official `mini_val` `NDS`
- secondary metric: official `mini_val` `mAP`
- tertiary metric: validation loss

The loop may not keep a recipe purely because it has a lower surrogate loss if its official
`mini_val` metrics are worse.

## Forbidden

- full `v1.0-trainval` search through the loop
- silent mutation of data contracts or metric contracts
- deleting failed experiments from the ledger
- open-ended or recursive loop execution

## References

- Andrej Karpathy `autoresearch`: <https://github.com/karpathy/autoresearch>
- nuScenes official mini split support: <https://github.com/nutonomy/nuscenes-devkit>
