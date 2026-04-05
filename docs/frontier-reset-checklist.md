# Frontier Reset Checklist

_Last updated: 2026-04-05_

This checklist is the operating contract for the current repo reset. Work proceeds top to bottom unless a completed item reveals a higher-ROI blocker. No item is considered done without code, validation, and a durable artifact.

## Control Plane

- [x] Replace the silent supervisor startup path with immediate state and log output.
- [x] Make critic rejection non-blocking so the GPU does not idle by default.
- [ ] Persist active phase, planner verdict, critic verdict, and checklist item in supervisor state after launch.
- [ ] Split planning from heavy memory rebuild so pre-run control never blocks on full backfill.
- [ ] Add a bounded fallback path that launches the approved run directly when the supervisor loop fails to transition within a timeout.

## Memory And State

- [ ] Replace `hash` semantic embeddings with strong local retrieval using a real supported frontier local embedder.
- [ ] Enable strong local reranking using a real supported frontier local reranker.
- [ ] Rebuild the local research memory after the embedder upgrade.
- [ ] Verify that the pre-run brief cites the real incumbent, the joint-collapse failure, and the active frontier branch from exact and semantic evidence.

## Evaluation Plane

- [x] Add export sanity diagnostics to official nuScenes evaluation.
- [x] Make joint periodic official evaluation fail soft instead of crashing training.
- [x] Save `checkpoint_best_official.pt` when periodic official metrics improve.
- [ ] Promote only `best_official.pt` in joint or detection runs when official metrics are available.
- [ ] Add a direct report artifact that compares `loss-best` versus `official-best` checkpoints.

## Frontier Camera Branch

- [x] Add `DINOv3` backbone support in the repo model/config/CLI surface.
- [x] Clone official `SAM 2.1` repo locally.
- [x] Add runtime dependencies needed by the official frontier camera stack (`torchmetrics`, `transformers`).
- [ ] Replace the blocked `torch.hub` DINOv3 loading path with a robust gated-weight workflow or an official supported local checkpoint path.
- [ ] Consume `SAM 2.1` priors in training instead of keeping them as config-only hooks.

## Multitask Reset

- [x] Prove that the old naive joint path is invalid by official detection metrics.
- [ ] Add staged multitask curriculum: detection-only control, frozen-trunk lane, then joint finetune.
- [ ] Add task-isolation controls before any new joint promotion attempt.
- [ ] Add official lane metrics to checkpoint selection and non-regression gating.

## Run Queue

- [x] Keep daily maintenance separate from research automation.
- [x] Keep the supervisor able to launch bounded runs with hosted planner and critic.
- [ ] Relaunch the frontier supervisor after the memory upgrade so it plans from stronger retrieval.
- [ ] Launch the next validated detection run from the fixed control plane.
- [ ] Launch the next frontier camera pilot once the official DINOv3 weight-access path is resolved.

## External Blockers

- [ ] Official `DINOv3` pretrained weights are gated. Current failures:
- Meta hosted weight URL returns `403`.
- Hugging Face official repo returns `401` without granted access.
- [ ] No verified public official `SAM v3` target exists. Current public frontier remains `SAM 2.1`.

## Current Truth

- Best confirmed local detection model remains `v28`:
- [summary.json](/home/achbogga/projects/tsqbev-poc/artifacts/research_v28_continuation_v1/research_loop/summary.json)
- `NDS = 0.1833`
- `mAP = 0.1814`

- Current control-plane branch:
- [research_supervisor.log](/home/achbogga/projects/tsqbev-poc/artifacts/autoresearch_frontier/research_supervisor.log)
- [state.json](/home/achbogga/projects/tsqbev-poc/artifacts/autoresearch_frontier/state.json)
