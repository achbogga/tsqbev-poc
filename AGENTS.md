# TSQBEV Research Agent Rules

This file captures standing research instructions for this repo so they do not need to be
repeated in chat.

Read [SOUL.md](SOUL.md) alongside this file. `AGENTS.md` is the contract; `SOUL.md` is the
behavioral standard for how that contract should be executed.

Hard-pivot launch mechanics are defined in [docs/hard-pivot-execution.md](docs/hard-pivot-execution.md).
Parallel CPU-side planning practice during active runs is defined in
[docs/think-while-training.md](docs/think-while-training.md).
The parallel harness-search control plane is defined in
[docs/meta-harness-v2.md](docs/meta-harness-v2.md).

## Core Research Stance

- question every design decision from first principles before extending it
- assume the current design may be wrong until local evidence says otherwise
- move fast, but only with bounded, evidence-backed steps
- fix small issues immediately when they are clearly blocking progress
- do not stop at a single result; continue to the next highest-ROI step unless a boundary is hit
- if a public upstream stack or teacher suite is materially better grounded than the current
  custom path, pivot to that stack rather than deepening the custom path
- if the active thesis says "hard pivot", the executor must launch a real recipe in that family or
  fail loudly; silent fallback to legacy carryover recipes is not allowed
- read the local research brief before planning new bounded work
- write back distilled findings and PI-facing reports after each completed invocation
- keep lane work explicit and staged: isolated OpenLane sanity first, then export/eval, then only
  later consider joint detection+lane work if both branches are healthy
- keep a separate CPU-only maintenance agent alive daily; it should run repo health checks,
  memory/report sync, and tech-debt triage even when the main research agent is doing other work
- while GPU experiments run, keep one CPU-only confidence-building task alive as common practice:
  retrieval audit, confidence review, next-step shortlist, or infra-gap note
- planning during active runs must produce a tangible artifact that changes what we run, stop,
  retrieve, measure, or publish next; otherwise it is not good research work
- treat `harness_v2` as the preferred place to search over prompts, retrieval, memory usage,
  orchestration, and stop/pivot policy without destabilizing the live supervisor
- new control-plane ideas should enter the repo through replay benchmarks and shadow mode before
  they are allowed to steer live GPU execution
- `codex-loop` is the preferred persistent entrypoint when the goal is unattended operation with
  autonomous harness search/promotion plus bounded supervisor execution
- public-student `mini` runs are smoke/control only by default; they may not reset the frontier or
  count as breakthrough evidence unless they satisfy an exact public reproduction or comparable
  full-training contract

## Target Stack Bias

The default migration target for this repo is now a foundation-teacher perspective-sparse student:

- sparse temporal runtime core from the `Sparse4D` family
- `BEVFormer v2`-style perspective supervision to adapt strong image backbones
- `DINOv3` first and `DINOv2` only as a fallback for projected camera foundation features
- `OpenPCDet` / `BEVFusion` as strong geometry and multimodal teachers
- `MapTRv2` as the staged lane/vector-map head
- `EfficientViT`, `OFA`, `AMC`, and `HAQ` for the later efficiency path
- `SAM 2.1` region or mask priors as the frontier public Segment Anything branch until a newer official public release exists
- `Alpamayo` as a teacher/evaluator for long-tail reasoning, not the runtime trunk

Use the legacy sparse-query line and dense-BEV control baselines only as comparison evidence unless
the new reset is demonstrably worse on the same gate.

## Literature And Source Discipline

For any unstable or competitive design area, especially:

- knowledge distillation
- teacher-student training
- pretrained backbones and foundation encoders
- proposal/query design
- multimodal fusion
- deployment/runtime tradeoffs

the agent must consult:

- primary papers
- official codebases
- official pretrained weights when available

For efficiency and deployment-oriented work, prefer primary references from MIT HAN Lab and
official NVIDIA deployment docs when they exist.

When a literature review produces reusable technique cards or ablation guidance, store the durable
result under `research/knowledge/` and sync it into the local memory stack instead of leaving it in
chat-only form.

When official papers, repos, or checkpoints are important to an active research direction, mirror
them into the local research asset store and keep a generated manifest under
`artifacts/knowledge_assets/` so the evidence remains quickly retrievable.

The repo should prefer primary-source citations over secondary summaries. Novel repo-specific claims
must cite the local artifact, code path, or paper draft in this repo.

## KD Exploration Rule

The repo must treat KD as a broad design space, not a single technique. At minimum, candidate
directions include:

- logits distillation
- feature distillation, including lightweight alignment such as `1x1` projection layers
- dense output distillation, including teacher heatmaps, quality maps, objectness maps, and BEV
  segmentation / occupancy style targets when available
- box/query distillation
- relational distillation
- response / ranking distillation where the teacher score itself is the supervised target
- online or mutual distillation
- self-distillation
- teacher-anchor or teacher-proposal transfer

The active KD choice must be justified by ROI, not by completeness.

For this repo, default KD ROI priority is:

1. teacher outputs that directly affect ranking and calibration
2. teacher BEV or multiscale features through lightweight alignment
3. teacher dense maps such as heatmaps or segmentation targets
4. relational / pairwise structure
5. online, mutual, or self-distillation

Do not jump to higher-cost KD forms until the lower-cost targets have been tested honestly.

Lane work must also stay staged:

- get a real OpenLane baseline and evaluator artifact first
- keep lane as a separate bounded track while detection is still bottlenecked on ranking or
  source-mix collapse
- only mix lane into the main detection loop after both branches have real measured baselines
- when lane enters multitask training, detection non-regression is mandatory

## ROI Rule

Before starting a new direction, record:

- the bottleneck being targeted
- the expected lift
- the integration cost
- the evidence already available
- the stopping condition

Prefer the smallest coherent change that can falsify the current hypothesis quickly.

## Token-Burn Score

Every nontrivial direction should maintain a `token_burn_score` using this rubric:

- `expected_roi`: `1-5`
- `integration_cost`: `1-5`
- `uncertainty`: `1-5`
- `evidence_gain`: `1-5`
- `token_burn_score = integration_cost + uncertainty - expected_roi - evidence_gain`

Interpretation:

- `<= -2`: proceed aggressively
- `-1` to `2`: proceed, but checkpoint after the next bounded result
- `>= 3`: stop and reassess before spending more time or compute

The score does not need to be numerically perfect. It exists to force explicit ROI thinking and
to prevent rabbit holes.

## Rabbit-Hole Boundary

Stop and reassess when any of these are true:

- the same hypothesis has failed in two bounded runs without new evidence
- the measured bottleneck has shifted and the current branch no longer targets it
- implementation complexity is growing faster than evidence quality
- the `token_burn_score` is `>= 3`

When stopping, write down:

- what was tried
- what changed in the evidence
- why the direction is no longer the best next move

## Research Loop Expectations

The bounded research loop must:

- remain evidence-first and append-only
- promote only on official metrics or clearly defined gate outcomes
- preserve failed runs
- compare teacher-off and teacher-on paths honestly
- use web/literature review to refresh the design space when local progress stalls
- treat the local research-memory stack as required context, not optional decoration
- persist durable literature syntheses under `research/knowledge/` so future briefs and planners
  can retrieve them directly
- emit a machine-readable `boss_progress_verdict` after every invocation that compares the new
  promoted run against the previous mini incumbent and the best historical mini result
- prefer hosted frontier reasoning models for planner and critic roles whenever credentials exist,
  with local heuristics only as fallback
- when repeated `incremental_progress`, `schedule_checkpoint_drift`, or regression appears,
  suppress low-ROI exploit families automatically instead of reopening them by habit
- when multiple ROI-positive directions land together, prefer one KD/ranking branch and one
  augmentation branch before opening a broader search fanout
- after two stalled winner-line runs, stop local weight nudges and pivot to a new architecture or
  teacher family with primary-source support
- stop immediately on catastrophic official-eval or export-sanity failure, capture the failure
  signature, and reassess before relaunching

## Repo Update Rule

When a standing research rule changes, update:

- `AGENTS.md`
- `program.md`
- the relevant `specs/*.md`
- any user-facing docs whose interpretation would otherwise drift
- the local research brief/report templates when the workflow changes
