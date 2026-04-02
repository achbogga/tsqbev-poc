# TSQBEV Research Agent Rules

This file captures standing research instructions for this repo so they do not need to be
repeated in chat.

## Core Research Stance

- question every design decision from first principles before extending it
- assume the current design may be wrong until local evidence says otherwise
- move fast, but only with bounded, evidence-backed steps
- fix small issues immediately when they are clearly blocking progress
- do not stop at a single result; continue to the next highest-ROI step unless a boundary is hit
- if a public upstream dense-BEV stack is materially better grounded than the current custom
  path, pivot to that stack rather than deepening the custom path
- read the local research brief before planning new bounded work
- write back distilled findings and PI-facing reports after each completed invocation
- keep lane work explicit and staged: isolated OpenLane sanity first, then export/eval, then only
  later consider joint detection+lane work if both branches are healthy

## Target Stack Bias

The default migration target for this repo is now a public dense-BEV fusion stack built from
official upstreams such as:

- BEVFusion
- OpenPCDet
- BEVDet / BEVDepth
- MapTR / MapTRv2
- EfficientViT
- DINOv2 / DINOv3
- OFA / AMC / HAQ

Use the legacy sparse-query line only as comparison evidence unless the dense-BEV reset is
demonstrably worse on the same gate.

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
- when multiple ROI-positive directions land together, prefer one KD/ranking branch and one
  augmentation branch before opening a broader search fanout

## Repo Update Rule

When a standing research rule changes, update:

- `AGENTS.md`
- `program.md`
- the relevant `specs/*.md`
- any user-facing docs whose interpretation would otherwise drift
- the local research brief/report templates when the workflow changes
