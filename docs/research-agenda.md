# Research Agenda

_Last refreshed from promoted research memory on `2026-04-05`._

This agenda is the public, evidence-backed version of the repo's current direction. It is built
from exact local artifacts, the promoted research-memory catalog, and the frontier knowledge base.

## Snapshot

| Item | Current state | Evidence |
| --- | --- | --- |
| Trusted local control | `NDS 0.1833`, `mAP 0.1814`, `18.34 ms`, `40.0` boxes/sample | [../artifacts/research_v29_continuation_v1/research_loop/summary.json](../artifacts/research_v29_continuation_v1/research_loop/summary.json) |
| Reproduced teacher/control ceiling | `BEVFusion` on `nuScenes` val at `NDS 0.7072`, `mAP 0.6730` | [../artifacts/bevfusion_repro/bevfusion_bbox_summary.json](../artifacts/bevfusion_repro/bevfusion_bbox_summary.json) |
| Best overfit frontier | `recovery_v14_teacher_anchor_quality_focal` with `NDS 0.1553`, `mAP 0.1992`, `train_total_ratio 0.3624` | [../artifacts/gates/recovery_v14_teacher_anchor_quality_focal/overfit_gate/summary.json](../artifacts/gates/recovery_v14_teacher_anchor_quality_focal/overfit_gate/summary.json) |
| `DINOv3` probe | real signal, but still below trusted control at `NDS 0.1616`, `mAP 0.1591` | [../artifacts/foundation_v3_dinov3_teacher_vits16_36ep_v1/epoch022_probe_r4/metrics/nuscenes/metrics_summary.json](../artifacts/foundation_v3_dinov3_teacher_vits16_36ep_v1/epoch022_probe_r4/metrics/nuscenes/metrics_summary.json) |
| Lane status | isolated training is viable; naive joint detection+lane remains invalid | [../artifacts/openlane_v1_warmstart_v1/openlane_train.log](../artifacts/openlane_v1_warmstart_v1/openlane_train.log), [../artifacts/joint_public_v2_manual_eval/official_eval/epoch_031/nuscenes/metrics/metrics_summary.json](../artifacts/joint_public_v2_manual_eval/official_eval/epoch_031/nuscenes/metrics/metrics_summary.json) |
| Knowledge base coverage | `57` cards, `117` asset refs, `105` unique mirrored assets | [../artifacts/knowledge_assets/coverage_summary.json](../artifacts/knowledge_assets/coverage_summary.json) |
| Promoted memory build | `repo_sha 647b888`, `65` facts, `2649` events, `1301` evidence chunks | [../artifacts/memory/sync_manifest.json](../artifacts/memory/sync_manifest.json) |

## What We Know

### 1. The repo has a real local control, but the gap to the teacher is still large

- The best trusted local control is no longer the early `0.0158` line. It is the `v29`
  quality-ranked teacher-quality continuation at `NDS 0.1833`, `mAP 0.1814`.
- The reproduced `BEVFusion` ceiling is still far ahead, so the next steps must target
  world-aligned geometry and supervision, not more small local schedule tweaks.

### 2. `DINOv3` is justified, but not yet promoted

- The first real `DINOv3` probe is nonzero and credible.
- It is still below the trusted control, so `DINOv3` only stays on the agenda if paired with the
  missing bridge: perspective supervision and stronger teacher targets.

### 3. Lane is viable in isolation and unsafe in naive joint training

- Isolated `OpenLane V1` training runs.
- The current joint training path can look acceptable on loss while collapsing to zero on official
  detection metrics.
- That means checkpointing, task isolation, and non-regression gates must come before more joint
  claims.

### 4. The repo should not resume a broad local loop until the next branch is hypothesis-tight

- Repeated memory facts now explicitly show too many runs ending in `incremental_progress`,
  `schedule_checkpoint_drift`, and related rabbit holes.
- The agenda must therefore be selective, not exploratory by default.

## Immediate Agenda

### A. Keep one trusted control and stop ambiguous promotion

- Treat the `v29` quality-ranked teacher-quality line as the only trusted local control.
- Require every new branch to beat it on official `mini_val` `NDS/mAP`.
- Keep the latency and export geometry envelope visible at the same time.

Why:

- The memory layer shows the repo already has too many low-signal local mutations.
- The control must stay stable while frontier branches are judged against it.

### B. Continue only the `DINOv3` branch that includes teacher distillation

- Continue the `DINOv3` path only when it includes teacher distillation and official periodic eval.
- Judge it on official `mini_val` metrics, not training loss.

Why:

- The `DINOv3` probe already shows that backbone strength alone is not enough.
- The KB supports `DINOv3` as a foundation branch, but only with a geometry-aware bridge.

Primary evidence:

- [foundation-teacher-direction-2026-04-04.md](foundation-teacher-direction-2026-04-04.md)
- [../research/knowledge/frontier_vision_foundations_kb.json](../research/knowledge/frontier_vision_foundations_kb.json)

### C. Do not resume naive joint detection+lane training

- Freeze the old joint recipe as negative evidence.
- Allow lane back in only through a staged path with detection non-regression gates.

Why:

- The current joint branch has already failed on official detection metrics.
- More iterations of the same coupling would only spend compute to relearn the same lesson.

## Next Agenda

### 1. Add `BEVFormer v2`-style perspective supervision to the `DINOv3` branch

This is the highest-ROI missing bridge between stronger image priors and better 3D geometry.

Apply when:

- the camera backbone is already stronger than the downstream 3D head can naturally exploit
- the failure mode is 3D precision/world alignment, not basic image semantics

Avoid when:

- the bottleneck is clearly dataset corruption, export breakage, or zeroed supervision

### 2. Add `DistillBEV` / `UniDistill` style world-aligned teacher targets

The next distillation target should be BEV or world aligned:

- dense BEV features
- quality-aware class maps
- region occupancy / foreground priors
- box-parameter supervision in world coordinates

Why:

- The teacher ceiling is a dense world model.
- The student should be taught to match world structure, not just sparse local heuristics.

### 3. Use `SAM 2.1` as an offline region-prior teacher, not as a runtime trunk

Use `SAM 2.1` to build:

- region priors
- occupancy support targets
- mask-derived consistency supervision

Do not use it as:

- the primary runtime trunk
- a substitute for world-coordinate teacher supervision

## Strategic Agenda

### A. Lock the long-term architecture thesis

The stable long-term thesis is:

- deployable student: `Sparse4D`-style sparse temporal core
- camera priors: `DINOv3`
- camera/world bridge: `BEVFormer v2`-style perspective supervision
- strong teachers: `BEVFusion` and `OpenPCDet`
- lane/map head: staged `MapTRv2`-style vector output

### B. Reintroduce lane only after the world latent is stable

The correct order is:

1. stable detection-only control
2. frozen/shared-trunk lane warm-start
3. joint training with task isolation and non-regression gates
4. only then selective shared-trunk finetune

### C. Keep systems tricks in the right place

The knowledge base now covers:

- gated cross-attention
- deformable cross-attention
- latent-query interfaces
- selective SSM / Mamba
- MIT HAN Lab sparse/serving/kernel work
- NVIDIA autonomy, Cosmos, TensorRT-LLM, and Alpamayo

The agenda implication is:

- use fusion mechanisms for spatial alignment
- use SSMs later for temporal memory when the perception path is already winning
- use long-context/kv-cache/MLA/MoE tricks primarily in the control plane or teacher stack unless
  a perception-specific reason is proven

## Not In The Active Agenda

- more naive joint detection+lane reruns
- more tiny sparse-query schedule nudges without a new teacher/world hypothesis
- generic Mamba/MoE rewrites as the next perception fix
- treating Alpamayo or large world-model stacks as direct runtime perception trunks

## Public Release Direction

Before asking external collaborators to spend serious effort, the repo should present:

- one trusted local control
- one trusted reproduced upstream teacher/control
- a live memory-backed agenda
- a clear contribution path
- a clear release/bootstrap path for forks

Those public-facing pieces are now maintained in:

- [../README.md](../README.md)
- [../CONTRIBUTING.md](../CONTRIBUTING.md)
- [../RELEASE.md](../RELEASE.md)
- [frontier-knowledge-base.md](frontier-knowledge-base.md)
- [research-memory.md](research-memory.md)
