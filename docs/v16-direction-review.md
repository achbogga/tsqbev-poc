# Post-v16 Direction Review

This note re-evaluates the next TSQBEV move from first principles after the `v16` mini loop.

Primary external sources:

- Generalized Focal Loss: <https://arxiv.org/abs/2006.04388>
- VarifocalNet: <https://arxiv.org/abs/2008.13367>
- TOOD: <https://openaccess.thecvf.com/content/ICCV2021/html/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.html>
- BEVFusion official repo: <https://github.com/mit-han-lab/bevfusion>
- NVIDIA DeepStream BEVFusion docs: <https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_3D_MultiModal_Lidar_Camera_BEVFusion.html>
- BEVDepth official repo: <https://github.com/Megvii-BaseDetection/BEVDepth>
- OpenPCDet official repo: <https://github.com/open-mmlab/OpenPCDet>
- DistillBEV: <https://openaccess.thecvf.com/content/ICCV2023/html/Wang_DistillBEV_Boosting_Multi-Camera_3D_Object_Detection_with_Cross-Modal_Knowledge_Distillation_ICCV_2023_paper.html>
- BEVDistill: <https://arxiv.org/abs/2211.09386>

Primary local evidence:

- `v16` promoted mini result:
  [artifacts/research_v16_qfl_mini_v1/research_loop/summary.json](../artifacts/research_v16_qfl_mini_v1/research_loop/summary.json)
- `v19` post-fix mini result:
  [artifacts/research_v19_anchor_mix_teachercache_v2/research_loop/summary.json](../artifacts/research_v19_anchor_mix_teachercache_v2/research_loop/summary.json)
- best passed overfit gate:
  [artifacts/gates/recovery_v14_teacher_anchor_quality_focal/overfit_gate/summary.json](../artifacts/gates/recovery_v14_teacher_anchor_quality_focal/overfit_gate/summary.json)
- reproduced BEVFusion baseline:
  [artifacts/bevfusion_repro/bevfusion_bbox_summary.json](../artifacts/bevfusion_repro/bevfusion_bbox_summary.json)

## What The Repo Evidence Actually Says

- `v16` is still the strongest mini result so far:
  - `NDS = 0.1491`
  - `mAP = 0.1848`
  - `boxes/sample mean = 77.62`
- `v19` reduced export overproduction materially:
  - `boxes/sample mean = 35.88`
  - but regressed to `NDS = 0.1192`, `mAP = 0.0968`
- the best overfit gate already passes:
  - `recovery_v14_teacher_anchor_quality_focal`
  - `NDS = 0.1553`
  - `mAP = 0.1992`
  - `train_total_ratio = 0.3624`
  - `car AP@4m = 0.4958`
- the same overfit artifact still diagnoses ranking/export as the main remaining bottleneck
- the student is still nowhere near the reproduced upstream ceiling:
  - local BEVFusion repro is `mAP 0.6730`, `NDS 0.7072`

Conclusion: the repo is no longer failing on basic geometry or optimization only. It is failing on ranking quality and on the fact that the selected student branch still collapses to LiDAR-only routing.

## What The Literature Says

### 1. Quality should live in the ranking score, not only in a separate objectness branch

`GFL`, `VarifocalNet`, and `TOOD` all push the same direction:

- classification and localization quality should be aligned
- the inference ranking score should encode localization quality directly
- a post-hoc product of independent score heads is weaker than a trained quality-aware score

For TSQBEV, this matters because the current student still ranks detections through separate class and objectness terms. Quality-aware training was added to the objectness branch, but the class branch is still trained as a plain binary target, and export still combines them after the fact.

Inference from those sources: the next bounded sparse-student change should be a quality-aware class score or task-aligned ranking target, not another generic router sweep.

### 2. Teacher information should be preserved as quality supervision, not only as additive anchor priors

`DistillBEV` and `BEVDistill` both argue for transferring stronger teacher signal through BEV/object supervision rather than relying only on architectural priors.

For TSQBEV, this means the next teacher-aware step should preserve teacher confidence/quality in the matched class score and calibration target, not only inject teacher logits additively into the head.

### 3. The strongest public deployable ceiling is already known

`BEVFusion` is the strongest directly relevant public reference because:

- it has official code and checkpoints
- we reproduced it locally near paper level
- the official repo and NVIDIA DeepStream docs provide a real deployment path, including Jetson Orin

`BEVDepth` remains the strongest lightweight camera-BEV component source with public code and checkpoints, and a better small-student camera branch candidate than inventing another custom proposal path.

## Decision

### Keep one bounded sparse-student line alive

Do exactly one high-ROI sparse-student change next:

1. replace separate export ranking with a quality-aware class score
2. make teacher-anchor supervision influence that quality-aware score directly
3. re-run the same `32`-sample overfit gate first

Do not spend more cycles on generic source-mix or query-budget sweeps until that ranking change is tested.

### Start the dense-BEV control arm in parallel

The medium-term direction should be:

- compact BEVDepth / BEVDet-style camera lift
- PointPillars / CenterPoint-style LiDAR branch
- BEVFusion-style shared BEV fusion
- distillation from the reproduced local BEVFusion + CenterPoint teachers

Reason: the repo now has hard evidence that the custom sparse student is far below the mature public ceiling, while the public ceiling is both reproducible and deployable.

### Defer lane training for now

OpenLane data prep should continue, but OpenLane training should stay deferred until detection is healthier.

Reason:

- today’s lane branch shares the same core training path and total loss
- adding lane optimization now would create a second moving target while detection ranking is still the main blocker
- lane work makes more sense after the detection branch or dense-BEV control arm is stable
