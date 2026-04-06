# Hard Pivot Execution Strategy

This document defines the executable launch strategy for the repo's hard pivot. It exists to make
the control plane auditable and to prevent silent fallback into legacy carryover recipes.

## Launch Family

When the active proposal includes frontier tags such as:

- `dino_v3`
- `sam21_offline_support`
- `world_aligned_distillation`
- `bevformer_v2_perspective_supervision`
- `sparse4d_efficiency`
- `openpcdet_teacher`
- `bevfusion_teacher`

the executor must start from a concrete hard-pivot family, not the legacy MobileNet carryover line.

Current executable hard-pivot family:

1. `frontier_dinov3_teacher_distill_vits16`
2. `frontier_dinov3_teacher_distill_vits16_vitb16`
3. `frontier_dinov3_teacher_distill_vits16_no_sam2`
4. `frontier_dinov3_teacher_distill_vits16_teacher_control`

These recipes are the current executable slice of the larger thesis:

- `DINOv3` projected camera branch
- `SAM 2.1` proposal priors
- strong teacher-guided ranking and distillation
- aggressive official-metric gating

## Training Contract

Frontier launch recipes use:

- `epochs = 30`
- `max_train_steps = None`
- `official_eval_every_epochs = 5`
- `official_eval_score_threshold = 0.20`
- `official_eval_top_k = 40`
- `early_stop_patience = 3`
- `early_stop_min_delta = 0.02`
- `early_stop_min_epochs = 10`

This is intentionally different from the old short local exploitation loops.

## Catastrophic Stop Rule

If official eval shows catastrophic failure, the run must stop immediately.

Failure examples:

- `NDS = 0` and `mAP = 0`
- export sanity fails
- unrealistic box scale or geometry
- saturated scores with broken boxes

When this happens:

1. the invocation is marked `catastrophic_stop`
2. no benchmark/export/source-mix follow-up work is run
3. the invocation ends
4. the next supervisor cycle must reassess before relaunching

## Frontier Exploitation Family

Once a hard-pivot incumbent exists, exploitation is restricted to frontier-only follow-ups:

- `*_official_guardrail`
- `*_world_distill`
- `*_no_sam2`
- `*_vitb16`
- `*_teacher_control`

Legacy exploit families such as `query_boost`, `lr_down`, `teacher_bag`, `anchor_mix`,
`focal_hardneg`, and `unfreeze` must stay suppressed on the hard-pivot branch unless the evidence
explicitly forces a retreat.

## Failure Policy

If the active proposal requests a frontier pivot but no executable hard-pivot recipe survives
filtering, the executor must fail loudly instead of silently reopening old carryover baselines.
