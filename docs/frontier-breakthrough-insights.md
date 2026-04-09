# Frontier Breakthrough Insights

This note is the bridge between the expanded knowledge base and the next real research loop. It
records the first source-backed insights that are worth turning into major architecture changes,
not tiny local tweaks.

## 1. Separate Fusion From Temporal Memory

The strongest near-term insight is structural:

- use gated or deformable cross-attention for *spatial and multimodal fusion*
- use selective SSM or Mamba-style blocks for *temporal carry*

Do not ask one module to solve both problems. Cross-attention decides what to read from another
source. SSMs decide what survives over time.

## 2. Foundation Features Need Geometry Adapters

Strong vision backbones like `DINOv3` are not enough by themselves. The bridge from image-space
features into the world model needs:

- view-aware projection
- perspective supervision
- either deformable or gated sparse fusion

Without that bridge, rich image features remain semantically strong but geometrically weak.

For the deployable student, the practical implication is stronger:

- keep `DINOv3` and `SAM 2.1` on the teacher side first
- let the embedded student stay lightweight
- change the student bridge/fusion path before changing the student backbone again

## 3. Use SAM 2.1 As Dense Supervision, Not As The Trunk

`SAM 2.1` is valuable when box-only supervision is too sparse and the student is losing object
support or region consistency.

The right usage is:

- teacher boxes or prompts in
- dense support masks or region priors out
- supervise region objectness, occupancy, or proposal filtering

The wrong usage is turning the student runtime into a full SAM stack.

## 4. World Models And Reasoning Models Belong On The Teacher Side

`Alpamayo`, `Cosmos-Reason2`, `Cosmos-Predict2.5`, and `Cosmos-Transfer2.5` should be used for:

- scenario critique
- synthetic or future-state supervision
- sim-to-real transfer
- planner-side hard-example generation

They should not replace the metric-grounded perception student.

## 5. Efficiency Work Must Follow A Real Metric Frontier

MIT HAN Lab and NVIDIA efficiency stacks give us a strong deployment and serving playbook, but the
order matters:

1. prove the model path
2. prove official metrics
3. then optimize with TensorRT, Model Optimizer, hardware-aware specialization, and advanced sparse
   kernels

The only exception is when a systems bottleneck is blocking the *lab itself*, such as long-context
planner serving. Then control-plane efficiency work is justified.

## 6. The Breakthrough Direction To Test

The current high-conviction architectural hypothesis is:

1. lightweight student camera backbone (`MobileNetV3` / `EfficientNet-B0`)
2. gated latent cross-attention bridge on the student
3. perspective supervision before world aggregation
4. teacher-side `DINOv3` / `SAM 2.1` / BEV supervision
5. teacher BEV / quality-map / region-prior distillation
6. selective SSM for temporal memory only after geometry is stable
7. staged lane/vector head only after detection is stable

This is the smallest frontier-consistent step that is meaningfully different from the local
incremental loop that stalled.
