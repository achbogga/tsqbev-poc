# MIT HAN Lab Efficient ML Playbook

This repo now carries a structured HAN Lab knowledge base at
[research/knowledge/mit_han_efficient_ml_kb.json](../research/knowledge/mit_han_efficient_ml_kb.json).
The goal is not to collect slogans. The goal is to preserve reusable engineering laws, ablation
lessons, and deployment instincts from HAN Lab's efficient-ML work in a form the repo can query and
the research loop can act on.

Primary sources used here are official MIT HAN Lab project/topic pages, official code repositories,
and official paper pages linked from those sources.

## Design Laws

1. Optimize the real deployment metric, not the proxy.
   AMC, HAQ, ProxylessNAS, OFA, APQ, MCUNet, and BEVFusion all point to the same rule:
   FLOPs, params, and nominal asymptotic complexity are weak objectives unless they correlate with
   the actual device bottleneck. Source: [AMC](https://github.com/mit-han-lab/amc),
   [HAQ](https://hanlab18.mit.edu/projects/haq/papers/haq_arxiv.pdf),
   [ProxylessNAS](https://arxiv.org/abs/1812.00332),
   [Once-for-All](https://github.com/mit-han-lab/once-for-all),
   [MCUNet](https://hanlab.mit.edu/projects/mcunet),
   [BEVFusion](https://github.com/mit-han-lab/bevfusion).

2. Search spaces matter as much as search algorithms.
   OFA, APQ, MCUNet, and SPVNAS do not just search harder; they search inside spaces already shaped
   around feasible operators, memory schedules, or sparse 3D primitives. Source:
   [OFA](https://github.com/mit-han-lab/once-for-all),
   [APQ](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_APQ_Joint_Search_for_Network_Architecture_Pruning_and_Quantization_Policy_CVPR_2020_paper.pdf),
   [MCUNet](https://hanlab.mit.edu/projects/mcunet),
   [SPVNAS](https://github.com/mit-han-lab/spvnas).

3. Memory traffic is often the real bottleneck.
   TinyTL, MCUNet, SmoothQuant, AWQ, StreamingLLM, QServe, and LServe each attack memory pressure
   rather than only reducing arithmetic. Source: [TinyTL](https://hanlab.mit.edu/projects/tinytl),
   [MCUNet](https://hanlab.mit.edu/projects/mcunet),
   [SmoothQuant](https://hanlab.mit.edu/projects/tinyml),
   [AWQ](https://github.com/mit-han-lab/llm-awq),
   [StreamingLLM](https://github.com/mit-han-lab/streaming-llm),
   [QServe](https://hanlab.mit.edu/projects/qserve),
   [LServe](https://openreview.net/pdf?id=KQ4pJFFqT1).

4. Algorithm-system co-design is a requirement, not a polish step.
   TinyEngine, TorchSparse++, optimized BEV pooling, OmniServe, and Nunchaku all show that the
   model trick is only half the result. Source: [MCUNet](https://hanlab.mit.edu/projects/mcunet),
   [TorchSparse](https://github.com/mit-han-lab/torchsparse),
   [BEVFusion](https://github.com/mit-han-lab/bevfusion),
   [QServe](https://hanlab.mit.edu/projects/qserve),
   [SVDQuant / Nunchaku](https://github.com/mit-han-lab/nunchaku).

5. Uniform treatment is usually wrong.
   Layerwise pruning, mixed precision, attention sinks, retrieval vs streaming heads, and low-rank
   outlier handling all beat one-size-fits-all rules. Source:
   [AMC](https://github.com/mit-han-lab/amc),
   [HAQ](https://hanlab18.mit.edu/projects/haq/papers/haq_arxiv.pdf),
   [SmoothQuant](https://hanlab.mit.edu/projects/tinyml),
   [AWQ](https://github.com/mit-han-lab/llm-awq),
   [DuoAttention](https://arxiv.org/abs/2410.10819),
   [SVDQuant](https://arxiv.org/abs/2411.05007).

6. Preserve just enough global structure while making the bulk of computation cheap and regular.
   EfficientViT, FlatFormer, StreamingLLM, DuoAttention, and Radial Attention all fit this pattern.
   Source: [EfficientViT](https://hanlab.mit.edu/projects/efficientvit),
   [FlatFormer](https://github.com/mit-han-lab/flatformer),
   [StreamingLLM](https://github.com/mit-han-lab/streaming-llm),
   [DuoAttention](https://arxiv.org/abs/2410.10819),
   [Radial Attention](https://github.com/mit-han-lab/radial-attention).

## Technique Map

### Search and Specialization

- `AMC`: use when the architecture family is already good and deployment cost is the main problem.
  The lesson is to prune against measured latency or energy, not just FLOPs.
- `ProxylessNAS`: use when you can search directly on the target device and operator set.
  The lesson is that proxy gaps are expensive.
- `Once-for-All`: use when one student family must support many budgets or SKUs.
  The lesson is to train once, specialize many times.
- `APQ`: use when architecture, pruning, and precision interact strongly.
  The lesson is to optimize them jointly, not sequentially.

### Memory-First Edge Methods

- `MCUNet`: use when peak activation memory makes the architecture infeasible.
  The lesson is that runtime scheduling expands the model search space.
- `TinyTL`: use when you need lightweight adaptation on a strong frozen trunk.
  The lesson is that activation storage dominates fine-tuning memory.
- `PockEngine`: use when sparse or partial on-device fine-tuning must be real, not just theoretical.

### Efficient Vision and 3D Perception

- `EfficientViT`: use when high-resolution camera processing needs deployment-grade latency.
  The lesson is that lightweight linear attention needs local convolution and multi-scale fusion.
- `EfficientViT-SAM`: use when dense region priors matter but full SAM is too slow to run often.
- `SPVNAS`: use when LiDAR/backbone quality and efficiency both matter and sparse 3D search is
  justified.
- `TorchSparse++`: use when sparse GPU kernels, not architecture quality, are the current blocker.
- `FlatFormer`: use when point-cloud transformers are attractive but still too irregular to deploy.
- `BEVFusion`: use when one shared BEV representation must drive multiple perception tasks.

### Quantization and Serving

- `SmoothQuant`: use when activation outliers block activation quantization.
- `AWQ`: use when weight bandwidth dominates and weight-only quantization is enough.
- `QServe`: use when quantization exists on paper but runtime overhead erases the win.
- `SVDQuant`: use when outlier structure is concentrated enough for a low-rank correction to carry
  the hard part of quantization.

### Long-Context and Sparse Attention

- `StreamingLLM`: use when you need stable streaming behavior rather than faithful global recall.
- `DuoAttention`: use when head roles naturally separate into retrieval and streaming behavior.
- `LServe`: use when long-context serving is now a systems problem, not just an attention-pattern
  problem.
- `Radial Attention`: use when sequence/video length dominates cost and a decaying sparse prior is
  acceptable.

## What This Means For TSQBEV

### Camera Branch

- `DINOv3` remains the frontier frozen teacher-style image backbone.
- `EfficientViT` is the right deployable fallback or later student camera backbone.
- The transferable HAN Lab trick is not “use ViTs blindly”; it is:
  preserve global context with hardware-friendly operators and only pay expensive token interaction
  where it clearly earns its keep.

Inference from the sources:
- for our student, keep frozen or lightly adapted foundation features;
- make the projector and camera-to-BEV path regular enough for TensorRT or equivalent deployment;
- do not accept a camera branch that is accurate but structurally hostile to the target runtime.

### LiDAR / Sparse 3D Branch

- `SPVNAS`, `TorchSparse++`, and `FlatFormer` matter more than generic 2D efficiency tricks.
- The reusable law is that sparse 3D efficiency is mostly about matching architecture and kernel
  dataflow to irregular workloads.

Inference from the sources:
- if LiDAR stays in the student, we should eventually search or specialize inside a sparse 3D family
  instead of treating LiDAR as a fixed generic encoder;
- if sparse attention is used, it needs the FlatFormer-style regularization story, not a naive
  point-attention transplant.

### Multimodal Fusion

- `BEVFusion` remains the strong control arm and teacher.
- The reusable trick is not “BEV is always best”; it is that the representation space should align
  with the shared tasks and with the optimized runtime primitives.

Inference from the sources:
- detection plus lane/map should share BEV only after we have official-metric checkpointing and
  adapter isolation;
- naive joint loss minimization is not a trustworthy multitask signal.

### Deployment and Control Plane

- `QServe`, `LServe`, and `StreamingLLM` are more relevant to our agent/supervisor stack than to the
  perception student directly.
- Their core lesson is that long-running autonomy depends on exact state, cache discipline, and
  serving-system design, not just better prompts.

Inference from the sources:
- our control plane should keep exact state in `DuckDB`, semantic evidence in `Qdrant`, and use
  bounded official-metric evaluation workers;
- long-context reasoning should eventually use real sparse/streaming serving rather than naive
  endlessly growing contexts.

## Default HAN-Lab-Informed Playbook For This Repo

1. Keep the strongest unrestricted teacher possible.
2. Make every student/export decision answer to official metrics and export sanity, not loss alone.
3. Treat camera, LiDAR, sparse attention, and serving as hardware-aware design problems.
4. Use adapters and staged multitask training before full shared-trunk joint training.
5. Compress and specialize only after the architecture family is stable.
6. When the repo gets stuck, switch families, not just weights.

## Living Database

- Structured database:
  [research/knowledge/mit_han_efficient_ml_kb.json](../research/knowledge/mit_han_efficient_ml_kb.json)
- This file is ingested by the local research-memory sync and becomes queryable exact/semantic
  evidence for future briefs and planning.
