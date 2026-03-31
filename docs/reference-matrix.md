# Reference Matrix

This repo stays grounded by mapping each subsystem to an original paper and, where available, official source code.

Reference policy as of March 31, 2026:

- Local generated design summaries are not cited directly.
- The sources below are the underlying original references used to ground the implementation.
- The legacy sparse-query line is retained for comparison only; the migration target is a dense
  BEV fusion stack assembled from public upstreams.

| Subsystem | Original paper | Official code | Repo usage |
| --- | --- | --- | --- |
| Dense multimodal fusion target | [BEVFusion](https://arxiv.org/abs/2205.13542) | [mit-han-lab/bevfusion](https://github.com/mit-han-lab/bevfusion) | Primary reset target for shared BEV fusion and public checkpoints |
| Orin deployment substrate | [NVIDIA DS3D BEVFusion docs](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_3D_MultiModal_Lidar_Camera_BEVFusion.html) | [NVIDIA DeepStream](https://docs.nvidia.com/metropolis/deepstream/) | Deployment and TensorRT / DeepStream path for dense multimodal fusion |
| LiDAR runtime baseline | [CenterPoint](https://openaccess.thecvf.com/content/CVPR2021/papers/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.pdf) | [tianweiy/CenterPoint](https://github.com/tianweiy/CenterPoint), [open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet) | Default LiDAR encoder and teacher baseline for the reset stack |
| Camera BEV temporal lifting | [BEVDepth](https://arxiv.org/abs/2206.10092) | [HuangJunJie2017/BEVDet](https://github.com/HuangJunJie2017/BEVDet) | Multi-view camera BEV and temporal depth control arm |
| Dense BEV detector head | [CenterPoint](https://openaccess.thecvf.com/content/CVPR2021/papers/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.pdf) | [open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet) | Dense detection head and public checkpoint source |
| Vector map / lane reset target | [MapTR](https://arxiv.org/abs/2208.14437) | [hustvl/MapTR](https://github.com/hustvl/MapTR) | MapTRv2-style shared-BEV vector lane/map head |
| Lane evaluation reference | [PersFormer](https://arxiv.org/abs/2203.11089) | [OpenDriveLab/PersFormer_3DLane](https://github.com/OpenDriveLab/PersFormer_3DLane) | OpenLane-compatible lane transfer and evaluation reference |
| Dense vision priors | [DINOv2](https://arxiv.org/abs/2304.07193), [DINOv3](https://github.com/facebookresearch/dinov3) | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2), [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | Optional teacher or feature-prior branch, not the first runtime backbone |
| Efficient deployment backbones | [EfficientViT](https://arxiv.org/abs/2305.07027) | [mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit) | Orin-friendly specialization and backbone compression candidate |
| Hardware-aware architecture search | [Once for All](https://arxiv.org/abs/1908.09791) | [mit-han-lab/once-for-all](https://github.com/mit-han-lab/once-for-all) | Backbone specialization playbook |
| Hardware-aware pruning / compression | [AMC](https://arxiv.org/abs/1810.09492) | [mit-han-lab/amc](https://github.com/mit-han-lab/amc) | Structured pruning reference |
| Hardware-aware quantization | [HAQ](https://arxiv.org/abs/1811.08886) | [mit-han-lab/haq](https://github.com/mit-han-lab/haq) | Orin / TensorRT-aware quantization reference |
| 3D-to-2D sparse query sampling | [DETR3D](https://proceedings.mlr.press/v164/wang22b/wang22b.pdf) | [WangYueFt/detr3d](https://github.com/WangYueFt/detr3d) | Calibrated sparse camera sampling and query-based OD framing |
| 3D positional conditioning | [PETR](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870523.pdf) | [megvii-research/PETR](https://github.com/megvii-research/PETR) | Position-aware camera features and proposal-ray initialization |
| Temporal alignment and multitask queries | [PETRv2](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_PETRv2_A_Unified_Framework_for_3D_Perception_from_MultiCamera_Images_ICCV_2023_paper.pdf) | [megvii-research/PETR](https://github.com/megvii-research/PETR) | Temporal alignment and lane/map multitask framing |
| Streaming sparse temporal state | [StreamPETR](https://arxiv.org/abs/2303.11926) | [exiawsh/StreamPETR](https://github.com/exiawsh/StreamPETR) | Persistent sparse temporal state |
| Sparse multimodal aggregation patterns | [Sparse4D](https://arxiv.org/pdf/2211.10581) | [HorizonRobotics/Sparse4D](https://github.com/HorizonRobotics/Sparse4D) | Sparse multi-keypoint sampling and efficient query fusion |
| Sparse camera-only baseline | [SparseBEV](https://arxiv.org/abs/2308.09244) | [MCG-NJU/SparseBEV](https://github.com/MCG-NJU/SparseBEV) | Sampling efficiency and sparse attention discipline |
| External pretrained LiDAR teacher | [CenterPoint](https://openaccess.thecvf.com/content/CVPR2021/papers/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.pdf) | [tianweiy/CenterPoint](https://github.com/tianweiy/CenterPoint), [open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet) | First planned teacher boxes, scores, and seed priors |
| LiDAR-to-camera BEV distillation | [BEVDistill](https://arxiv.org/abs/2211.09386) | [zehuichen123/bevdistill](https://github.com/zehuichen123/bevdistill) | Teacher-target interface and distillation objectives |
| Real-time pillar-native LiDAR encoder candidate | [PillarNet](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700034.pdf) | [VISION-SJTU/PillarNet](https://github.com/VISION-SJTU/PillarNet) | Follow-on in-repo LiDAR encoder candidate if the teacher path validates the direction |
| Camera-LiDAR transformer fusion | [CMT](https://openaccess.thecvf.com/content/ICCV2023/papers/Yan_Cross_Modal_Transformer_Towards_Fast_and_Robust_3D_Object_Detection_ICCV_2023_paper.pdf) | [junjie18/CMT](https://github.com/junjie18/CMT) | Multimodal query fusion inspiration |
| Unified BEV multimodal robustness | [BEVFusion](https://arxiv.org/abs/2205.13542) | [mit-han-lab/bevfusion](https://github.com/mit-han-lab/bevfusion) | Fallback discipline and multimodal design tradeoffs |
| Query-based radar-camera fusion | [RaCFormer](https://openaccess.thecvf.com/content/CVPR2025/html/Chu_RaCFormer_Towards_High-Quality_3D_Object_Detection_via_Query-based_Radar-Camera_Fusion_CVPR_2025_paper.html) | [cxmomo/RaCFormer](https://github.com/cxmomo/RaCFormer) | Query-centric multimodal fusion patterns |
| Multi-view overlap robustness | [Graph-DETR3D](https://arxiv.org/abs/2204.11582) | [zehuichen123/Graph-DETR3D](https://github.com/zehuichen123/Graph-DETR3D) | Overlap/seam reasoning references |
| Dense depth control baseline | [BEVDepth](https://arxiv.org/abs/2206.10092) | [Megvii-BaseDetection/BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth) | Depth-centric control arm for ablations |
| Lane detection baseline | [PersFormer](https://arxiv.org/abs/2203.11089) | [OpenDriveLab/PersFormer_3DLane](https://github.com/OpenDriveLab/PersFormer_3DLane) | Camera-dominant lane reasoning and evaluation references |
| HD-map tokenization | [MapTR](https://arxiv.org/abs/2208.14437) | [hustvl/MapTR](https://github.com/hustvl/MapTR) | Public map-prior adapter and vector token inspiration |
| Orin-aware latency design | [HotBEV](https://proceedings.neurips.cc/paper_files/paper/2023/file/081b08068e4733ae3e7ad019fe8d172f-Paper-Conference.pdf) | Paper only | Latency predictor and operator-budget discipline |
| Public lane dataset | [OpenLane V1](https://github.com/OpenDriveLab/OpenLane) | [OpenDriveLab/OpenLane](https://github.com/OpenDriveLab/OpenLane) | Public lane supervision path |
| LiDAR self-supervised pretraining reference | [3DTrans / AD-PT](https://github.com/PJLab-ADG/3DTrans) | [PJLab-ADG/3DTrans](https://github.com/PJLab-ADG/3DTrans) | Longer-range pretraining reference, not the first bootstrap target |
| Repo implementation and RTX 5000 measurements | [tsqbev-poc repo](https://github.com/achbogga/tsqbev-poc) | [tsqbev-poc repo](https://github.com/achbogga/tsqbev-poc) | Legacy sparse-query implementation details, TensorRT utilities, and local benchmark artifacts |
