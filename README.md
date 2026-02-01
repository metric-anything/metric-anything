# MetricAnything: Scaling Depth Pretraining with Noisy Heterogeneous Sources

<div align="center">

[Paper]() | [Project Page](https://metric-anything.github.io/metric-anything-io/)

<p align="center">
  <a href="https://mabaorui.github.io/">Baorui Ma †*</a> •
  <a href="">Jiahui Yang *</a> •
  <a href="https://scholar.google.com/citations?user=L8tcNioAAAAJ&hl=en">Donglin Di ‡</a> •
  <a href="https://scholar.google.com/citations?user=gGAoxSAAAAAJ&hl=en">Xuancheng Zhang</a> •
  <a href="">Jianxun Cui</a> •
  <a href="">Hao Li</a> •
  <a href="">Xie Yan</a> •
  <a href="">Wei Chen</a> <br>
  † Corresponding author | * Equal contribution | ‡ Project leader
</p>


</div>

## Abstract

**Metric Anything** introduces a simple and scalable pretraining framework that learns metric depth from noisy, diverse 3D sources without manually engineered prompts, camera-specific modeling, or task-specific architectures. Our key insight is the **Sparse Metric Prompt**, created by randomly masking depth maps, which serves as a universal interface that decouples spatial reasoning from sensor and camera biases.

<div align="center">
  <img src="assets/pipe.jpeg" width="90%">
</div>

## Key Ideas

1. **Sparse Metric Prompt**: Randomly mask depth maps to create sparse prompts that decouple spatial understanding from sensor-specific biases, enabling effective learning from diverse, noisy sources.

2. **Large-Scale Data Aggregation**: We assemble ~20M image-depth pairs spanning reconstructed (SfM/SLAM/MVS), captured (LiDAR/ToF/RGB-D), and rendered 3D data across 10,000+ camera models.

3. **Prompt-Free Distillation**: Distill the pretrained model into a prompt-free student that achieves SOTA performance on monocular depth estimation without requiring prompts.


## Release plan
We will follow the open-source plan below in the coming weeks:
<details open>
<summary><b>Pre-trained checkpoints</b> </summary>

- [ ] 1. Prompt-Based Metric Depth Map Model
- [x] 2. Prompt-Free Metric Point Map Model
- [ ] 3. Prompt-Free Metric Depth Map Model
</details>

<details open>
<summary><b>Inference Code</b> </summary>

- [ ] Inference scripts and demo

> **2. Prompt-Free Metric Point Map Model:**
See [HERE](./models/student_pointmap/README.md) | [Huggingface demo](https://huggingface.co/spaces/yjh001/metricanything-student-pointmap)

</details>


## Citation

```bibtex
@article{metricanything2026,
  title={MetricAnything: Scaling Metric Depth Pretraining with Noisy Heterogeneous Sources},
  author={Baorui Ma, Jiahui Yang, Donglin Di, Xuancheng Zhang, Jianxun Cui, Hao Li, Xie Yan and Wei Chen},
  journal={arXiv preprint},
  year={2026}
}
```
