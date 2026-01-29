# Metric Anything: Scaling Depth Pretraining with Noisy Heterogeneous Sources

<div align="center">

[Paper]() | [Project Page]()

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


## Code

- [ ] Release code

## Citation

```bibtex
@article{metricanything2025,
  title={Metric Anything: Scaling Depth Pretraining with Noisy Heterogeneous Sources},
  author={Metric Anything Team},
  journal={arXiv preprint},
  year={2026}
}
```
