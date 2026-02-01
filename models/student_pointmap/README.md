
### Model Card
- **Hugging Face Model**: [yjh001/metricanything_student_pointmap](https://huggingface.co/yjh001/metricanything_student_pointmap)
- **Base Model**: MoGe-2 ViT-l (finetuned)

### Quick Start

1. **Install dependencies**: Follow [MoGe-2](https://github.com/microsoft/MoGe)
```bash
cd models/student_pointmap
pip install -r requirements.txt
```

2. **Run inference** (with [Hugging Face model 🤗](https://huggingface.co/yjh001/metricanything_student_pointmap)):
```bash
bash infer.sh
```

### Acknowledgments

We thank the [MoGe-2](https://github.com/microsoft/MoGe) team for their excellent work. If you use this model, please also consider citing MoGe-2:

```bibtex
@misc{wang2025moge2,
      title={MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details}, 
      author={Ruicheng Wang and Sicheng Xu and Yue Dong and Yu Deng and Jianfeng Xiang and Zelong Lv and Guangzhong Sun and Xin Tong and Jiaolong Yang},
      year={2025},
      eprint={2507.02546},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.02546}, 
}
```
