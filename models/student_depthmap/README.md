# MetricAnything — Student DepthMap

Monocular **metric depth estimation** with focal-length-aware inference.

---

# 🚀 Installation

```bash
cd models/student_depthmap
pip install -r requirements.txt
```

---

# 🔍 Inference  
Using the Hugging Face model:  
👉 [yjh001/metricanything_student_depthmap](https://huggingface.co/yjh001/metricanything_student_depthmap)

---

## 📐 Focal Length (`f_px`) Resolution Order

During inference, the focal length in pixels (`f_px`) is determined by the following priority:

1. **Manual override (`--f_px`)**  
   If provided, this value is used directly.

2. **Per-image JSON intrinsics**  
   If a JSON file with the same filename stem exists  
   (e.g. `image.png` → `image.json`) and contains:
   ```json
   { "cam_in": [fx, fy, cx, cy] }
   ```
   then `f_px = fx`.

3. **Fallback (default behavior)**  
   If neither of the above is available,  
   the **image width (in pixels)** is used as `f_px`.

---

## 🧪 Examples

### Use JSON intrinsics (automatic fallback if missing)

```bash
python infer.py \
  --image_path example_images \
  --output_path output_infer \
  --pretrained yjh001/metricanything_student_depthmap
```

### Manually specify focal length

```bash
python infer.py \
  --image_path example_images/1.png \
  --output_path output_infer \
  --pretrained yjh001/metricanything_student_depthmap \
  --f_px 1000
```

---

# 📊 Evaluation

## Zero-shot Evaluation on Booster Dataset

```bash
mkdir -p dataset/raw_data/eval_booster
cd dataset/raw_data/eval_booster

wget https://amsacta.unibo.it/id/eprint/6876/1/booster_gt.zip
unzip booster_gt.zip

cd -
bash eval.sh 0
```

Results will be saved to:

```
./eval/booster_xxx/metrics.json
```

Example output:

```json
{
  "d1": 0.5942887663841248,
  "d2": 0.8411244750022888,
  "d3": 0.9683409929275513,
  "abs_rel": 0.2821439206600189,
  "sq_rel": 0.15640950202941895,
  "rmse": 0.41277775168418884,
  "mae": 0.28914588689804077,
  "rmse_log": 0.302986741065979,
  "log10": 0.1004059687256813,
  "silog": 0.27337896823883057
}
```

---

# 🙏 Acknowledgments

We thank the excellent work of [Depth Pro](https://github.com/apple/ml-depth-pro/tree/main). 
If you use this repository in academic work, please consider citing:

```bibtex
@inproceedings{depthpro,
  author     = {Aleksei Bochkovskii and Ama\"{e}l Delaunoy and Hugo Germain and Marcel Santos and
               Yichao Zhou and Stephan R. Richter and Vladlen Koltun},
  title      = {Depth Pro: Sharp Monocular Metric Depth in Less Than a Second},
  booktitle  = {International Conference on Learning Representations},
  year       = {2025},
  url        = {https://arxiv.org/abs/2410.02073},
}
```