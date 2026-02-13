
now=$(date +"%Y%m%d_%H%M%S")
output_dir="./eval/booster_${now}"
mkdir -p "$output_dir"

pretrained=${1:-yjh001/metricanything_student_depthmap}
python eval.py \
  --pretrained "$pretrained" \
  --dataset e_booster \
  --output_dir "$output_dir" \
  --seed 42 2>&1 | tee -a "$output_dir/eval.log"
