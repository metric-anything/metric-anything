python infer.py \
    --image_path example_images \
    --output_path output_infer \
    --pretrained yjh001/metricanything_student_depthmap
    # --f_px 1000 # indicate the GT focal length in pixels (or finetune --f_px if GT focal length is not available)

# we provide some example images (with GT focal length, check xxx.json) in `example_images` folder, you can run the script to get the metric depth map