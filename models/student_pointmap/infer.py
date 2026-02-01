import argparse
import itertools
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from moge.model.v2 import MoGeModel
from moge.utils.io import save_glb, save_ply
from moge.utils.vis import colorize_depth, save_plt
import utils3d

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='example_images')
    parser.add_argument('--output', type=str, default='out_infer')
    parser.add_argument('--weights', type=str, default='yjh001/metricanything_student_pointmap')
    parser.add_argument('--save_glb', action='store_true')
    parser.add_argument('--save_ply', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MoGeModel.from_pretrained(args.weights).to(device)
    model.eval()

    input_path = Path(args.input)
    output_base_dir = Path(args.output)

    include_suffices = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']
    if input_path.is_dir():
        image_paths = sorted(itertools.chain(*(input_path.rglob(f'*.{suffix}') for suffix in include_suffices)))
    else:
        image_paths = [input_path]

    for image_path in tqdm(image_paths):
        try:
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                continue
            
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            height, width = image_rgb.shape[:2]
            image_tensor = torch.tensor(image_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)

            fov_x = None
            with torch.no_grad():
                output = model.infer(image_tensor, fov_x=fov_x)

            if input_path.is_dir():
                relative_path = image_path.relative_to(input_path)
                output_sub_dir = output_base_dir / relative_path.parent / relative_path.stem
            else:
                output_sub_dir = output_base_dir / image_path.stem

            output_sub_dir.mkdir(parents=True, exist_ok=True)

            points = output['points'].cpu().numpy()
            depth = output['depth'].cpu().numpy()
            mask = output['mask'].cpu().numpy()
            intrinsics = output['intrinsics'].cpu().numpy()

            cv2.imwrite(str(output_sub_dir / 'image.jpg'), image_bgr)
            
            vis_depth = colorize_depth(depth, cmap='turbo_r')
            if isinstance(vis_depth, torch.Tensor):
                vis_depth = vis_depth.cpu().numpy()
            cv2.imwrite(str(output_sub_dir / 'depth_vis_turbo.png'), cv2.cvtColor(vis_depth, cv2.COLOR_RGB2BGR))
            
            save_plt(output_sub_dir, depth, mask, 'depth_vis_bar.png')
            cv2.imwrite(str(output_sub_dir / 'mask.png'), (mask * 255).astype(np.uint8))

            fov_x, fov_y = utils3d.np.intrinsics_to_fov(intrinsics)
            with open(output_sub_dir / 'fov.json', 'w') as f:
                json.dump({
                    'fov_x': round(float(np.rad2deg(fov_x)), 2),
                    'fov_y': round(float(np.rad2deg(fov_y)), 2),
                }, f, indent=4)

            if args.save_glb or args.save_ply:
                mask_cleaned = mask & ~utils3d.np.depth_map_edge(depth, rtol=0.1)
                faces, vertices, vertex_colors, vertex_uvs = utils3d.np.build_mesh_from_map(
                    points,
                    image_rgb.astype(np.float32) / 255.0,
                    utils3d.np.uv_map(height, width),
                    mask=mask_cleaned,
                    tri=True
                )
                
                vertices = vertices * np.array([1, -1, -1])
                vertex_uvs = vertex_uvs * np.array([1, -1]) + np.array([0, 1])
                
                if args.save_glb:
                    save_glb(output_sub_dir / 'mesh.glb', vertices, faces, vertex_uvs, image_rgb, None)
                if args.save_ply:
                    save_ply(output_sub_dir / 'pointcloud.ply', vertices, np.zeros((0, 3), dtype=np.int32), vertex_colors, None)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

if __name__ == '__main__':
    main()