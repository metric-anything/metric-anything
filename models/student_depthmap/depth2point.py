import torch
import numpy as np
import trimesh
from PIL import Image
import os


def depth_to_pointcloud(depth, intrinsic_normalized, depth_scale=1.0):
    """
    Convert 2D depth map to 3D point cloud.
    """
    if depth.dim() == 2:
        depth = depth.unsqueeze(0)
        intrinsic_normalized = intrinsic_normalized.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    B, H, W = depth.shape
    device = depth.device

    fx = intrinsic_normalized[:, 0, 0] * W
    fy = intrinsic_normalized[:, 1, 1] * H
    cx = intrinsic_normalized[:, 0, 2] * W
    cy = intrinsic_normalized[:, 1, 2] * H

    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )

    u = u.unsqueeze(0).expand(B, -1, -1)
    v = v.unsqueeze(0).expand(B, -1, -1)

    z = depth / depth_scale

    fx = fx.view(B, 1, 1)
    fy = fy.view(B, 1, 1)
    cx = cx.view(B, 1, 1)
    cy = cy.view(B, 1, 1)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = torch.stack([x, y, z], dim=-1)

    if squeeze_output:
        points = points.squeeze(0)

    return points

def compute_edge_mask(depth, rtol=0.05):
    """
    Mimics the behavior of utils3d.np.depth_map_edge.
    Detect edges (flying points / noise) in a depth map by checking gradient
    changes between neighboring pixels.

    Args:
        depth: (H, W) numpy array
        rtol: relative tolerance threshold. Smaller values make the filtering
              more strict (more points are removed). A typical range is
              0.02 ~ 0.1.

    Returns:
        mask: (H, W) boolean array. True indicates an edge/noisy pixel that
              should be discarded.
    """

    diff_x = np.abs(np.diff(depth, axis=1, append=depth[:, -1:]))
    diff_y = np.abs(np.diff(depth, axis=0, append=depth[-1:, :]))
    
    is_edge_x = diff_x > (depth * rtol)
    is_edge_y = diff_y > (depth * rtol)
    
    edge_mask = is_edge_x | is_edge_y
    
    return edge_mask


def process_and_save(depth_path, rgb_path, output_path, K, depth_scale=1000.0, edge_rtol=0.04):
    print(f"[-] Loading depth: {depth_path}")
    if depth_path.endswith('.npy'):
        depth_np = np.load(depth_path)
    elif depth_path.endswith('.npz'):
        with np.load(depth_path) as file:
            depth_np = file['depth'].astype(np.float32)
    
    print(f"[-] Loading RGB: {rgb_path}")
    rgb_img = Image.open(rgb_path).convert('RGB')
    
    if rgb_img.size != (depth_np.shape[1], depth_np.shape[0]):
        print(f"[*] Resizing RGB to match depth: {(depth_np.shape[1], depth_np.shape[0])}")
        rgb_img = rgb_img.resize((depth_np.shape[1], depth_np.shape[0]), Image.Resampling.BILINEAR)
    rgb_np = np.array(rgb_img)

    print(f"[-] Filtering edges (rtol={edge_rtol})...")
    edge_mask = compute_edge_mask(depth_np, rtol=edge_rtol)
    
    H, W = depth_np.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_tensor = torch.from_numpy(depth_np).float().to(device)

    K = np.array(K)
    intrinsic_norm = torch.tensor([
        [K[0,0]/W, 0,       K[0,2]/W],
        [0,       K[1,1]/H, K[1,2]/H],
        [0,       0,        1       ]
    ], dtype=torch.float32).to(device)

    points_tensor = depth_to_pointcloud(depth_tensor, intrinsic_norm, depth_scale=depth_scale)
    
    points = points_tensor.cpu().numpy().reshape(-1, 3) 
    colors = rgb_np.reshape(-1, 3)                      
    edges_flat = edge_mask.reshape(-1)                  

    valid_depth = (points[:, 2] > 0) & (np.isfinite(points[:, 2]))
    clean_mask = valid_depth & (~edges_flat)

    points_clean = points[clean_mask]
    colors_clean = colors[clean_mask]

    print(f"[-] Original points: {len(points)}, Cleaned points: {len(points_clean)}")

    print(f"[-] Saving to {output_path}...")
    pcd = trimesh.PointCloud(vertices=points_clean, colors=colors_clean)
    
    pcd.export(output_path)
    print("[+] Done!")


if __name__ == "__main__":
    # check your output_infer folder, here we use the booster_0.png as example
    DEPTH_FILE = 'output_infer/example_images/booster_0/depth.npy' # .npy or .npz
    OUTPUT_FILE = "pred_depth_booster.glb" # .glb or .ply
    EDGE_RTOL = 0.05

    RGB_FILE   = 'example_images/booster_0.png'
    cam_intrinsic = json.load(open('example_images/booster_0.json'))
    fx,fy,cx,cy = cam_intrinsic['cam_in']
    
    DEPTH_SCALE = 1.0 

    INTRINSIC_K = [
        [fx, 0.0, cx], 
        [0.0, fy, cy], 
        [0.0, 0.0, 1.0]
    ]

    if os.path.exists(DEPTH_FILE):
        process_and_save(DEPTH_FILE, RGB_FILE, OUTPUT_FILE, INTRINSIC_K, DEPTH_SCALE, EDGE_RTOL)
    else:
        print("Please check input file paths.")