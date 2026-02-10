#!/usr/bin/env python3
"""Preprocess RELLIS-3D dataset."""

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='RELLIS-3D root')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Preprocessing RELLIS-3D from {input_path}")
    print(f"Output: {output_path}")
    
    # Process each sequence
    sequences = sorted(input_path.glob('*/'))
    
    for seq_dir in tqdm(sequences, desc='Sequences'):
        seq_name = seq_dir.name
        
        # Find point cloud directory
        pcd_dir = seq_dir / 'os1_cloud_node_kitti_bin'
        if not pcd_dir.exists():
            pcd_dir = seq_dir / 'velodyne'
            
        if not pcd_dir.exists():
            print(f"No point cloud directory in {seq_dir}")
            continue
            
        # Create output directory
        out_seq = output_path / seq_name
        out_seq.mkdir(exist_ok=True)
        
        # Process point clouds
        pcd_files = sorted(pcd_dir.glob('*.bin'))
        
        for pcd_file in tqdm(pcd_files, desc=f'Processing {seq_name}', leave=False):
            # Load point cloud
            points = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, 4)
            
            # Basic preprocessing
            # 1. Range filter
            distances = np.linalg.norm(points[:, :3], axis=1)
            mask = (distances > 0.5) & (distances < 70)
            points = points[mask]
            
            # 2. Height filter
            mask = (points[:, 2] > -3) & (points[:, 2] < 5)
            points = points[mask]
            
            # 3. Normalize intensity
            points[:, 3] = points[:, 3] / 255.0
            
            # Save processed
            out_file = out_seq / pcd_file.name
            points.astype(np.float32).tofile(out_file)
            
    print("Preprocessing complete!")


if __name__ == '__main__':
    main()
