#!/usr/bin/env python3
"""Preprocess TartanDrive dataset."""

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Preprocessing TartanDrive from {input_path}")
    
    # Find all sequences
    sequences = []
    for item in input_path.rglob('lidar'):
        if item.is_dir():
            sequences.append(item.parent)
            
    print(f"Found {len(sequences)} sequences")
    
    for seq_dir in tqdm(sequences, desc='Sequences'):
        rel_path = seq_dir.relative_to(input_path)
        out_seq = output_path / rel_path
        out_seq.mkdir(parents=True, exist_ok=True)
        
        lidar_dir = seq_dir / 'lidar'
        pcd_files = sorted(lidar_dir.glob('*.bin')) + sorted(lidar_dir.glob('*.npy'))
        
        for pcd_file in pcd_files:
            if pcd_file.suffix == '.npy':
                points = np.load(pcd_file)
            else:
                points = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, 4)
                
            # Preprocess
            distances = np.linalg.norm(points[:, :3], axis=1)
            mask = (distances > 0.5) & (distances < 70)
            points = points[mask]
            
            mask = (points[:, 2] > -3) & (points[:, 2] < 5)
            points = points[mask]
            
            if points.shape[1] > 3:
                points[:, 3] = np.clip(points[:, 3], 0, 255) / 255.0
                
            out_file = out_seq / f'{pcd_file.stem}.bin'
            points.astype(np.float32).tofile(out_file)
            
    print("Done!")


if __name__ == '__main__':
    main()
