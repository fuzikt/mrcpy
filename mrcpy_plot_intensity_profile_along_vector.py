#!/usr/bin/env python3

import argparse
import numpy as np
import mrcfile
import matplotlib.pyplot as plt
import sys
import csv

def parse_args():
    parser = argparse.ArgumentParser(description="Plot intensity profile along a vector in an MRC volume.")
    parser.add_argument('--i', required=True, help='Input MRC file path')
    parser.add_argument('--vector', required=True, help='Vector direction as comma-separated values x,y,z (e.g., 1,0,0)')
    parser.add_argument('--o', required=True, help='Output CSV file path')
    # Check if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()

def sample_along_vector(data, voxel_size, vector):
    shape = np.array(data.shape)
    center = (shape - 1) / 2
    n_points = (shape // 2)[0]
    vector = vector / np.linalg.norm(vector)
    max_dist = min(shape) // 2
    distances = np.linspace(0, max_dist, n_points)
    coords = center + np.outer(distances, vector)
    coords = np.clip(coords, 0, shape - 1)
    values = [data[tuple(coord.astype(int))] for coord in coords]
    angstroms = distances * float(voxel_size.x)
    return distances, angstroms, values

def main():
    args = parse_args()



    with mrcfile.open(args.i, permissive=True) as mrc:
        data = mrc.data
        voxel_size = mrc.voxel_size

    vector = np.array([float(x) for x in args.vector.split(',')[::-1]])

    distances, angstroms, values = sample_along_vector(data, voxel_size, vector)

    plt.plot(angstroms, values)
    plt.xlabel("Distance (Å)")
    plt.ylabel("Intensity")
    plt.title("Intensity Profile Along Vector")
    plt.show()

    with open(args.o, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Pixel Distance", "Distance (Å)", "Intensity"])
        for d, a, v in zip(distances, angstroms, values):
            writer.writerow([d, a, v])

if __name__ == "__main__":
    main()