#!/usr/bin/env python3

import argparse
import sys
from lib.mrc_tools import *
import numpy as np

def adaptive_format(value):
    if value == 0:
        return "0.0000"
    exponent = int(np.floor(np.log10(abs(value))))
    if exponent > 4 or exponent < -4:
        return f"{value:.4e}"
    else:
        return f"{value:.4f}"

def print_statistics(data, description):
    mean_value = np.mean(data)
    min_value = np.min(data)
    max_value = np.max(data)
    std_dev = np.std(data)

    print(f"\nStatistics for the {description}:")
    print(f"Min: {adaptive_format(min_value)}")
    print(f"Max: {adaptive_format(max_value)}")
    print(f"Mean: {adaptive_format(mean_value)}")
    print(f"Standard Deviation: {adaptive_format(std_dev)}")

def main():
    parser = argparse.ArgumentParser(description="MRC file voxel value statistics. Calculates mean, min, max, and standard deviation for non-zero values in MRC file.")
    parser.add_argument('--i', required=True, help="Input MRC file")

    # Check if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    inputMRCfile = args.i

    mrcData = readMrcData(inputMRCfile)

    print_statistics(mrcData, "whole MRC file")

    non_zero_values = mrcData[mrcData != 0]
    if non_zero_values.size == 0:
        non_zero_values = np.array([0])

    print_statistics(non_zero_values, "non-zero values in the MRC file")

if __name__ == "__main__":
    main()