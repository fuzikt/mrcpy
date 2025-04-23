#!/usr/bin/env python3
import argparse
import sys
import mrcfile
import numpy as np

def main():
    # Create a command-line parser
    parser = argparse.ArgumentParser(description="Compute max or min intensity projection of an MRC file along a specified axis.")
    parser.add_argument('--i', required=True, help="Input MRC file")
    parser.add_argument('--o', required=True, help="Output MRC file")
    parser.add_argument('--ax', default='z', choices=['x', 'y', 'z'], help="Axis for projection (x, y, or z). Default is 'z'.")
    parser.add_argument('--min', action='store_true', help="Compute minimum projection instead of maximum")

    # Check if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # Parse the arguments
    args = parser.parse_args()
    input_mrc_file = args.i
    output_mrc_file = args.o
    axis = args.ax
    minimum = args.min

    # Map axis string to axis index
    axis_map = {"x": 2, "y": 1, "z": 0}
    axis_index = axis_map[axis]

    # Read the input MRC file
    with mrcfile.open(input_mrc_file, mode='r') as inMrc:
        mrcData = inMrc.data

    # Compute max or min projection along the specified axis
    if not minimum:
        outData = np.max(mrcData, axis=axis_index)
    else:
        outData = np.min(mrcData, axis=axis_index)

    # Write the output MRC file
    with mrcfile.new(output_mrc_file, overwrite=True) as outMrc:
        outMrc.set_data(outData)

if __name__ == "__main__":
    main()