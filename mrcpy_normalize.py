#!/usr/bin/env python3
import argparse
import sys
import mrcfile
import numpy as np

def main():
    # Create a command-line parser
    parser = argparse.ArgumentParser(description="Normalize MRC file voxel values to mean=0 and SD=1.")
    parser.add_argument('--i', required=True, help="Input MRC file")
    parser.add_argument('--o', required=True, help="Output MRC file")

    # Check if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # Parse the arguments
    args = parser.parse_args()
    input_mrc_file = args.i
    output_mrc_file = args.o

    # Read the input MRC file
    with mrcfile.open(input_mrc_file, mode='r') as inMrc:
        mrcData = inMrc.data
        voxel_size = inMrc.voxel_size
        nstart = inMrc.nstart

    outData = mrcData - np.mean(mrcData)
    outData = outData / np.std(outData)

    # Write the output MRC file
    with mrcfile.new(output_mrc_file, overwrite=True) as outMrc:
        outMrc.set_data(outData)
        outMrc.update_header_from_data()
        outMrc.update_header_stats()
        outMrc.voxel_size = (voxel_size)
        outMrc.nstart = (nstart)

if __name__ == "__main__":
    main()