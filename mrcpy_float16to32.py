#!/usr/bin/env python3

import os
import argparse
import mrcfile
import numpy as np

def convert_float16_to_float32(input_path, output_path):
    with mrcfile.open(input_path, permissive=True) as mrc:
        if mrc.data.dtype != np.float16:
            print(f"Skipping {input_path}: not float16")
            return
        data32 = mrc.data.astype(np.float32)
        voxel_size = mrc.voxel_size
    with mrcfile.new(output_path, overwrite=True) as out_mrc:
        out_mrc.set_data(data32)
        out_mrc.update_header_from_data()
        out_mrc.update_header_stats()
        out_mrc.voxel_size = voxel_size


def process(input_path, output_path):
    if os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)
        for fname in sorted(os.listdir(input_path)):
            if fname.lower().endswith('.mrc'):
                print('Processing:', fname)
                in_file = os.path.join(input_path, fname)
                out_file = os.path.join(output_path, fname)
                convert_float16_to_float32(in_file, out_file)
    else:
        convert_float16_to_float32(input_path, output_path)

def main():
    parser = argparse.ArgumentParser(
        description="Convert mrc file from float16 format to float32.")
    parser.add_argument('--i', required=True, help="Input MRC file or directory with mrc files")
    parser.add_argument('--o', required=True, help="Output MRC file or directory for converted mrc files")
    args = parser.parse_args()

    process(args.i, args.o)

if __name__ == "__main__":
    main()