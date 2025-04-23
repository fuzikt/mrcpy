#!/usr/bin/env python3
import argparse
import sys
import mrcfile
import numpy as np

def main():
    # Create a command-line parser
    parser = argparse.ArgumentParser(description="Pad MRC file to cubic size with average value. The origin of the new map is modified to reflect the padding - the map content appears at the same position in Chimera as in the input map.")
    parser.add_argument('--i', required=True, help="Input MRC file")
    parser.add_argument('--o', required=True, help="Output MRC file")
    parser.add_argument('--pad_value', type=str, default='avg',
                        help="Value to pad the cropped region with. Default is 'avg'.")

    # Check if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # Parse the arguments
    args = parser.parse_args()
    input_mrc_file = args.i
    output_mrc_file = args.o
    pad_value = args.pad_value

    # Read the input MRC file
    with mrcfile.open(input_mrc_file, mode='r') as inMrc:
        mrcData = inMrc.data
        voxel_size = inMrc.voxel_size
        origin = inMrc.header.origin
        nstart = inMrc.nstart

    def pad_to_cubic(mrcData, pad_value, uniform=True):
        shape = mrcData.shape
        max_dim = max(shape)
        # Find the closest larger even number
        target_size = max_dim if max_dim % 2 == 0 else max_dim + 1

        # Calculate padding for each axis
        if uniform:
            pad_width = [(int((target_size - dim) // 2), int((target_size - dim + 1) // 2)) for dim in shape]
        else:
            pad_width = [(0, target_size - dim) for dim in shape]

        # Apply padding with the average value
        padded_data = np.pad(mrcData, pad_width, mode='constant', constant_values=pad_value)

        return padded_data, pad_width

    if pad_value == 'avg':
        avg_value = np.mean(mrcData)
    else:
        try:
            avg_value = float(pad_value)
        except ValueError:
            raise ValueError("pad_value must be 'avg' or a numeric value.")

    # Pad the data and get the padding applied
    outData, pad_width = pad_to_cubic(mrcData, avg_value, uniform=True)

    origin_px_list = [origin.x / voxel_size.x, origin.y / voxel_size.y, origin.z / voxel_size.z]

    # Adjust the origin in the header by the applied padding
    for i, (pad_before, _) in enumerate(reversed(pad_width)):
        origin_px_list[i] -= pad_before

    # Write the output MRC file
    with mrcfile.new(output_mrc_file, overwrite=True) as outMrc:
        outMrc.set_data(outData)
        outMrc.update_header_from_data()
        outMrc.update_header_stats()
        outMrc.voxel_size = voxel_size
        outMrc.nstart = nstart
        outMrc.header.origin.x = origin_px_list[0] * voxel_size.x
        outMrc.header.origin.y = origin_px_list[1] * voxel_size.y
        outMrc.header.origin.z = origin_px_list[2] * voxel_size.z

if __name__ == "__main__":
    main()