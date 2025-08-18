#!/usr/bin/env python3
import argparse
import sys
import mrcfile
import numpy as np
import voltools


def main():
    # Create a command-line parser
    parser = argparse.ArgumentParser(description="Post process icosahedral map for model building.")
    parser.add_argument('--i', required=True, help="Input MRC file")
    parser.add_argument('--o', required=True, help="Output prefix")
    parser.add_argument('--i4to1', action='store_true',
                        help="Rotate the MRC file from I4 icosahedral symmetry to I1 (i222) ")
    parser.add_argument('--crop', type=int, default=-1,
                        help="Crop the MRC file to a cube of this size in pixels (default: -1, no cropping)")
    parser.add_argument('--norm', action='store_true', help="Normalize the MRC file to mean=0 and SD=1")
    parser.add_argument('--flip', action='store_true', help="Flip the MRC file along the x-axis")
    parser.add_argument('--shift_origin', action='store_true', help="Shift origin to the center of the map")
    parser.add_argument('--p23', action='store_true', help="Create additional copy of map converted to P23 symmetry")
    parser.add_argument('--gpu', action='store_true', help="Use GPU for map rotation (--i4to1)")

    # Check if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # Parse the arguments
    args = parser.parse_args()
    input_mrc_file = args.i
    output_mrc_prefix = args.o

    def normalize_mrc_data(mrc_data):
        """Normalize MRC data to mean=0 and SD=1."""
        return (mrc_data - np.mean(mrc_data)) / np.std(mrc_data)

    # Read the input MRC file
    with mrcfile.open(input_mrc_file, mode='r') as inMrc:
        mrcData = inMrc.data
        voxel_size = inMrc.voxel_size
        nstart = inMrc.nstart

    outData = mrcData

    if args.i4to1:
        print("Rotating MRC file from I4 icosahedral symmetry to I1 (i222)...")
        if args.gpu:
            use_device = 'gpu'
        else:
            use_device = 'cpu'
        outData = voltools.transform(outData, interpolation='linear', device=use_device,
                                     rotation=(0, -58.283, 0), rotation_units='deg', rotation_order='rzyz')

    if args.crop > 0:
        print(f"Cropping MRC file to {args.crop} pixels...")
        # make even size
        crop_size = args.crop
        if crop_size % 2 != 0:
            crop_size += 1
        z_center, y_center, x_center = np.array(outData.shape) // 2
        z_start = max(0, z_center - crop_size // 2)
        z_end = min(outData.shape[0], z_start + crop_size)
        y_start = max(0, y_center - crop_size // 2)
        y_end = min(outData.shape[1], y_start + crop_size)
        x_start = max(0, x_center - crop_size // 2)
        x_end = min(outData.shape[2], x_start + crop_size)
        outData = outData[z_start:z_end, y_start:y_end, x_start:x_end]
        output_mrc_prefix += f"_crop{crop_size}px"

    if args.norm:
        print("Normalizing MRC data to mean=0 and SD=1...")
        outData = normalize_mrc_data(outData)
        output_mrc_prefix += "_norm"

    if args.flip:
        print("Flipping MRC data along the X-axis...")
        # Flip the MRC data along the x-axis
        outData = np.flip(outData, axis=2)
        outData = np.roll(outData, 1, axis=2)  # Roll to adjust the origin of symmetry
        output_mrc_prefix += "_flip"

    if args.shift_origin:
        output_mrc_prefix += "_shift_ori"

    # Write the output MRC file
    with mrcfile.new(output_mrc_prefix + ".mrc", overwrite=True) as outMrc:
        outMrc.set_data(outData)
        outMrc.update_header_from_data()
        outMrc.update_header_stats()
        outMrc.voxel_size = (voxel_size)
        outMrc.nstart = (nstart)
        outMrc.header.nversion = 0
        if args.shift_origin:
            outMrc.header.origin.x = round(-(outData.shape[2] * voxel_size.x) / 2, 1)
            outMrc.header.origin.y = round(-(outData.shape[1] * voxel_size.y) / 2, 1)
            outMrc.header.origin.z = round(-(outData.shape[0] * voxel_size.z) / 2, 1)
        else:
            outMrc.header.origin.x = 0.0
            outMrc.header.origin.y = 0.0
            outMrc.header.origin.z = 0.0

    if args.p23:
        print("Converting MRC data to P23 symmetry...")
        outData = np.roll(outData, outData.shape[2] // 2, axis=2)
        outData = np.roll(outData, outData.shape[1] // 2, axis=1)
        outData = np.roll(outData, outData.shape[0] // 2, axis=0)
        output_mrc_prefix = output_mrc_prefix.replace("_shift_ori", "")
        output_mrc_prefix += "_p23"
        with mrcfile.new(output_mrc_prefix + ".mrc", overwrite=True) as outMrc:
            outMrc.set_data(outData)
            outMrc.update_header_from_data()
            outMrc.update_header_stats()
            outMrc.voxel_size = (voxel_size)
            outMrc.nstart = (nstart)
            outMrc.header.ispg = 195  # Set P23 symmetry
            outMrc.header.origin.x = 0.0
            outMrc.header.origin.y = 0.0
            outMrc.header.origin.z = 0.0


if __name__ == "__main__":
    main()
