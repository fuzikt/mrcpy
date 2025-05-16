#!/usr/bin/env python3
import argparse
import mrcfile
import sys
import numpy as np


def main():
    # Create a command-line parser
    parser = argparse.ArgumentParser(
        description="Crop MRC file to a bounding box defined by a threshold of the map intensities. The origin of the new map is modified to reflect the padding - the map content appears at the same position in Chimera as in the input map.")
    parser.add_argument('--i', required=True, help="Input MRC file")
    parser.add_argument('--o', required=True, help="Output MRC file")
    parser.add_argument('--threshold', type=float, default=0.1, help="Threshold for cropping")
    parser.add_argument('--pad_factor', type=float, default=1.0, help="Pad the cropped region by this factor")
    parser.add_argument('--pad_value', type=str, default='avg',
                        help="Value to pad the cropped region with. Default is 'avg' = average value of the background (values below threshold).")
    parser.add_argument('--cubic', action='store_true', help="Make the box cubic")

    # Check if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # Parse the arguments
    args = parser.parse_args()
    input_mrc_file = args.i
    output_mrc_file = args.o
    threshold = args.threshold
    pad_factor = args.pad_factor
    make_cubic = args.cubic
    pad_value = args.pad_value

    def crop_to_threshold_even(mrcData, threshold, extend_factor=1.0, pad_value='avg', make_cubic=False):
        """
        Crop the mrcData array to include only values above the given threshold,
        expand the bounding box by the pad_factor, and ensure the resulting size
        is even along each axis. If the expanded size exceeds the original box size,
        pad the array.

        Parameters:
            mrcData (numpy.ndarray): The input 3D array.
            threshold (float): The threshold value.
            extend_factor (float): Factor by which to expand the bounding box.

        Returns:
            numpy.ndarray: The cropped and padded array with even dimensions.
            tuple: The bounding box as ((z_min, z_max), (y_min, y_max), (x_min, x_max)).
        """
        # Find the indices where the values exceed the threshold
        indices = np.argwhere(mrcData > threshold)

        # If no values exceed the threshold, return an empty array
        if indices.size == 0:
            return np.array([]), ((0, 0), (0, 0), (0, 0))

        # Compute the bounding box
        z_min, y_min, x_min = indices.min(axis=0)
        z_max, y_max, x_max = indices.max(axis=0)

        # Calculate the size of the bounding box
        z_size = z_max - z_min + 1
        y_size = y_max - y_min + 1
        x_size = x_max - x_min + 1

        if make_cubic:
            # Make the bounding box cubic
            max_size = max(z_size, y_size, x_size)
            z_min = z_min - (max_size - z_size) // 2
            z_max = z_min + max_size - 1
            y_min = y_min - (max_size - y_size) // 2
            y_max = y_min + max_size - 1
            x_min = x_min - (max_size - x_size) // 2
            x_max = x_min + max_size - 1
            z_size = max_size
            y_size = max_size
            x_size = max_size

        # Expand the bounding box by the pad_factor
        z_expand = int((z_size * (extend_factor - 1)) // 2)
        y_expand = int((y_size * (extend_factor - 1)) // 2)
        x_expand = int((x_size * (extend_factor - 1)) // 2)

        # Adjust the bounding box with expansion
        z_min = z_min - z_expand
        z_max = z_max + z_expand
        y_min = y_min - y_expand
        y_max = y_max + y_expand
        x_min = x_min - x_expand
        x_max = x_max + x_expand

        # Ensure the bounding box fits within the original array dimensions
        pad_z_min = max(0, -z_min)
        pad_z_max = max(0, z_max - mrcData.shape[0] + 1)
        pad_y_min = max(0, -y_min)
        pad_y_max = max(0, y_max - mrcData.shape[1] + 1)
        pad_x_min = max(0, -x_min)
        pad_x_max = max(0, x_max - mrcData.shape[2] + 1)

        if pad_value == 'avg':
            avg_value = np.mean(mrcData[mrcData < threshold])
        else:
            try:
                avg_value = float(pad_value)
            except ValueError:
                raise ValueError("pad_value must be 'avg' or a numeric value.")

        mrcData = np.pad(
            mrcData,
            ((pad_z_min, pad_z_max), (pad_y_min, pad_y_max), (pad_x_min, pad_x_max)),
            mode='constant', constant_values=avg_value)

        origin_pad = (z_min, y_min, x_min)

        # Recalculate bounding box after padding

        z_min = max(0, z_min)
        y_min = max(0, y_min)
        x_min = max(0, x_min)

        z_max = z_min+int((z_size * extend_factor))-1
        y_max = y_min+int((y_size * extend_factor))-1
        x_max = x_min+int((x_size * extend_factor))-1

        # Ensure even size by further adjusting the bounding box
        if (z_max - z_min + 1) % 2 != 0:
            z_max = min(mrcData.shape[0] - 1, z_max - 1)
        if (y_max - y_min + 1) % 2 != 0:
            y_max = min(mrcData.shape[1] - 1, y_max - 1)
        if (x_max - x_min + 1) % 2 != 0:
            x_max = min(mrcData.shape[2] - 1, x_max - 1)

        # Crop the array using the adjusted bounding box
        cropped_data = mrcData[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]

        return cropped_data, ((z_min, z_max), (y_min, y_max), (x_min, x_max)), origin_pad

    # Read the input MRC file
    with mrcfile.open(input_mrc_file, mode='r') as inMrc:
        mrcData = inMrc.data
        voxel_size = inMrc.voxel_size
        origin = inMrc.header.origin
        nstart = inMrc.nstart

    # Crop the data to the threshold
    outData, crop_limits, origin_pad = crop_to_threshold_even(mrcData, threshold, pad_factor, pad_value, make_cubic)

    # Adjust the origin in the header by the applied padding
    origin_px_list = [origin.x / voxel_size.x, origin.y / voxel_size.y, origin.z / voxel_size.z]
    pad_width = [origin_pad[0], origin_pad[1], origin_pad[2]]
    for i, pad_before in enumerate(reversed(pad_width)):
        origin_px_list[i] += pad_before

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
