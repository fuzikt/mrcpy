#!/usr/bin/env python3

import os
import argparse
import mrcfile
import numpy as np
from tqdm import tqdm
from numba import njit
from concurrent.futures import ProcessPoolExecutor, as_completed


def scale_real_part(arr, scale):
    return np.real(arr) * scale

def fourier_crop(data, bin_factor):
    shape = np.array(data.shape)
    new_shape = (shape / bin_factor).astype(int)
    fdata = np.fft.fftn(data)
    fdata = np.fft.fftshift(fdata)
    center = shape // 2
    start = center - new_shape // 2
    end = start + new_shape
    slices = tuple(slice(s, e) for s, e in zip(start, end))
    cropped = fdata[slices]
    cropped = np.fft.ifftshift(cropped)
    result = np.fft.ifftn(cropped)
    scale = 1.0 / (bin_factor ** data.ndim)
    result = scale_real_part(result, scale)
    return result.astype(np.float32)

def bin_mrc(input_path, output_path, bin_factor):
    with mrcfile.open(input_path, permissive=True) as mrc:
        data = mrc.data
        voxel_size = mrc.voxel_size
        if bin_factor == 1:
            data_binned = data.astype(np.float32)
        else:
            data_binned = fourier_crop(data, bin_factor)
        new_voxel_size = tuple(float(getattr(voxel_size, name)) * bin_factor for name in voxel_size.dtype.names)

    with mrcfile.new(output_path, overwrite=True) as out_mrc:
        out_mrc.set_data(data_binned)
        out_mrc.update_header_from_data()
        out_mrc.update_header_stats()
        out_mrc.voxel_size = new_voxel_size

def process(input_path, output_path, bin_factor, n_workers=4):
    if os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)
        files = [
            (os.path.join(input_path, fname), os.path.join(output_path, fname), bin_factor)
            for fname in sorted(os.listdir(input_path))
            if fname.lower().endswith('.mrc')
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(bin_mrc, in_file, out_file, bin_factor): in_file for in_file, out_file, bin_factor in files}
            for future in tqdm(as_completed(futures), total=len(files), desc="Processing files"):
                fname = os.path.basename(futures[future])
                try:
                    future.result()
                except Exception as e:
                    print(f'Error processing {fname}: {e}')
    else:
        bin_mrc(input_path, output_path, bin_factor)

def main():
    parser = argparse.ArgumentParser(
        description="Bin MRC file(s) using Fourier cropping method with parallelization.")
    parser.add_argument('--i', required=True, help="Input MRC file or directory with mrc files")
    parser.add_argument('--o', required=True, help="Output MRC file or directory for binned mrc files")
    parser.add_argument('--bin', type=int, required=True, help="Binning factor (integer > 0)")
    parser.add_argument('--j', type=int, default=4, help="Number of parallel workers (default: 4)")
    args = parser.parse_args()

    if args.bin < 1:
        raise ValueError("--bin must be >= 1")
    process(args.i, args.o, args.bin, args.j)

if __name__ == "__main__":
    main()