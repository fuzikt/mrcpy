#!/usr/bin/env python3
"""mrcpy_rotational_average.py

Compute rotational average of a 2D or 3D MRC file around a specified axis.

Usage: python mrcpy_rotational_average.py --i input.mrc --o output.mrc [--axis x,y,z]

The axis should be provided as three comma-separated floats (e.g. 0,0,1).
Default axis is (0,0,1) (z axis).

Behavior:
- For 2D input (ny,nx): the script treats the axis as passing through the center of the image
  pointing in the given direction; it computes the distance of each pixel to the axis (in-plane)
  and averages pixels by integer-radius bins. The output is the same shape where each pixel is
  replaced by the average of all pixels at the same radial bin.
- For 3D input (nz,ny,nx): the script computes cylindrical coordinates around the axis whose
  origin is the volume center. It preserves the coordinate along the axis (z') and averages
  over the azimuthal angle; each voxel is replaced by the average of voxels that share the
  same radius (perpendicular distance to the axis) and the same axial coordinate (rounded to
  nearest voxel index along the axis).

Notes:
- The radial bins are integer-radius bins (0..max) so values within the same bin are averaged.
- The script requires numpy and mrcfile (listed in requirements.txt)."""

import argparse
import numpy as np
import mrcfile

def parse_axis(axis_str: str) -> np.ndarray:
    parts = axis_str.split(',')
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--axis must be three comma-separated numbers, e.g. 0,0,1")
    try:
        vec = np.array([float(p) for p in parts], dtype=float)
    except Exception:
        raise argparse.ArgumentTypeError("Axis components must be floats")
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise argparse.ArgumentTypeError("Axis vector must be non-zero")
    return vec / norm


def compute_rotational_average_2d(data: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Compute rotational average for a 2D array.

    We assume axis passes through the center of the image. For in-plane images, the axis
    direction only matters for centering; a vector along z (0,0,1) is the default and
    essentially means rotation in XY plane about image center. We'll compute radial distance
    from the center and average by integer-radius bins.
    """
    if data.ndim != 2:
        raise ValueError("data must be 2D for compute_rotational_average_2d")

    ny, nx = data.shape
    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0

    y = np.arange(ny) - cy
    x = np.arange(nx) - cx
    X, Y = np.meshgrid(x, y)

    # distance to center in pixels
    R = np.sqrt(X * X + Y * Y)
    # bin by nearest integer radius
    Rbin = np.rint(R).astype(np.int64)
    maxbin = int(Rbin.max())

    out = np.zeros_like(data, dtype=data.dtype)

    for r in range(maxbin + 1):
        mask = (Rbin == r)
        if not np.any(mask):
            continue
        meanval = data[mask].mean()
        out[mask] = meanval

    return out


def compute_rotational_average_3d(data: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Compute rotational average for a 3D array around the given axis passing through the
    volume center.

    Approach:
    - Represent each voxel position relative to center.
    - For each position, compute axial coordinate (projection along axis, continuous) and
      radial distance (perpendicular distance to axis).
    - Convert axial coordinate to nearest voxel index along axis direction by projecting the
      position onto the axis and then mapping onto an integer index using spacing=1.
    - Bin by (rbin, zidx) where rbin = round(radial distance), zidx = round((proj) + center_offset).
    - Average values within each (rbin, zidx), assign averaged value back to those voxels.

    Note: This keeps the same shape as input.
    """
    if data.ndim != 3:
        raise ValueError("data must be 3D for compute_rotational_average_3d")

    nz, ny, nx = data.shape
    cz = (nz - 1) / 2.0
    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0

    # Create coordinates
    z = np.arange(nz) - cz
    y = np.arange(ny) - cy
    x = np.arange(nx) - cx
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')

    # Position vectors
    # shape (3, nz, ny, nx)
    pos = np.stack((X, Y, Z), axis=0).astype(float)

    # axis unit vector
    a = axis.astype(float)

    # projection length along axis (signed)
    proj = a[0] * pos[0] + a[1] * pos[1] + a[2] * pos[2]

    # vector parallel component = proj * a
    # perpendicular vector = pos - parallel_component
    par = np.empty_like(pos)
    par[0] = proj * a[0]
    par[1] = proj * a[1]
    par[2] = proj * a[2]
    perp_x = pos[0] - par[0]
    perp_y = pos[1] - par[1]
    perp_z = pos[2] - par[2]

    R = np.sqrt(perp_x * perp_x + perp_y * perp_y + perp_z * perp_z)

    # bin radial distances and axial positions
    Rbin = np.rint(R).astype(np.int64)

    # Map projection coordinate to axial index (centered): proj is in same units (voxels)
    # We want an integer index for grouping; shift by center index cz
    zidx = np.rint(proj + cz).astype(np.int64)

    # clamp zidx to valid range
    zidx = np.clip(zidx, 0, nz - 1)

    maxr = int(Rbin.max())

    out = np.zeros_like(data, dtype=data.dtype)

    # We'll iterate over zidx and r; to be efficient, group by flat indices
    # Create linear indices for grouping
    # For each voxel we have (rbin, zidx)
    # Encode pair into single index: idx = zidx * (maxr+1) + rbin
    code = zidx * (maxr + 1) + Rbin
    code_flat = code.ravel()
    data_flat = data.ravel()

    # Compute means per code
    # sort by code
    order = np.argsort(code_flat)
    code_sorted = code_flat[order]
    data_sorted = data_flat[order]

    # find unique codes and slice ranges
    unique_codes, idx_start, counts = np.unique(code_sorted, return_index=True, return_counts=True)

    # create array to hold mean for each unique code
    means = np.empty(unique_codes.shape, dtype=data.dtype)
    for i, start in enumerate(idx_start):
        cnt = counts[i]
        means[i] = data_sorted[start:start + cnt].mean()

    # map means back to flat volume
    # create a lookup from code -> mean (via searchsorted)
    # Note: unique_codes are sorted
    lookup_idx = np.searchsorted(unique_codes, code_flat)
    mapped = means[lookup_idx]
    out = mapped.reshape(data.shape)

    return out


def compute_spherical_average_3d(data: np.ndarray) -> np.ndarray:
    """Compute spherical (shell) average for a 3D array around the volume center.

    Each voxel is assigned the mean of all voxels that lie at the same integer radial
    distance (rounded) from the center.
    """
    if data.ndim != 3:
        raise ValueError("data must be 3D for compute_spherical_average_3d")

    nz, ny, nx = data.shape
    cz = (nz - 1) / 2.0
    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0

    z = np.arange(nz) - cz
    y = np.arange(ny) - cy
    x = np.arange(nx) - cx
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')

    R = np.sqrt(X * X + Y * Y + Z * Z)
    Rbin = np.rint(R).astype(np.int64)
    code_flat = Rbin.ravel()
    data_flat = data.ravel()

    order = np.argsort(code_flat)
    code_sorted = code_flat[order]
    data_sorted = data_flat[order]
    unique_codes, idx_start, counts = np.unique(code_sorted, return_index=True, return_counts=True)

    means = np.empty(unique_codes.shape, dtype=data.dtype)
    for i, start in enumerate(idx_start):
        cnt = counts[i]
        means[i] = data_sorted[start:start + cnt].mean()

    lookup_idx = np.searchsorted(unique_codes, code_flat)
    mapped = means[lookup_idx]
    out = mapped.reshape(data.shape)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rotational average of 2D/3D MRC around an axis")
    parser.add_argument('--i', '--input', required=True, dest='input', help='Input MRC file')
    parser.add_argument('--o', '--output', required=True, dest='output', help='Output MRC file')
    parser.add_argument('--axis', default='0,0,1', type=parse_axis,
                        help='Rotation axis as comma-separated floats (x,y,z). Default: 0,0,1')
    parser.add_argument('--sph', action='store_true', help='Make spherical (shell) averages instead of rotational averages')
    args = parser.parse_args(argv)

    input_path = args.input
    output_path = args.output
    axis = args.axis

    # Load MRC
    with mrcfile.open(input_path, permissive=True) as mrc:
        data = mrc.data.copy()
        header = mrc.header.copy()
        try:
            voxel_size = mrc.voxel_size
        except Exception:
            voxel_size = None

    # Compute
    if args.sph:
        # spherical (shell) averages
        if data.ndim == 2:
            # 2D spherical == radial average
            out = compute_rotational_average_2d(data, axis)
        elif data.ndim == 3:
            out = compute_spherical_average_3d(data)
        else:
            raise RuntimeError(f"Unsupported data dimensionality: {data.ndim}. Only 2D or 3D supported")
    else:
        # cylindrical / rotational averages around axis
        if data.ndim == 2:
            out = compute_rotational_average_2d(data, axis)
        elif data.ndim == 3:
            out = compute_rotational_average_3d(data, axis)
        else:
            raise RuntimeError(f"Unsupported data dimensionality: {data.ndim}. Only 2D or 3D supported")

    # Write output preserving header if possible
    with mrcfile.new(output_path, overwrite=True) as mrc_out:
        mrc_out.set_data(np.asarray(out))
        # copy select header fields where sensible
        mrc_out.update_header_from_data()
        mrc_out.update_header_stats()
        if voxel_size is not None:
            mrc_out.voxel_size = voxel_size

    print(f"Wrote rotationally averaged output to: {output_path}")


if __name__ == '__main__':
    main()
