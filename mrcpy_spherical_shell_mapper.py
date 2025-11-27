#!/usr/bin/env python3
import argparse
import sys
import mrcfile
import numpy as np
import voltools

def trilinear_interpolate(data, x, y, z):
    x0, y0, z0 = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    xd, yd, zd = x - x0, y - y0, z - z0

    def get_val(a, b, c):
        if 0 <= a < data.shape[2] and 0 <= b < data.shape[1] and 0 <= c < data.shape[0]:
            return data[c, b, a]
        return 0.0

    c000 = get_val(x0, y0, z0)
    c001 = get_val(x0, y0, z1)
    c010 = get_val(x0, y1, z0)
    c011 = get_val(x0, y1, z1)
    c100 = get_val(x1, y0, z0)
    c101 = get_val(x1, y0, z1)
    c110 = get_val(x1, y1, z0)
    c111 = get_val(x1, y1, z1)

    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    c = c0 * (1 - zd) + c1 * zd
    return c

def spherical_shell_series(data, img_size=256, r=0, scale_to_radius=True):

    center = [s / 2 for s in data.shape[::-1]]

    shells = []

    if r > 0:
        max_radius = r+1 #do a single shell
        min_radius = r
    else:
        min_radius = 1
        max_radius = min(center)

    for radius in range(min_radius, int(max_radius)):
        if scale_to_radius:
            r_proj = (radius / max_radius) * (img_size // 2)
        else:
            r_proj = img_size // 2

        if r_proj <= 0:
            continue

        img = np.zeros((img_size, img_size), dtype=np.float32)
        cx, cy = img_size // 2, img_size // 2
        for i in range(img_size):
            for j in range(img_size):
                dx = (j - cx) / r_proj
                dy = (i - cy) / r_proj
                val = 1.0 - dx ** 2 - dy ** 2
                if val >= 0.0:
                    dz = np.sqrt(val)
                    x = center[0] + radius * dx
                    y = center[1] + radius * dy
                    z = center[2] + radius * dz
                    img[i, j] = trilinear_interpolate(data, x, y, z)
                else:
                    img[i, j] = 0
        img = (img - img.min()) / (np.ptp(img) + 1e-8)
        shells.append(img)

    shells = np.stack(shells, axis=0).astype(np.float32)
    return shells


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract intensity values by shells from the center of the map and map them on sphere.")
    parser.add_argument('--i', required=True, help="Input MRC file")
    parser.add_argument('--o', required=True, help="Output MRC file")
    parser.add_argument('--r', type=int, default=0,
                        help="Radius of the shell in pixels to extract. If 0, all the shells as series will be extracted (default: 0)")
    parser.add_argument('--rot', type=float, default=0.0,
                        help="Rotate the input map by this ROT Euler angle (ZYZ) in degrees before shell extraction (default: 0.0)")
    parser.add_argument('--tilt', type=float, default=0.0,
                        help="Rotate the input map by this TILT Euler angle (ZYZ) in degrees before shell extraction (default: 0.0)")
    parser.add_argument('--psi', type=float, default=0.0,
                        help="Rotate the input map by this PSI Euler angle (ZYZ) in degrees before shell extraction (default: 0.0)")
    parser.add_argument('--gpu', action='store_true', help="Use GPU for map rotation")

    # Check if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # Parse the arguments
    args = parser.parse_args()
    input_mrc_file = args.i
    output_mrc_prefix = args.o

    mrc_path = args.i
    radius = args.r
    out_mrc = args.o
    rot = args.rot
    tilt = args.tilt
    psi = args.psi

    #spherical_shell_orthographic_png(mrc_path, radius)
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        mrc_data = mrc.data.astype(np.float32)

    # rotate if asked
    if (rot != 0) or (tilt != 0) or (psi != 0):
        if args.gpu:
            use_device = 'gpu'
        else:
            use_device = 'cpu'
        mrc_data = np.transpose(
                voltools.transform(np.transpose(mrc_data), interpolation='linear', device=use_device,
                                   rotation=(-rot, -tilt, -psi), rotation_units='deg', rotation_order='rzyz'))

    shells = spherical_shell_series(mrc_data, r=radius, scale_to_radius=True)

    with mrcfile.new(out_mrc, overwrite=True) as mrc_out:
        mrc_out.set_data(shells)
