#!/usr/bin/env python3

import argparse
import mrcfile
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.ndimage import gaussian_filter


def icosahedron_vertices(diameter):
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    a, b = 1, phi
    verts = np.array([
        [-a,  b,  0], [ a,  b,  0], [-a, -b,  0], [ a, -b,  0],
        [ 0, -a,  b], [ 0,  a,  b], [ 0, -a, -b], [ 0,  a, -b],
        [ b,  0, -a], [ b,  0,  a], [-b,  0, -a], [-b,  0,  a]
    ])
    # Normalize and scale to diameter
    verts /= np.linalg.norm(verts[0])
    verts *= (diameter / 2)
    return verts

def euler_zyz_matrix(rot, tilt, psi):
    Rz1 = np.array([
        [np.cos(rot), -np.sin(rot), 0],
        [np.sin(rot),  np.cos(rot), 0],
        [0,            0,           1]
    ])
    Ry = np.array([
        [np.cos(tilt), 0, np.sin(tilt)],
        [0,            1, 0],
        [-np.sin(tilt),0, np.cos(tilt)]
    ])
    Rz2 = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi),  np.cos(psi), 0],
        [0,            0,           1]
    ])
    return Rz2 @ Ry @ Rz1

def fill_icosahedron(box_size, diameter, face_smooth_sigma=1.0, symmetry="I1", rot=0.0, tilt=0.0, psi=0.0):
    verts = icosahedron_vertices(diameter)

    if symmetry == "I2":
        rot_icos, tilt_icos, psi_icos = 0.0, 0.0, 0.0
    elif symmetry == "I1":
        rot_icos, tilt_icos, psi_icos = np.pi / 2, 0.0, 0.0  # Replace with desired angles
    elif symmetry == "I3":
        rot_icos, tilt_icos, psi_icos = np.pi / 2, np.deg2rad(121.7175), 0.0
    elif symmetry == "I4":
        rot_icos, tilt_icos, psi_icos = np.pi / 2, np.deg2rad(-121.7175), 0.0

    rot_zyz = euler_zyz_matrix(rot_icos, tilt_icos, psi_icos)
    verts = verts @ rot_zyz.T

    # Make additional rotations by the user provided rot, tilt, psi angles
    rot_zyz = euler_zyz_matrix(np.deg2rad(rot), np.deg2rad(tilt), np.deg2rad(psi))
    verts = verts @ rot_zyz.T

    center = np.array([box_size, box_size, box_size]) / 2
    verts += center - np.mean(verts, axis=0)
    hull = ConvexHull(verts)
    delaunay = Delaunay(verts[hull.vertices])
    grid = np.indices((box_size, box_size, box_size)).reshape(3, -1).T

    inside = delaunay.find_simplex(grid) >= 0
    arr = np.zeros((box_size, box_size, box_size), dtype=np.float32)
    arr.flat[inside] = 1.0

    # Smooth faces with Gaussian filter
    if face_smooth_sigma > 0:
        arr = gaussian_filter(arr, sigma=face_smooth_sigma)
        arr = np.clip(arr, 0.0, 1.0)

    return arr

def main():
    parser = argparse.ArgumentParser(description="Creates an icosahedron of a defined diameter in a desired box size.")
    parser.add_argument('--box_size', required=True, type=int, help="Box size of the MRC file")
    parser.add_argument('--diameter', required=True, type=int, help="Diameter of the icosahedron in pixels")
    parser.add_argument('--smooth_sigma',  default=3.0, type=int, help="Gaussian smoothing sigma for the faces (Default: 3.0)")
    parser.add_argument('--apix', type=float, default=1.0, help="Pixel size in Angstroms (Default: 1.0 A/pixel)")
    parser.add_argument('--o', type=str, default='icosahedron.mrc', help="Output MRC file name")
    parser.add_argument('--sym', default="I1", type=str,
                        help="Symmetry convention of the icosahedron (I1, I2. I3, I4; Default: I1).")
    parser.add_argument('--rot', type=float, default=0.0, help="Rot Euler angle applied to icosahedron in ZYZ Euler angle convetion (Default: 0.0 degrees)")
    parser.add_argument('--tilt', type=float, default=0.0,
                        help="Tilt Euler angle applied to icosahedron in ZYZ Euler angle convetion (Default: 0.0 degrees)")
    parser.add_argument('--psi', type=float, default=0.0,
                        help="Psi Euler angle applied to icosahedron in ZYZ Euler angle convetion (Default: 0.0 degrees)")
    args = parser.parse_args()

    arr = fill_icosahedron(args.box_size, args.diameter, args.smooth_sigma, args.sym.upper(), args.rot, args.tilt, args.psi)

    with mrcfile.new(args.output, overwrite=True) as mrc:
        mrc.set_data(arr)
        mrc.voxel_size = args.apix

if __name__ == "__main__":
    main()