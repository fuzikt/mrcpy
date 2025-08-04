# mrcpy
Python scripts for MRC file editing.

## Table of Contents
- [Install](#install)
- [mrcpy_create_icosahedron.py](#mrcpy_create_icosahedronpy)
- [mrcpy_crop_box_to_intensity.py](#mrcpy_crop_box_to_intensitypy)
- [mrcpy_fft_binning.py](#mrcpy_fft_binningpy)
- [mrcpy_float16to32.py](#mrcpy_float16to32py)
- [mrcpy_max_intensity_projection.py](#mrcpy_max_intensity_projectionpy)
- [mrcpy_non_zero_mean.py](#mrcpy_non_zero_meanpy)
- [mrcpy_normalize.py](#mrcpy_normalizepy)
- [mrcpy_pad_to_cubic.py](#mrcpy_pad_to_cubicpy)

## Install
To install the scripts, clone the repository and add the path to the scripts to your PATH variable.

Some of the script need external libraries listed in requirements.txt. Install them ideally into separate virtual environment:
```
# Clone the repository
git clone https://github.com/fuzikt/mrcpy.git

# Change to the directory
cd mrcpy

# Create a virtual environment
python -m venv mrcpy_env

# Activate the virtual environment
source mrcpy_env/bin/activate

# Install requirements from the requirements.txt file
pip install -r requirements.txt
```

## mrcpy_create_icosahedron.py
Creates an icosahedron of a defined diameter in a desired box size.
```
    --box_size          Box size of the MRC file")
    --diameter          Diameter of the icosahedron in pixels")
    --smooth_sigma      Gaussian smoothing sigma for the faces (Default: 3.0)")
    --apix              Pixel size in Angstroms (Default: 1.0 A/pixel)")
    --o                 Output MRC file name")
    --sym                Symmetry convention of the icosahedron (I1, I2. I3, I4; Default: I1).")
    --rot               Rot Euler angle applied to icosahedron in ZYZ Euler angle convetion (Default: 0.0 degrees)")
    --tilt              Tilt Euler angle applied to icosahedron in ZYZ Euler angle convetion (Default: 0.0 degrees)")
    --psi               Psi Euler angle applied to icosahedron in ZYZ Euler angle convetion (Default: 0.0 degrees)")
```

## mrcpy_crop_box_to_intensity.py
Crop MRC file to a bounding box defined by a threshold of the map intensities. The origin of the new map is modified to reflect the padding - the map content appears at the same position in Chimera as in the input map.
```
  --i            Input MRC file
  --o            Output MRC file
  --threshold    Threshold for cropping
  --pad_factor   Pad the cropped region by this factor
  --pad_value    Value to pad the cropped region with. Default is 'avg'.
  --cubic        Make the box cubic
```

## mrcpy_fft_binning.py
Binning of MRC file(s) using Fourier cropping method with parallelization.
```
    --i     Input MRC file or directory with mrc files
    --o     Output MRC file or directory for binned mrc files
    --bin   Binning factor (integer > 0)
    --j     Number of parallel workers (default: 4)
```

## mrcpy_float16to32.py
Converts MRC file(s) from float16 format to float32.
```
    --i     Input MRC file or directory with mrc files
    --o     Output MRC file or directory for converted mrc files
```

## mrcpy_max_intensity_projection.py
Compute max or min intensity projection of an MRC file along a specified axis.
```
  --i     Input MRC file
  --o     Output MRC file
  --ax    Axis for projection (x, y, or z). Default is 'z'.
  --min   Compute minimum projection instead of maximum
```

## mrcpy_non_zero_mean.py
MRC file voxel value statistics. Calculates mean, min, max, and standard deviation for non-zero values in MRC file.
```
  --i    Input MRC file
```

## mrcpy_normalize.py
Normalize MRC file voxel values to mean=0 and SD=1.
```
  --i   Input MRC file
  --o   Output MRC file
```

## mrcpy_pad_to_cubic.py
Pad MRC file to cubic size with average value. The origin of the new map is modified to reflect the padding - the map content appears at the same position in Chimera as in the input map.
```
  --i           Input MRC file
  --o           Output MRC file
  --pad_value   Value to pad the cropped region with. Default is 'avg'.
```

## mrcpy_postprocess_icos_map.py
Post process icosahedral map (i222 sym) for model building.
```    
    --i                 Input MRC file
    --o                 Output prefix
    --crop              Crop the MRC file to a cube of this size in pixels (default: -1, no cropping)
    --norm              Normalize the MRC file to mean=0 and SD=1
    --flip              Flip the MRC file along the x-axis
    --shift_origin      Shift origin to the center of the map
    --p23               Create additional copy of map converted to P23 symmetry
```