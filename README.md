# mrcpy
Python scripts for MRC file editing.

## Table of Contents
- [Install](#install)
- [mrcpy_crop_box_to_intensity.py](#mrcpy_crop_box_to_intensitypy)
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
