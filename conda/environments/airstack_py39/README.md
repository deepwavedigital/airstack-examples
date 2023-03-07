# AirStack Conda Environment for Python 3.9

This directory contains an example of using python 3.9 in an AirStack environment. NVIDIA
ships Jetpack 4.6 with TensorRT support for up to python 3.6. It is possible to build
TensorRT from source for newer versions of python [[ref](https://forums.developer.nvidia.com/t/tensorrt-on-jetson-with-python-3-9/196131/9)].

This directory provides the compiled .whl files for the following packages that may
be used within a python 3.9 environment:
* pyCUDA
* pyTools
* TensorRT

## Installation
1. Copy all files from this directory to your AIR-T
2. Login to the AIR-T and cd into the directory
3. Create the `airstack-py39` conda environment. This can take a few minutes
    ```
    conda env create -f airstack-py39.yml
    ```
4. Activate the conda environment
    ```
    conda activate airstack-py39
    ```

### Customize Environment
You may add packages to the .yml file and then update the existing environment using the
following command:
   ```
   conda activate airstack-py39
   conda env update -f airstack-py39.yml --prune
   ```

Many .whl and installation file for common ML tools may be found in the 
[Jetson Zoo](https://elinux.org/Jetson_Zoo). Make sure that the tools you download are
built for the version of JetPack installed on your AIR-T. 

## Notes
1. The CUDA version will still be pinned to 10.2.