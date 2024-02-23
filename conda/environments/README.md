# AirStack Conda Environment

This directory contain examples of conda `.yml` files to create Python virtual
environments that will work on the AIR-T. AirStack is the operating system, firmware, and
drivers for Deepwave's AIR-T and it is based on NVIDIA's JetPack.

**Note**: JetPack 4.6 ships with Python 3.6, therefore you will be limited in what Python
packages are supported. For example, you will not be able install PyTorch > 1.10 on a
Jetson SOM unless it supports JetPack 5.0+. The NVIDIA Jetson TX2 supports up to JetPack
4.6. Nvidia has no plans to release JetPack 5.0 for the Jetson TX2 (or Nano) for now.

NVIDIA ships Jetpack 4.6 with TensorRT support for up to Python 3.6. It is possible
to build TensorRT's Python bindings from source for newer versions of Python <sup>1</sup>
and we have done that here for Python 3.6-3.9.

Also note that, even though many packages and the Python version may be updated, JetPack
4.6 always pins the CUDA version to version 10.2. Trying to install a newer version of
CUDA will likely cause the board to not function and need to be re-flashed.

## Software Compatibility Matrix for AirStack

| AirStack | JetPack | Python | TensorRT | ONNX RT | GNU Radio |    TensorFlow    |     PyTorch      |       CuPy       | CUDA |
|:--------:|:-------:|:------:|:--------:|:-------:|:---------:|:----------------:|:----------------:|:----------------:|:----:|
|  0.5.X   |   4.6   |  3.6   |    Y     |    Y    |     Y     |        Y         |        Y         | Y <sup>[2]</sup> | 10.2 |
|  0.5.X   |   4.6   |  3.7   |    Y     |    Y    |     Y     | N <sup>[3]</sup> | N <sup>[3]</sup> |        Y         | 10.2 |
|  0.5.X   |   4.6   |  3.8   |    Y     |    Y    |     Y     | N <sup>[3]</sup> | N <sup>[3]</sup> |        Y         | 10.2 |
|  0.5.X   |   4.6   |  3.9   |    Y     |    Y    |     Y     | N <sup>[3]</sup> | N <sup>[3]</sup> |        Y         | 10.2 |

### gr-wavelearner Compatibility
As of AirStack 1.0.0, gr-wavelearner no longer supports Python 3.6 and 3.7 due to the fact
that these Python versions are EOL. As a result, users who require GPU accelerated computing
via GNU Radio should use the `airstack-py39.yml` conda environment as a starting point.

### Deprecation Warning
The Deepwave team is constantly evaluating support for older Python versions as such
software maintenance is extremely taxing on our small team. As a result, all conda
environments that use Python versions that have been marked as EOL (status of various
versions can be found [here](https://devguide.python.org/versions/)) should be considered
deprecated and support can be dropped at any time. Please port your code to a newer version
of Python ASAP.

## Provided Conda Environments
We provide three baseline AIR-T conda environment files to help users get started:

* `airstack.yml` - basic airstack environment with radio drivers and inference packages
* `airstack-py36.yml` - same as `airstack.yml` but it also includes cupy, tensorflow, and pytorch
* `airstack-py39.yml` - basic Python 3.9 environment with radio drivers and inference packages

## Installation
1. Copy the desired `.yml` file to your AIR-T
2. Create the conda environment. This can take a few minutes 
    ```
    conda env create -f <yml-file-name>
    ```
3. Activate the conda environment
    ```
    conda activate <environment-name>
    ```

### Customize Environment
You may add packages to the .yml file and then update the existing environment using the
following command:
   ```
   conda activate <environment-name>
   conda env update -f <yml-file-name> --prune
   ```

Many .whl and installation file for common ML tools may be found in the 
[Jetson Zoo](https://elinux.org/Jetson_Zoo). Make sure that the tools you download are
built for the version of JetPack installed on your AIR-T.

## References
[1] [https://forums.developer.nvidia.com/t/tensorrt-on-jetson-with-python-3-9/196131/9]()<br>
[2] Built for Python 3.6 as the conda environment is created. It can take up to 30 minutes.<br>
[3] Possible to build from source.
