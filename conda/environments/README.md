# AirStack Conda Environment

This directory contains examples of conda `.yml` files to create Python virtual
environments that will work on the AIR-T. AirStack is the operating system, firmware, and
drivers for Deepwave's AIR-T and it is based on NVIDIA's JetPack.


## Provided Conda Environments
We provide baseline AIR-T conda environment files to help users get started:

* `airstack.yml` - basic airstack environment with radio drivers
* `airstack-infer.yml` - basic airstack environment with radio drivers and inference packages including pytorch
* `airstack-infer-tf.yml` - basic airstack environment with radio drivers and inference packages including tensorflow

## Import TensorRT Note
AirStack 2.X currently includes TensorRT 8.6.1 from NVIDIA. At the time of this release the appropriate python binding file for this version is not publically available using the established conda channels or available for download directly. To build either of the inference environments you will first need to download the 8.6.1 TensorRT release from NVIDIA, extract the contents and copy the python whl file to your device. Note that this download uses cuda 12.0 vs 12.2. This incompatibility will be fixed with the AirStack 2.2 release.

The 8.6.1 release can be found [here]( https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Ubuntu-20.04.aarch64-gnu.cuda-12.0.tar.gz) and you are looking for the  "tensorrt-8.6.1-cp310-none-linux_aarch64.whl" found in the python folder. Place this file on your radio in /opt/conda-wheels/ (you will need to create this directory).

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

