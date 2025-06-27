# AirStack Conda Environment

This directory contains examples of conda `.yml` files to create Python virtual
environments that will work on the AIR-T. AirStack is the operating system, firmware, and
drivers for Deepwave's AIR-T and it is based on NVIDIA's JetPack.


## Provided Conda Environments
We provide baseline AIR-T conda environment files to help users get started:

* `airstack.yml` - basic airstack environment with radio drivers
* `airstack-infer.yml` - basic airstack environment with radio drivers and inference packages including pytorch
* `airstack-infer-tf.yml` - basic airstack environment with radio drivers and inference packages including tensorflow

## Important TensorRT Note
AirStack 2.2+ currently includes TensorRT 10.3 from NVIDIA. At the time of this release the appropriate python binding file for this version is not publically available using the established conda channels or available for download directly. To build either of the inference environments you will first need to download the 10.3.0 TensorRT release from NVIDIA and extract the contents. 

The 10.3 release can be found [here](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.3.0/tars/TensorRT-10.3.0.26.l4t.aarch64-gnu.cuda-12.6.tar.gz). Extract the contents of the 'python' folder from this file on your radio to /opt/conda-wheels/ (you will need to create this directory). After extracting the contents Deepwave provides a bash script that can be run to create your conda environment and install and configure TensorRT.

## Installation without TensorRT
1. Copy the desired `.yml` file to your AIR-T
2. Create the conda environment. This can take a few minutes 
    ```
    conda env create -f <environment-name>
    ```
3. Activate the conda environment
    ```
    conda activate <environment-name>
    ```

## Installation with TensorRT
1. Copy the desired `.yml` file to your AIR-T
2. Create the conda environment and install tensorrt. This can take a few minutes 
    ```
    conda_tensorrt_setup.sh <environment-name>
    ```
3. Activate the conda environment
    ```
    conda activate <environment-name>
    ```
If you do not want to use the provided TensorRT install script you will need to run the following commands to install TensorRT in a conda environment. Note you will need to choose the appropriate TensorRT whl file that matches your python version:
``` 
conda activate <environment>
pip install <wheel file>
patchelf --set-rpath '$ORIGIN/../../..' "$CONDA_PREFIX/lib/python3.*/site-packages/tensorrt/tensorrt.so"
conda deactivate
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

