# AirStack Conda Environment

This directory contain examples of conda `.yml` files to create Python virtual
environments that will work on the AIR-T. AirStack is the operating system, firmware, and
drivers for Deepwave's AIR-T and it is based on NVIDIA's JetPack.

*Note*: TensorRT is not included in the provided conda file. If you intend to use TensorRT from within a conda environment you will likely have to build from source on the AIR-T to match the TensorRT version pre installed on the device. Deepwave recommends using TensorRT outside of conda or from within a docker container that uses the same version.

## Provided Conda Environments
We provide a baseline AIR-T conda environment file to help users get started:

* `airstack.yml` - basic airstack environment with radio drivers and inference packages

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

