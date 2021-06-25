# AIR-T Deep Learning Inference Examples

<p align="center">
<img src="https://deepwavedigital.com/media/images/dwd2_crop_transparent.png" Width="50%" />
</p>

&nbsp;

These scripts demonstrate the recommended training to inference workflow for deploying a neural
network on the AIR-T. The example neural network model inputs an arbitrary length input
buffer and has only one output node that calculates the average of the instantaneous
power across each batch for the input buffer.

The toolbox demonstrates how to create a simple neural network for
[Tensorflow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), and
[MATLAB](https://www.mathworks.com/products/matlab.html) on a host computer.
Installation of these packages for training is made easy by the inclusion .yml file to create an
[Anaconda](https://www.anaconda.com/products/individual) environment. For the inference execution,
all python packages and dependencies are pre-installed on AIR-Ts running AirStack 0.3.0+.


## Author
This software is written by [**Deepwave Digital, Inc.**]([www.deepwavedigital.com)

General company contact: [https://deepwavedigital.com/inquiry](https://deepwavedigital.com/inquiry/)

&nbsp;

## Training to Deployment Workflow Overview
The figure below outlines the workflow for training, optimizing, and deploying a neural
network on the AIR-T. All python packages and dependencies are included on the AirStack
0.3.0+, which is the API for the [AIR-T](https://deepwavedigital.com/hardware-products/sdr/).

By default, the toolbox will use PyTorch. This can be changed by updating the file folder
definitions in the python scripts.

* **Step 1: Train** - In this toolbox, the neural network performs a simple mathematical
 calculation instead of being trained on data. Therefore the user must modify the
 `make_avg_pow_net.py` file to adapt the code base to their specific application. This
 toolbox provides all of the necessary infrastructure to guide the user in the training
 to deployment workflow by example. The output of the training process is a file
 containing the neural network in the [Open Neural Network Exchange](https://onnx.ai/)
 (ONNX) file format<sup>1</sup>. While this toolbox creates a neural network, but does
 not do any training, the process will be the exact same for ONNX files that contain a trained
 neural network.

* **Step 2: Optimize** - Optimize the neural network model using NVIDIA's TensorRT. The
 output of this step is a PLAN file containing the optimized network for deployment on
 the AIR-T.

* **Step 3: Deploy** - The final step is to deploy the optimized neural network on the
 AIR-T for inference. This toolbox accomplishes this task by leveraging the GPU/CPU
 shared memory interface on the AIR-T to receive samples from the receiver and feed the
 neural network using Zero Copy, i.e., no device-to-host or host-to device copies are
 necessary.

&nbsp;

<p align="center">
<img src="https://deepwavedigital.com/wp-content/uploads/2019/06/Flow-diagram-1500x702.jpg" Width="75%" />
</p>

&nbsp;

## Step 1: Train Neural Network
This toolbox provides examples for both TensorFlow and PyTorch. The training process
should be run on a computer with TensorFlow and/or PyTorch installed. Since there is no
training involved, any computer will work, i.e., no GPU required. Note that the AIR-T does
not come pre-installed with TensorFlow or PyTorch, however users may be able to install it
on the platform.

### Install TensorFlow and PyTorch
To build the neural networks using PyTorch or TensorFlow, a .yml file is provided to create the
Anaconda environment. This is for creating the training environment on the host computer, not the
AIR-T. To create the Anaconda environment, open a terminal and type:

```
conda env create -f infer_test_env.yml
```

and activate the environment:

```
conda activate infer_test_env
```

This toolbox has been tested on Linux and Mac OSX, but is expected to run on Window as well.

### Create Neural Network
Run the `make_avg_pow_net.py` script and it will create an ONNX file that contains a 
neural network with input dimensions `(batch_size, input_len)` and output dimensions 
`(batch_size, 1)`. The output is the average power for each batch. It uses a formulation
similar to:

```
output = sum( signal * conjugate(signal) ) / N
```

However, instead of computing the summation above, the first layer squares each value of
the interleaved I/Q input array:

```
layer1 = torch.pow(batch_x, 2.0)
```

and the second layer uses a matrix multiplication with a normalization factor to compute
the average power.

```
norm_factor = 1.0 / (float(input_len) / 2.0)
norm_matrix = np.ones((input_len, 1)) * norm_factor
norm_matrix = torch.tensor(norm_matrix, dtype=torch.float32)
layer2 = torch.mm(layer1, self.norm_matrix)
```

The above shows the neural network structure is for PyTorch. A TensorFlow implementation is 
also provided in the toolbox for the equivalent layers. The next two sections show how to
execute the code in PyTorch and TensorFlow. Note that TensorFlow requires the the output
node name and the output port name to be defined, while PyTorch only requires the node name.

After running the script an `avg_pow_net.onnx` file is created that can be used on the 
AIR-T to create a PLAN file for inference. Note that the PLAN file must be created on the
platform that will be used for inference.

### Running in PyTorch
The output of the script should look like the following.

```
$ python3 pytorch/make_avg_pow_net.py
Input buffer shape: (128, 4096)
Output buffer shape: torch.Size([128, 1])
Passed Test

Network Parameters optimization and inference on AIR-T:
ONNX_FILE_NAME = 'pytorch/avg_pow_net.onnx'
INPUT_NODE_NAME = 'input_buffer'
INPUT_PORT_NAME = ''
INPUT_LEN = 4096
CPLX_SAMPLES_PER_INFER = 2048
```

### Running in TensorFlow
The output of the script should look like the following.

```
$ python3 tensorflow/make_avg_pow_net.py
Input buffer shape: (128, 4096)
Output buffer shape: (128, 1)
Passed Test

Network Parameters optimization and inference on AIR-T:
ONNX_FILE_NAME = 'tensorflow/avg_pow_net.onnx'
INPUT_NODE_NAME = 'input_buffer'
INPUT_PORT_NAME = ':0'
INPUT_LEN = 4096
CPLX_SAMPLES_PER_INFER = 2048
```

### Running in MATLAB
The MATLAB script that generates this neural net leverages the Deep Learning Toolbox. The
neural net accepts a vector of 4096 samples which are fed to both inputs of a multiplier
layer (performing a square operation), followed by a fully connected layer. The fully
connected layer has all weights set to the normalization factor and all biases set to
zero. The fully connected layer calculates the sum of all the squared elements multiplied
by the normalization factor. It outputs the scalar result, which is the average power,
i.e., `sum(norm_factor*(I.^2 + Q.^2))`.

The MATLAB Deep Learning Toolbox requires that a neural net be "trained" before setting
weights and biases. This is not required in the other frameworks which makes the MATLAB
example here a little more complicated.  Once training is complete and the weights and
biases set, the output of the neural net is found to be equal to the result of an average
power calculation. The neural network is then saved an ONNX file. Once the MATLAB neural
network is exported as an ONNX file, this file should be moved to the AIR-T to a suitable
directory.

Note: To export the trained MATLAB model to an ONNX file, you will need to install the
free MATLAB add-on [Deep Learning Toolbox Converter for ONNX Model Format](
https://www.mathworks.com/matlabcentral/fileexchange/67296-deep-learning-toolbox-converter-for-onnx-model-format)
which is not included in the Deep Learning Toolbox. 

#### Known issues with MATLAB
Some users have reported a bug with the Deep Learning Toolbox Converter for ONNX Model
Format add-on installation.  In order to function properly, the ONNX Converter requires
two files to be installed:  the `onnxmex` file and an `onnxpb.dll` file.  Both will need
to be located in the following folder for a Windows system:

```
C:\ProgramData\MATLAB\SupportPackages\R<version>\toolbox\nnet\supportpackages\onnx\+nnet\+internal\+cnn\+onnx
```
where `R<version>` is the MATLAB version, e.g., R2021a. The installation reportedly
installs the `onnxpb.dll` file in a separate folder in:
```
C:\ProgramData\MATLAB\SupportPackages\R<version>\bin\win64\
```
In the event saving an ONNX file fails, try manually copy the `onnxpb.dll` to the required
folder as described above.  The issue and the fix for this was reported
[here](https://www.mathworks.com/matlabcentral/fileexchange/67296-deep-learning-toolbox-converter-for-onnx-model-format).


## Step 2: Optimize Neural Network
The TensorRT software pre-installed on the AIR-T will map neural network layers to
optimized CUDA kernels for inference. To optimize the network, run the `onnx2plan.py`
script to convert the `avg_pow_net.onnx` file to an optimized PLAN file that can be run on
the AIR-T for inference. Note that the `onnx2plan.py` script should be run on the same 
platform that will be used for inference. Additionally, the `INPUT_PORT_NAME` variable
differs between the PyTorch and TensorFlow frameworks, so be sure to define this according
to the output from the `make_avg_power_net.py` script. After running the optimization script,
users will see the verbose output describing the neural network optimizations that are
performed. Once completed, a PLAN file will be saved to the same directory as the ONNX file
and the the final output should look as follows:

```
$ python3 onnx2plan.py

...

ONNX File Name  : pytorch/avg_pow_net.onnx
ONNX File Size  : 16777
PLAN File Name : pytorch/avg_pow_net.plan
PLAN File Size : 40052

Network Parameters inference on AIR-T:
CPLX_SAMPLES_PER_INFER = 2048
BATCH_SIZE <= 128
```

### Benchmarking the PLAN File
The `plan_bench.py` script can be run to benchmark the maximum data rate throughput of the
neural network. The result will look like the following output.

#### PyTorch Benchmark

```
$ python3 plan_bench.py

[TensorRT] VERBOSE: Deserialize required 1550590 microseconds.
TensorRT Inference Settings:
  Batch Size           : 128
  Explicit Batch       : True
  Input Layer
    Name               : input_buffer
    Shape              : (128, 4096)
    dtype              : float32
  Output Layer
    Name               : output_buffer
    Shape              : (128, 1)
    dtype              : float32
  Receiver Output Size : 524,288 samples
  TensorRT Input Size  : 524,288 samples
  TensorRT Output Size : 128 samples
Result:
  Samples Processed : 33,554,432
  Processing Time   : 42.656 msec
  Throughput        : 786.622 MSPS
  Data Rate         : 50.344 Gbit / sec
```

#### TensorFlow Benchmark

```
$ python3 plan_bench.py
[TensorRT] VERBOSE: Deserialize required 1546484 microseconds.
TensorRT Inference Settings:
  Batch Size           : 128
  Explicit Batch       : True
  Input Layer
    Name               : input_buffer:0
    Shape              : (128, 4096)
    dtype              : float32
  Output Layer
    Name               : output_buffer:0
    Shape              : (128, 1)
    dtype              : float32
  Receiver Output Size : 524,288 samples
  TensorRT Input Size  : 524,288 samples
  TensorRT Output Size : 128 samples
Result:
  Samples Processed : 33,554,432
  Processing Time   : 35.670 msec
  Throughput        : 940.692 MSPS
  Data Rate         : 60.204 Gbit / sec
```

#### MATLAB Benchmark
```
$ python plan_bench.py
[TensorRT] VERBOSE: Deserialize required 1544750 microseconds.
TensorRT Inference Settings:
  Batch Size           : 128
  Explicit Batch       : True
  Input Layer
    Name               : input_buffer
    Shape              : (128, 4096)
    dtype              : float32
  Output Layer
    Name               : const_multiplier_Add
    Shape              : (128, 1)
    dtype              : float32
  Receiver Output Size : 524,288 samples
  TensorRT Input Size  : 524,288 samples
  TensorRT Output Size : 128 samples
Result:
  Samples Processed : 33,554,432
  Processing Time   : 175.177 msec
  Throughput        : 191.546 MSPS
  Data Rate         : 12.259 Gbit / sec
```
&nbsp;

## Step 3: Deploy Application on AIR-T

**Note:** All drivers and software packages required to run inference are already installed
in the native Python environment on the AIR-T. Users should not need to add additional
packages to run this toolbox. The provided .yml file to create the conda environment for
training is not needed for inference on the AIR-T and should not be installed. 

After creating a PLAN file using `onnx2plan.py`, users may execute the the neural network
on the AIR-T by running the `run_airt_inference.py` script in the native Python environment.
This script shows, in detail, how to setup pyCUDA memory buffers, feed RF samples from the
AIR-T's receiver to the memory buffers, and execute the neural network. After running this
script, users will see the following output.

```
$ python3 run_airt_inference.py
[TensorRT] VERBOSE: Deserialize required 1557644 microseconds.
TensorRT Inference Settings:
  Batch Size           : 128
  Explicit Batch       : True
  Input Layer
    Name               : input_buffer
    Shape              : (128, 4096)
    dtype              : float32
  Output Layer
    Name               : output_buffer
    Shape              : (128, 1)
    dtype              : float32
  Receiver Output Size : 524,288 samples
  TensorRT Input Size  : 524,288 samples
  TensorRT Output Size : 128 samples
linux; GNU C++ version 7.3.0; Boost_106501; UHD_003.010.003.000-0-unknown

Receiving Data
SUCCESS! All inference output values matched expected values!
```

While it is possible to directly run the `avg_pow_net.onnx` file for inference, it is
recommended to perform a network optimization using NVIDIA's TensorRT toolbox prior to
deployment (Step 2 above). Users interested in deploying without optimization may use the 
[ONNX Runtime for Jetson Package](https://developer.nvidia.com/blog/announcing-onnx-runtime-for-jetson/)
for AirStack 0.4.0+. For the simplicity of interacting with the AirStack drivers,
we recommend using the Python Packaging instead of Docker. The Python package should be
installed in the
[AirStack Anaconda environment](http://docs.deepwavedigital.com/Tutorials/6_conda.html).

&nbsp;
---
[1] Note that the toolbox also includes the ability to convert a UFF file to a PLAN inference
file using `tensorflow/uff2plan.py`. UFF files are used in a similar way as ONNX files but
are being slowly deprecated. In addition, the number of supported layers in TensorRT is much
larger for ONNX over UFF. For these reasons, best practice is to use the ONNX format.