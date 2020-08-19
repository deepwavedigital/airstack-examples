# Examples and Test Scripts for Inference on AIR-T
These scripts demonstrate the training to inference workflow for deploying a neural
network on the AIR-T. The example neural network model inputs an arbitrary length input
buffer and has only one output node that calculates the average of the instantaneous
power across each batch for the input buffer.

All python packages and dependencies are pre-installed on the AirStack 0.3.0+.

Note that the UFF parser does not support TensorFlow 2.0+ so we recommend using TensorFlow 1.X.

## Author
<p align="center">
<img src="https://deepwavedigital.com/media/images/dwd2_crop_transparent.png" Width="50%" />
</p>

This software is written by [**Deepwave Digital, Inc.**]([www.deepwavedigital.com).

General company contact: [https://deepwavedigital.com/inquiry](https://deepwavedigital.com/inquiry/)

&nbsp;

## Training to Deployment Workflow
The figure below outlines the workflow for training, optimizing, and deploying a neural
network on the AIR-T. All python packages and dependencies are included on the AirStack
0.3.0+, which is the API for the [AIR-T](https://deepwavedigital.com/hardware-products/sdr/).

* **Step 1: Train** - In this toolbox, the neural network performs a simple mathematical
 calculation instead of being trained on data. Therefore the user must modify the
 `make_avg_pow_net.py` file to adapt the code base to their specific application. This 
 toolbox provides all of the necessary infrastructure to guide the user in the training
 to deployment workflow by example. The process will be the exact same for uff files 
 that contain a trained neural network.

* **Step 2: Optimize** - Optimize the neural network model using NVIDIA's TensorRT. The
 output of this step is a file containing the optimized network for deployment on the 
 AIR-T.

* **Step 3: Deploy** - The final step is to deploy the optimized neural network on the 
AIR-T for inference. This toolbox accomplishes this task by leveraging the GPU/CPU 
shared memory interface on the AIR-T to receive samples from the receiver and feed the 
neural network using Zero Copy, i.e., no device-to-host or host-to device copies are 
performed. 

&nbsp;

<p align="center">
<img src="https://deepwavedigital.com/wp-content/uploads/2019/06/Flow-diagram-1500x702.jpg" Width="75%" />
</p>

&nbsp;

### Train Neural Network
Currently only TensorFlow is demonstrated with this example code. This process should be 
run on a computer with TensorFlow installed. Since there is no training involved, any 
computer with TensorFlow installed will work, i.e., no GPU required. Note that the AIR-T
does not come preinstalled with TensorFlow, however users may be able to install it on 
the platform.

Run the `make_avg_pow_net.py` script and it will create a UFF file that contains a 
neural network with input dimensions `(batch_size, input_len)` and output dimensions 
`(batch_size, 1)`. The output is the average power for each batch. It uses a formulation
similar to:

```
output = sum(signal.real^2 + signal.imag^2) / N
```

However, instead of computing the summation above, the first layer squares each value of
the interleaved I/Q input array:
```
layer1 = tf.math.pow(batch_x, 2, name='power')
```
and the second layer uses a matrix multiplication with a normalization factor to compute
the average power.
```
norm_factor = 1.0 / (float(input_len) / 2.0)
norm_matrix = tf.constant(norm_factor, shape=(input_len, 1))
layer2 = tf.matmul(layer1, norm_matrix, name=output_node_name)
```

The output of the script should look like the following output.
```
$ ./make_avg_pow_net.py
Input buffer shape: (128, 4096)
Output buffer shape: (128, 1)
Passed Test

Parameters for creating plan file on AIR-T:
UFF_FILE_NAME = 'avg_pow_net.uff'
INPUT_NODE_NAME = 'input_buffer'
INPUT_NODE_DIMS = (1, 1, 4096)
```

After running this script an `avg_pow_net.uff` file is created that can be used on the 
AIR-T to create a PLAN file for inference.

&nbsp;

### Optimize Neural Network
Using the `avg_pow_net.uff`, the `uff2plan.py` script can be used to create an optimized
network that can be run on the AIR-T for inference. Note that the `uff2plan.py`
script should be run on the same platform that will be used for inference. After running
this script, users should see the following output.

```
$ ./uff2plan.py
[TensorRT] INFO: Detected 1 inputs and 1 output network tensors.
UFF File Name  : avg_pow_net.uff
UFF File Size  : 16835
PLAN File Name : avg_pow_net.plan
PLAN File Size : 20066
```

#### Benchmarking the PLAN File
The `plan_bench.py` script can be run to benchmark the maximum throughput of the neural
network. The result will look like the following output.
```
$ ./plan_bench.py
[TensorRT] VERBOSE: Deserialize required 1554865 microseconds.
TensorRT Inference Settings:
  Batch Size           : 128
  Input Layer
    Name               : input_buffer
    Shape              : (1, 1, 4096)
    dtype              : float32
  Output Layer
    Name               : output_buffer
    Shape              : (1, 1, 1)
    dtype              : float32
  Receiver Output Size : 524,288 samples
  TensorRT Input Size  : 524,288 samples
  TensorRT Output Size : 128 samples
Result:
  Samples Processed : 33,554,432
  Processing Time   : 173.453 msec
  Throughput        : 193.450 MSPS
  Data Rate         : 12.381 Gbit / sec
```

&nbsp;

### Deploy Application on AIR-T
After creating a PLAN file using `uff2plan.py`, users may execute the the neural network
on the AIR-T by running the `run_airt_inference.py` script. This script shows, in 
detail, how to setup pyCUDA memory buffers, feed RF samples from the AIR-T's receiver
to the memory buffers, and execute the neural network. After running this script, users
will see the following output.

```
$ ./run_airt_inference.py
[TensorRT] VERBOSE: Deserialize required 1559323 microseconds.
TensorRT Inference Settings:
  Batch Size           : 128
  Input Layer
    Name               : input_buffer
    Shape              : (1, 1, 4096)
    dtype              : float32
  Output Layer
    Name               : output_buffer
    Shape              : (1, 1, 1)
    dtype              : float32
  Receiver Output Size : 524,288 samples
  TensorRT Input Size  : 524,288 samples
  TensorRT Output Size : 128 samples
linux; GNU C++ version 7.3.0; Boost_106501; UHD_003.010.003.000-0-unknown

Receiving Data
SUCCESS! All inference output values matched expected values!
```


Copyright 2020, Deepwave Digital, Inc.