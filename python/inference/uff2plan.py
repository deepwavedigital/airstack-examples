#!/usr/bin/env python3

"""
Copyright 2020, Deepwave Digital, Inc.

This script converts a uff file to a optimized plan file using NVIDIA's TensorRT. It
must be executed on the platform that will be used for inference, i.e., the AIR-T.
"""

import tensorrt as trt
import os

# Top-level inference settings.
UFF_FILE_NAME = 'avg_pow_net.uff'  # Name of input uff file
INPUT_NODE_NAME = 'input_buffer'  # Input node name defined in dnn
INPUT_NODE_DIMS = (1, 1, 4096)  # Input layer dimensions of dnn
MAX_BATCH_SIZE = 128  # Maximum batch size for which plan file will be optimized
MAX_WORKSPACE_SIZE = 1073741824  # 1 GB for example
FP16_MODE = True  # Use float16 if possible (all layers may not support this)


def main():
    # File and path checking
    plan_file = UFF_FILE_NAME.replace('.uff', '.plan')
    assert os.path.isfile(UFF_FILE_NAME), 'UFF file not found: {}'.format(UFF_FILE_NAME)
    if os.path.isfile(plan_file):
        os.remove(plan_file)

    # Setup TensorRT builder and create network
    builder = trt.Builder(trt.Logger(trt.Logger.INFO))
    network = builder.create_network()

    # Parse the UFF file
    parser = trt.UffParser()
    parser.register_input(INPUT_NODE_NAME, INPUT_NODE_DIMS)
    parser.parse(UFF_FILE_NAME, network)

    # Define DNN parameters for inference
    builder.max_batch_size = MAX_BATCH_SIZE
    builder.max_workspace_size = MAX_WORKSPACE_SIZE
    builder.fp16_mode = FP16_MODE

    # Optimize the network
    engine = builder.build_cuda_engine(network)

    # Write output plan file
    assert engine is not None, 'Unable to create TensorRT engine'
    with open(plan_file, 'wb') as file:
        file.write(engine.serialize())

    # Print information to user
    if os.path.isfile(plan_file):
        print('UFF File Name  : {}'.format(UFF_FILE_NAME))
        print('UFF File Size  : {}'.format(os.path.getsize(UFF_FILE_NAME)))
        print('PLAN File Name : {}'.format(plan_file))
        print('PLAN File Size : {}\n'.format(os.path.getsize(plan_file)))
        print('Network Parameters inference on AIR-T:')
        print('CPLX_SAMPLES_PER_INFER = {}'.format(int(INPUT_NODE_DIMS[2] / 2)))
        print('PLAN_FILE_NAME = \'{}\''.format(plan_file))
        print('BATCH_SIZE <= {}'.format(MAX_BATCH_SIZE))
    else:
        print('Result    : FAILED - plan file not created')


if __name__ == '__main__':
    main()
