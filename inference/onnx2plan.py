#!/usr/bin/env python3
#
# Copyright 2020, Deepwave Digital, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""
This script converts an ONNX file to a optimized plan file using NVIDIA's TensorRT. It
must be executed on the platform that will be used for inference, i.e., the AIR-T.
"""

import tensorrt as trt
import os
from plan_bench import plan_bench

# Top-level inference settings.
ONNX_FILE_NAME = 'pytorch/avg_pow_net.onnx'  # Name of input onnx file
INPUT_NODE_NAME = 'input_buffer'  # Input node name defined in dnn
INPUT_PORT_NAME = ''  # ':0' for tensorflow, '' for pytorch
INPUT_LEN = 4096  # Length of the input buffer (# of elements)
MAX_BATCH_SIZE = 128  # Maximum batch size for which plan file will be optimized
MAX_WORKSPACE_SIZE = 1073741824  # 1 GB for example
FP16_MODE = False  # Use float16 if possible (all layers may not support this)
LOGGER = trt.Logger(trt.Logger.VERBOSE)
BENCHMARK = False  # Run plan file benchmarking at end


def main():
    # File and path checking
    plan_file = ONNX_FILE_NAME.replace('.onnx', '.plan')
    assert os.path.isfile(
        ONNX_FILE_NAME), 'ONNX file not found: {}'.format(ONNX_FILE_NAME)
    if os.path.isfile(plan_file):
        os.remove(plan_file)

    # Setup TensorRT builder and create network
    builder = trt.Builder(LOGGER)
    batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags=batch_flag)

    # Parse the ONNX file
    parser = trt.OnnxParser(network, LOGGER)
    parser.parse_from_file(ONNX_FILE_NAME)

    # Define DNN parameters for inference
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, MAX_WORKSPACE_SIZE)
    if FP16_MODE:
        config.set_flag(trt.BuilderFlag.FP16)

    # Optimize the network
    optimized_input_dims = (MAX_BATCH_SIZE, INPUT_LEN)
    profile = builder.create_optimization_profile()
    input_name = INPUT_NODE_NAME + INPUT_PORT_NAME
    # Set the min, optimal, and max dimensions for the input layer.
    profile.set_shape(input_name, (1, INPUT_LEN), optimized_input_dims,
                      optimized_input_dims)
    config.add_optimization_profile(profile)
    engine = builder.build_serialized_network(
        network, config)

    # Write output plan file
    assert engine is not None, 'Unable to create TensorRT engine. Check settings'
    with open(plan_file, 'wb') as file:
        file.write(engine)

    # Print information to user
    if os.path.isfile(plan_file):
        print('\nONNX File Name  : {}'.format(ONNX_FILE_NAME))
        print('ONNX File Size  : {}'.format(os.path.getsize(ONNX_FILE_NAME)))
        print('PLAN File Name : {}'.format(plan_file))
        print('PLAN File Size : {}\n'.format(os.path.getsize(plan_file)))
        print('Network Parameters inference on AIR-T:')
        print('CPLX_SAMPLES_PER_INFER = {}'.format(int(INPUT_LEN / 2)))
        print('BATCH_SIZE <= {}'.format(MAX_BATCH_SIZE))
        if BENCHMARK:
            print('Running Inference Benchmark')
            plan_bench(plan_file_name=plan_file, cplx_samples=int(INPUT_LEN/2),
                       batch_size=MAX_BATCH_SIZE)
    else:
        print('Result    : FAILED - plan file not created')


if __name__ == '__main__':
    main()
