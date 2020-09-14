#!/usr/bin/env python3
#
# Copyright 2020, Deepwave Digital, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility script to benchmark the data rate that a neural network will support.
"""

import numpy as np
import time
import trt_utils

# Default inference settings.
PLAN_FILE_NAME = 'pytorch/avg_pow_net.plan'  # Plan file
CPLX_SAMPLES_PER_INFER = 2048  # This should be half input_len from the neural network
BATCH_SIZE = 128  # Must be less than or equal to max_batch_size when creating plan file
NUM_BATCHES = 128  # Number of batches to run. Set to float('Inf') to run continuously
INPUT_DTYPE = np.float32


def plan_bench(plan_file_name=PLAN_FILE_NAME, cplx_samples=CPLX_SAMPLES_PER_INFER,
         batch_size=BATCH_SIZE, num_batches=NUM_BATCHES, input_dtype=INPUT_DTYPE):
    # Setup the pyCUDA context
    trt_utils.make_cuda_context()

    # Use pyCUDA to create a shared memory buffer that will receive samples from the
    # AIR-T to be fed into the neural network.
    buff_len = 2 * cplx_samples * batch_size
    sample_buffer = trt_utils.MappedBuffer(buff_len, input_dtype)

    # Set up the inference engine. Note that the output buffers are created for
    # us when we create the inference object.
    dnn = trt_utils.TrtInferFromPlan(plan_file_name, batch_size,
                                     sample_buffer)

    # Populate input buffer with test data
    dnn.input_buff.host[:] = np.random.randn(buff_len).astype(input_dtype)

    # Time the DNN Execution
    start_time = time.monotonic()
    for _ in range(num_batches):
        dnn.feed_forward()
    elapsed_time = time.monotonic() - start_time
    total_cplx_samples = cplx_samples * batch_size * num_batches

    throughput_msps = total_cplx_samples / elapsed_time / 1e6
    rate_gbps = throughput_msps * 2 * sample_buffer.host.itemsize * 8 / 1e3
    print('Result:')
    print('  Samples Processed : {:,}'.format(total_cplx_samples))
    print('  Processing Time   : {:0.3f} msec'.format(elapsed_time / 1e-3))
    print('  Throughput        : {:0.3f} MSPS'.format(throughput_msps))
    print('  Data Rate         : {:0.3f} Gbit / sec'.format(rate_gbps))


if __name__ == '__main__':
    plan_bench()
