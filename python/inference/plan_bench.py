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

# Top-level inference settings.
CPLX_SAMPLES_PER_INFER = 2048  # This should be half input_len from the neural network
PLAN_FILE_NAME = 'avg_pow_net.plan'  # Plan file created from uff2plan.py
BATCH_SIZE = 128  # Must be less than or equal to max_batch_size from uff2plan.py
NUM_BATCHES = 128  # Number of batches to run. Set to float('Inf') to run continuously
INPUT_DTYPE = np.float32


def main():
    # Setup the pyCUDA context
    trt_utils.make_cuda_context()

    # Use pyCUDA to create a shared memory buffer that will receive samples from the
    # AIR-T to be fed into the neural network.
    buff_len = 2 * CPLX_SAMPLES_PER_INFER * BATCH_SIZE
    sample_buffer = trt_utils.MappedBuffer(buff_len, INPUT_DTYPE)

    # Set up the inference engine. Note that the output buffers are created for
    # us when we create the inference object.
    dnn = trt_utils.TrtInferFromPlan(PLAN_FILE_NAME, BATCH_SIZE, sample_buffer)

    # Populate input buffer with test data
    dnn.input_buff.host[:] = np.random.randn(buff_len).astype(np.float32)

    # Time the DNN Execution
    start_time = time.monotonic()
    for _ in range(NUM_BATCHES):
        dnn.feed_forward()
    elapsed_time = time.monotonic() - start_time
    total_cplx_samples = CPLX_SAMPLES_PER_INFER * BATCH_SIZE * NUM_BATCHES

    throughput_msps = total_cplx_samples / elapsed_time / 1e6
    rate_gbps = throughput_msps * 2 * sample_buffer.host.itemsize * 8 / 1e3
    print('Result:')
    print('  Samples Processed : {:,}'.format(total_cplx_samples))
    print('  Processing Time   : {:0.3f} msec'.format(elapsed_time / 1e-3))
    print('  Throughput        : {:0.3f} MSPS'.format(throughput_msps))
    print('  Data Rate         : {:0.3f} Gbit / sec'.format(rate_gbps))


if __name__ == '__main__':
    main()
