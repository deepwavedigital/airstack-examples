#!/usr/bin/env python3

"""
Copyright 2020, Deepwave Digital, Inc.

This application is used as an example on how to deploy a neural network for inference
on the AIR-T. The application assumes that the network has been optimized using TensorRT
to create a plan file. The method provided here leverages the pyCUDA interface for the
shared memory buffer. pyCUDA is installed by default in AirStack.
"""


import numpy as np
import trt_utils
from SoapySDR import Device, SOAPY_SDR_RX, SOAPY_SDR_CF32, SOAPY_SDR_OVERFLOW


# Top-level inference settings.
CPLX_SAMPLES_PER_INFER = 2048  # This should be half input_len from the neural network
PLAN_FILE_NAME = 'avg_pow_net.plan'  # Plan file created from uff2plan.py
BATCH_SIZE = 128  # Must be less than or equal to max_batch_size from uff2plan.py
NUM_BATCHES = 16  # Number of batches to run. Set to float('Inf') to run continuously

# Top-level SDR settings.
SAMPLE_RATE = 7.8125e6  # AIR-T sample rate
CENTER_FREQ = 2400e6  # AIR-T Receiver center frequency
CHANNEL = 0  # AIR-T receiver channel


def passed_test(buff_arr, tf_result):
    """ Make sure numpy calculation matches TensorFlow calculation. Returns True if the
     numpy calculation matches the TensorFlow calculation"""
    buff = buff_arr.reshape(BATCH_SIZE, -1)  # Reshape so first dimension is batch_size
    sig = buff[:, ::2] + 1j*buff[:, 1::2]  # Convert to complex valued array
    wlen = float(sig.shape[1])  # Normalization factor
    np_result = np.sum((sig.real**2) + (sig.imag**2), axis=1) / wlen
    return np.allclose(np_result, tf_result)


def main():
    # Setup the pyCUDA context
    trt_utils.make_cuda_context()

    # Use pyCUDA to create a shared memory buffer that will receive samples from the
    # AIR-T to be fed into the neural network. Note that this buffer is shared between
    # the SDR and the DNN to prevent unnecessary copies. The buffer fed into the AIR-T
    # for inference will be a 1 dimensional array that contains the samples for an
    # entire mini-batch with length defined below. Note that The AIR-T will produced
    # data of type SOAPY_SDR_CF32 which is the same as np.complex64. Because a
    # np.complex64 value has the same memory construct as two np.float32 values, we can
    # define the GPU memory buffer as twice the size of the SDR buffer but np.float32
    # dtypes. We do this because the neural network input layer expects np.float32. The
    # SOAPY_SDR_CF32 can be copied directly to the np.float32 buffer.
    buff_len = 2 * CPLX_SAMPLES_PER_INFER * BATCH_SIZE
    sample_buffer = trt_utils.MappedBuffer(buff_len, np.float32)

    # Set up the inference engine. Note that the output buffers are created for
    # us when we create the inference object.
    dnn = trt_utils.TrtInferFromPlan(PLAN_FILE_NAME, BATCH_SIZE, sample_buffer)

    # Create, configure and activate AIR-T's radio hardware
    sdr = Device()
    sdr.setGainMode(SOAPY_SDR_RX, CHANNEL, True)
    sdr.setSampleRate(SOAPY_SDR_RX, CHANNEL, SAMPLE_RATE)
    sdr.setFrequency(SOAPY_SDR_RX, CHANNEL, CENTER_FREQ)
    rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [CHANNEL])
    sdr.activateStream(rx_stream)
    
    # Start receiving signals and performing inference
    print('Receiving Data')
    ctr = 0
    while ctr < NUM_BATCHES:
        try:
            # Receive samples from the AIR-T buffer
            sr = sdr.readStream(rx_stream, [sample_buffer.host], CPLX_SAMPLES_PER_INFER)
            if sr.ret == SOAPY_SDR_OVERFLOW:  # Data was dropped, i.e., overflow
                print('O', end='', flush=True)
                continue
            # Run samples through neural network
            dnn.feed_forward()
            
            output_arr = dnn.output_buff.host  # Get data from DNN output layer
            """ Do something useful here with output array """
            # Run sanity check to make sure neural network output matches numpy result
            if not passed_test(sample_buffer.host, output_arr):
                raise ValueError('Neural network output does not match numpy')
        except KeyboardInterrupt:
            break
        ctr += 1
    sdr.closeStream(rx_stream)
    if ctr == NUM_BATCHES:
        print('SUCCESS! All inference output values matched expected values!')


if __name__ == '__main__':
    main()
