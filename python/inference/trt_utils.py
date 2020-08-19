#
# Copyright 2020, Deepwave Digital, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility functions and classes for memory management and TensorRT inference on the AIR-T.
"""

import atexit
import warnings
import pycuda.driver as cuda
import tensorrt as trt


def make_cuda_context():
    """ Set up the CUDA context. We choose the first GPU and ensure the context is
    created with the flags that will allow us to use device mapped memory. """
    cuda.init()
    ctx_flags = (cuda.ctx_flags.SCHED_AUTO | cuda.ctx_flags.MAP_HOST)
    cuda_context = cuda.Device(0).make_context(ctx_flags)
    atexit.register(cuda_context.pop)  # ensure context is cleaned up


class MappedBuffer:
    """ Class that creates a device mapped memory buffer, which is the fastest type of
    memory on embedded GPUs AIR-T. Note that device mapped memory is meant for embedded
    GPUs like the one found on the AIR-T and both the host and device pointers refer to
    the same physical memory. """
    _MEM_FLAGS = cuda.host_alloc_flags.DEVICEMAP

    def __init__(self, num_elems, dtype):
        self.host = cuda.pagelocked_empty(num_elems, dtype,
                                          mem_flags=MappedBuffer._MEM_FLAGS)
        self.device = self.host.base.get_device_pointer()


class TrtInferFromPlan:
    """ Class that performs inference using TensorRT based on the plan file passed in.
    Note that this class currently only supports device mapped memory and is designed
    for complex sample input. """

    def __init__(self, plan_file, batch_size, input_buffer, verbose=True):
        """ __init__ method for class

        Args:
            plan_file : Tensor RT plan file
            batch_size : Inference batch size
            input_buffer : signal buffer from AIR-T
            verbose : print verbose information (default = False)
        """

        # Create TensorRT Engine from plan file
        logger_settings = trt.Logger(trt.Logger.VERBOSE)
        deserializer = trt.Runtime(logger_settings).deserialize_cuda_engine
        with open(plan_file, 'rb') as f:
            trt_engine = deserializer(f.read())

        # Perform data size/shape checks
        assert batch_size <= trt_engine.max_batch_size, 'Invalid batch size'
        if batch_size != trt_engine.max_batch_size:
            warnings.warn('Unoptimized batch size detected', RuntimeWarning)
        self._batch_size = batch_size

        # Setup input layer. Make sure input_buffer size matches plan file input layer
        input_layer = trt_engine[0]
        sdr_out_size = input_buffer.host.size
        input_shape = trt_engine.get_binding_shape(input_layer)
        trt_in_size = trt.volume(input_shape) * self._batch_size
        input_dtype = trt.nptype(trt_engine.get_binding_dtype(input_layer))
        assert trt_in_size == sdr_out_size, 'Plan expected {} but got {} ' \
                                            'samples'.format(trt_in_size, sdr_out_size)
        self.input_buff = input_buffer

        # Setup output layer. Plan file defines size of output layer so create buffer
        output_layer = trt_engine[1]
        output_shape = trt_engine.get_binding_shape(output_layer)
        trt_out_size = trt.volume(output_shape) * self._batch_size
        output_dtype = trt.nptype(trt_engine.get_binding_dtype(output_layer))
        self.output_buff = MappedBuffer(trt_out_size, output_dtype)

        # Create inference context
        self._infer_context = trt_engine.create_execution_context()

        if verbose:
            print('TensorRT Inference Settings:')
            print('  Batch Size           : {}'.format(self._batch_size))
            print('  Input Layer')
            print('    Name               : {}'.format(input_layer))
            print('    Shape              : {}'.format(input_shape))
            print('    dtype              : {}'.format(input_dtype.__name__))
            print('  Output Layer')
            print('    Name               : {}'.format(output_layer))
            print('    Shape              : {}'.format(output_shape))
            print('    dtype              : {}'.format(output_dtype.__name__))
            print('  Receiver Output Size : {:,} samples'.format(sdr_out_size))
            print('  TensorRT Input Size  : {:,} samples'.format(trt_in_size))
            print('  TensorRT Output Size : {:,} samples'.format(trt_out_size))


    def feed_forward(self):
        """ Forward propagate input_buffer through neural network. Call this method
        each time samples from the radio are read into the AIR-T's buffer."""
        self._infer_context.execute(self._batch_size, [self.input_buff.device,
                                                       self.output_buff.device])
