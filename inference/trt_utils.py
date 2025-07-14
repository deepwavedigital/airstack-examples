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

        # Handle input/output layers, binding them to buffers
        assert trt_engine.num_io_tensors == 2, \
            "This example requires a network with one input and one output"

        for i in range(trt_engine.num_io_tensors):
            name = trt_engine.get_tensor_name(i)
            mode = trt_engine.get_tensor_mode(name)
            min_shape, _, max_shape = trt_engine.get_tensor_profile_shape(
                name, 0)

            if mode == trt.TensorIOMode.INPUT:
                print(
                    f"Input layer '{name}': min/max shape {min_shape}, {max_shape}")
                assert batch_size <= max_shape[0], 'Invalid batch size'
                if batch_size != max_shape[0]:
                    warnings.warn(
                        'Unoptimized batch size detected', RuntimeWarning)
                self._batch_size = batch_size

                input_binding_index = i
                input_layer = name
                input_shape = trt_engine.get_tensor_shape(name)
                input_dtype = trt.nptype(trt_engine.get_tensor_dtype(name))
            elif mode == trt.TensorIOMode.OUTPUT:
                output_binding_index = i
                output_layer = name
                output_shape = trt_engine.get_tensor_shape(name)
                output_dtype = trt.nptype(trt_engine.get_tensor_dtype(name))
            else:
                raise RuntimeError(
                    f"Unsupported tensor mode for {name}: {mode}")

        # By default, the input and output shape do not account for the batch size.
        # For sanity, we take care of this now, and set the "N" dimension to the
        # batch size. Note that there is also a special case for the input, where
        # if the current N dimension is -1, we will have to later set the binding
        # shape when we create the inference context.
        batch_dim_index = 0
        self._explicit_batch = (input_shape[batch_dim_index] == -1)
        input_shape[batch_dim_index] = self._batch_size
        output_shape[batch_dim_index] = self._batch_size

        # Now we can sanity check the input buffer provided by the caller. The
        # size of the input buffer should match the expected size from the
        # PLAN file.
        sdr_out_size = input_buffer.host.size
        trt_in_size = trt.volume(input_shape)
        assert trt_in_size == sdr_out_size, \
            f"Plan expected {trt_in_size} but got {sdr_out_size} samples"
        self.input_buff = input_buffer

        # For the output layer, we have a specified size and type from the PLAN
        # file. We use this info to create the output buffer for inference results.
        trt_out_size = trt.volume(output_shape)
        self.output_buff = MappedBuffer(trt_out_size, output_dtype)

        # Create inference context
        self._infer_context = trt_engine.create_execution_context()
        # If the PLAN file was created using a network with an explicit batch
        # size (likely as a result of using the ONNX workflow), we have to set
        # the binding shape now so that the batch size is accounted for in the
        # execution context.
        if self._explicit_batch:
            self._infer_context.set_input_shape(
                input_layer, input_shape)

        if verbose:
            print('TensorRT Inference Settings:')
            print('  Batch Size           : {}'.format(self._batch_size))
            print('  Explicit Batch       : {}'.format(self._explicit_batch))
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
        buffers = [self.input_buff.device, self.output_buff.device]
        # Based on how the PLAN file was generated, we either have already accounted
        # for the batch size or need to specify it again.
        if self._explicit_batch:  # batch size previously accounted for
            success = self._infer_context.execute_v2(buffers)
        else:
            success = self._infer_context.execute(self._batch_size, buffers)
        print(f"Status of inference operation: success = {success}")
