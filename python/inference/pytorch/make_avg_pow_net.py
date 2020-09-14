#!/usr/bin/env python
#
# Copyright 2020, Deepwave Digital, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""
This application is to be used as a starting point for creating a deep neural network
(using PyTorch) to be deployed on the AIR-T. An extremely simple network is created
that calculates the average of the instantaneous power for an input buffer. The output
of the application is a saved onnx file that may be optimized using NVIDIA's TensorRT
tool to create a plan file. That plan file is deployable on the AIR-T.

Users may modify the layers in the NeuralNetwork class for the network that is best
suited for that application. Note that any layer that is used must be supported by
TensorRT so that it can be deployed on the AIR-T. See the AirStack API documentation
for more information on supported layers:
http://docs.deepwavedigital.com/AirStack/airstack.html
"""

import numpy as np
import torch


# Top-level neural network settings
CPLX_SAMPLES_PER_INFER = 2048  # This should be half input_len from the neural network
ONNX_FILE_NAME = 'avg_pow_net.onnx'  # File name to save network
INPUT_NODE_NAME = 'input_buffer'  # User defined name of input node
OUTPUT_NODE_NAME = 'output_buffer'  # User defined name of output node
ONNX_VERSION = 10  # the ONNX version to export the model to


class NeuralNetwork(torch.nn.Module):
    """ Simple network that computes the average power of an input signal

    The method does this by squaring each sample, summing (I^2 + Q^2), and dividing by
    the number of samples. Since it is assumed that the input is interleaved I/Q, the
    divisor is input_len / 2.
    
    Attributes:
        norm_matrix: pytorch tensor for normalization factor

    Usage example:
        network = NeuralNetwork(4096)
        result = network.forward(data):
    """
    def __init__(self, input_len):
        """Initialization class for method that defines normalization factor

        Args:
            input_len: int,  length of the input layer (real samples)

        Returns:
            An instance of the NeuralNetwork class:
        """
        super().__init__()  # Initialize superclass
        norm_factor = 1.0 / (float(input_len) / 2.0)
        norm_matrix = np.ones((input_len, 1)) * norm_factor
        # Convert normalization matrix to a tensor
        self.norm_matrix = torch.tensor(norm_matrix, dtype=torch.float32)
        
    def forward(self, batch_x):
        """Forward propagation of network

        Args:
            batch_x: tensor, interleaved I and Q, i.e., <I0, Q0, I1, Q1,...,In, Qn>
                     batch_x is of size (batch_size, input_len)

        Returns:
            output: average power tensor, i.e., avg(sig.real^2 + sig.imag^2)
        """
        # Create layer that squares each sample independently
        layer1 = torch.pow(batch_x, 2.0)
        # Create layer2 that uses a matrix multiplication to multiply each squared
        # sample by a normalization factor then sums all samples to compute the average
        # of layer1.
        layer2 = torch.mm(layer1, self.norm_matrix)

        return layer2


def passed_test(buff, result):
    """Make sure numpy calculation matches PyTorch calculation

    Returns True if the numpy calculation matches the PyTorch calculation"""
    sig = buff[:, ::2] + 1j*buff[:, 1::2]
    wlen = float(sig.shape[1])
    np_result = np.sum(sig.real**2 + sig.imag**2, axis=1, keepdims=True) / wlen
    if np_result.shape != result.shape:
        raise ValueError('Output shape mismatch: numpy = {}, pytorch = {}'.
                         format(np_result.shape, result.shape))
    return np.allclose(np_result, result)


def main():
    # Define input length of neural network. Should be 2 x number of complex samples
    input_len = 2 * CPLX_SAMPLES_PER_INFER
    
    # Create test data in range of (-1, 1) for forward propagation test
    batch_size = 128  # Batch size to use for testing and validation here
    buff = np.random.randn(batch_size, input_len).astype(np.float32)

    # Create data tensor and initialize an instance of the NeuralNetwork class
    input_data = torch.tensor(buff, dtype=torch.float32)
    network = NeuralNetwork(input_len)
    
    # Run data through network
    result = network.forward(input_data)
    if passed_test(buff, result):
        # Create ONNX file
        torch.onnx.export(network, input_data, ONNX_FILE_NAME, export_params=True,
                          opset_version=ONNX_VERSION, do_constant_folding=True,
                          input_names=[INPUT_NODE_NAME], output_names=[OUTPUT_NODE_NAME],
                          dynamic_axes={INPUT_NODE_NAME: {0: 'batch_size'},
                                        OUTPUT_NODE_NAME: {0: 'batch_size'}})
        ## Print useful information for creating a plan file
        print('\nInput buffer shape: {}'.format(buff.shape))
        print('Output buffer shape: {}'.format(result.shape))
        print('Passed Test\n')
        print('Network Parameters optimization and inference on AIR-T:')
        print('ONNX_FILE_NAME = \'{}\''.format('pytorch/' + ONNX_FILE_NAME))
        print('INPUT_NODE_NAME = \'{}\''.format(INPUT_NODE_NAME))
        print('INPUT_PORT_NAME = \'\'')
        print('INPUT_LEN = {}'.format(input_len))
        print('CPLX_SAMPLES_PER_INFER = {}'.format(CPLX_SAMPLES_PER_INFER))
    else:
        print('Failed Test')


if __name__ == '__main__':
    main()