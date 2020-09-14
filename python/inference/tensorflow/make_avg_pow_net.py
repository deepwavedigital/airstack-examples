#!/usr/bin/env python
#
# Copyright 2020, Deepwave Digital, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""
This application is to be used as a starting point for creating a deep neural network
(using TensorFlow) to be deployed on the AIR-T. An extremely simple network is created
that calculates the average of the instantaneous power for an input buffer. The output
of the application is a saved uff file that may be optimized using NVIDIA's TensorRT
tool to create a plan file. That plan file is deployable on the AIR-T.

Users may modify the layers in the neural_network function for the network that is best
suited for that application. Note that any layer that is used must be supported by
TensorRT so that it can be deployed on the AIR-T. See the AirStack API documentation
for more information on supported layers:
http://docs.deepwavedigital.com/AirStack/airstack.html
"""

import numpy as np
import tensorflow as tf
import uff


# Top-level neural network settings
CPLX_SAMPLES_PER_INFER = 2048  # This should be half input_len from the neural network
UFF_FILE_NAME = 'avg_pow_net.uff'  # File name to save network
INPUT_NODE_NAME = 'input_buffer'  # User defined name of input node
OUTPUT_NODE_NAME = 'output_buffer'  # User defined name of output node


def neural_network(batch_x, input_len, output_node_name=OUTPUT_NODE_NAME):
    """ Simple network that computes the average power of an input signal by
    squaring each sample, summing (I^2 + Q^2), and dividing by the number of samples.
    Since it is assumed that the input is interleaved I/Q, the divisor is input_len / 2.

    Args:
        batch_x: tensor, interleaved I and Q, i.e., <I0, Q0, I1, Q1,...,In, Qn>
                 batch_x is of size (batch_size, input_len)
        input_len: int,  length of the input layer (real samples)
        output_node_name: str, user defined name to be given to output node

    Returns:
        output: average power tensor, i.e., avg(sig.real^2 + sig.imag^2)
    """
    # Create tensorflow layer that squares each sample independently
    layer1 = tf.math.square(batch_x, name='hidden_layer')
    # Create layer2 that uses a matrix multiplication to multiply each squared sample by
    # a normalization factor then sums all samples to compute the average of layer1.
    norm_factor = 1.0 / (float(input_len) / 2.0)
    norm_matrix = tf.constant(norm_factor, shape=(input_len, 1))
    layer2 = tf.matmul(layer1, norm_matrix, name=output_node_name)

    return layer2


def sess2uff(sess, out_node_name=OUTPUT_NODE_NAME, uff_file_name=UFF_FILE_NAME,
             quiet=True):
    """ Converts the session to a uff file and writes it to filename """
    # Convert computational graph to protobuf (graph_def)
    graph_def = sess.graph.as_graph_def()
    # Freeze all constants. Note this is actually not necessary for this example
    # because there are no variables (tf.Variable), but we leave it hear as an example.
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def,
                                                                    [out_node_name])
    # Convert frozen graph to a uff file
    uff.from_tensorflow(frozen_graph_def, [out_node_name], quiet=quiet,
                        output_filename=uff_file_name)


def passed_test(buff, tf_result):
    """ Make sure numpy calculation matches TensorFlow calculation. Returns True if the
     numpy calculation matches the TensorFlow calculation"""
    sig = buff[:, ::2] + 1j*buff[:, 1::2]
    wlen = float(sig.shape[1])
    np_result = np.sum(sig.real**2 + sig.imag**2, axis=1, keepdims=True) / wlen
    if np_result.shape != tf_result.shape:
        raise ValueError('Output shape mismatch: numpy = {}, tensorflow = {}'.
                         format(np_result.shape, tf_result.shape))
    print(np_result[0], tf_result[0])
    return np.allclose(np_result, tf_result)


def main():
    # Define input length of neural network. Should be 2 x number of complex samples
    input_len = 2 * CPLX_SAMPLES_PER_INFER

    # Create test data in range of (-1, 1) for forward propagation test
    batch_size = 128  # Batch size to use for testing and validation here
    buff = np.random.randn(batch_size, input_len).astype(np.float32)

    # Create TensorFlow Computational Graph
    with tf.Graph().as_default() as graph:
        input_data = tf.placeholder(tf.float32, shape=(None, input_len),
                                    name=INPUT_NODE_NAME)
        network = neural_network(input_data, input_len, OUTPUT_NODE_NAME)

    # Create TensorFlow session and run data through network
    with tf.Session(graph=graph) as sess:
        result = sess.run(network, feed_dict={input_data: buff})
        if passed_test(buff, result):
            # Create UFF file
            sess2uff(sess)
            # Print useful information for creating a plan file
            print('Input buffer shape: {}'.format(buff.shape))
            print('Output buffer shape: {}'.format(result.shape))
            print('Passed Test\n')
            print('Network Parameters optimization and inference on AIR-T:')
            print('CPLX_SAMPLES_PER_INFER = {}'.format(CPLX_SAMPLES_PER_INFER))
            print('UFF_FILE_NAME = \'{}\''.format(UFF_FILE_NAME))
            print('INPUT_NODE_NAME = \'{}\''.format(INPUT_NODE_NAME))
            print('INPUT_NODE_DIMS = (1, 1, {})'.format(input_len))
        else:
            print('Failed Test')


if __name__ == '__main__':
    main()
