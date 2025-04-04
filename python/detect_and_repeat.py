#!/usr/bin/env python3
#
# Copyright 2025, Deepwave Digital, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import sys
import argparse
import concurrent.futures
import SoapySDR
from SoapySDR import SOAPY_SDR_TX, SOAPY_SDR_RX
from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_OVERFLOW
import cupy as cp
import cupyx.scipy.signal as signal
from numba import cuda
import numpy as np
from matplotlib import pyplot as plt


def parse_command_line_arguments():
    help_formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description='Signal detector and repeater',
                                     formatter_class=help_formatter)
    parser.add_argument('-s', type=float, required=False, dest='samp_rate',
                        default=7.8128e6, help='Receiver sample rate in SPS')
    parser.add_argument('-t', type=int, required=False, dest='threshold',
                        default=5, help='Detection threshold above noise floor.')
    parser.add_argument('-f', type=float, required=False, dest='freq',
                        default=315e6, help='Receiver tuning frequency in Hz')
    parser.add_argument('-c', type=int, required=False, dest='channel',
                        default=0, help='Receiver channel')
    parser.add_argument('-g', type=str, required=False, dest='rx_gain',
                        default='agc', help='Receive Gain value in dB')
    parser.add_argument('-G', type=float, required=False, dest='tx_gain',
                        default=0, help='Transmit Gain value in dB')
    parser.add_argument('-n', type=int, required=False, dest='buff_len',
                        default=32768, help='Data buffer size (complex samples)')
    return parser.parse_args(sys.argv[1:])


class PowerDetector:
    """ Real-time power detector class for finding signals with AIR-T"""

    def __init__(self, buff, thresh_db, ntaps=65, cutoff=0.02,
                 samp_above_thresh=20):
        # Convert thresh to linear
        self._thresh_offset = 10 ** (thresh_db / 10)
        self._thresh = float('inf')
        self._samp_above_thresh = samp_above_thresh

        # Calculate filter coefficients
        filt_coef = signal.firwin(ntaps, cutoff,
                                  window=('kaiser', 0.5))
        group_delay = int(ntaps / 2)
        self._buff_len = len(buff)
        # cusignal filter returns array w/ padding so define index for signal ROI
        self._filter_roi = group_delay + cp.arange(self._buff_len, dtype=int)

        # Preallocate cupy arrays
        self._win = cp.asarray(filt_coef, dtype=cp.float32)
        self._envelope = cp.zeros(self._buff_len, dtype=cp.float32)
        self._seg_det_index = cp.zeros(self._buff_len, dtype=bool)

        # Create fifo for continuous threshold calculation
        self._fifo_len = 100
        self._fifo_index = 0
        self._bkg_sum_arr = np.full(self._fifo_len, np.inf)

        # Run detector one time to compile the CUDA kernels
        self.detect(buff)

    def plot_envelope(self, x_in=None):
        """ Plot the envelope of the most recent signal. Not that this will cause
        samples to drop if called every loop"""
        plt.figure(1)
        if x_in is not None:
            plt.plot(x_in.real, 'k', label='Signal (Real)')
            plt.plot(x_in.imag, 'g', label='Signal (Imag)')
        plt.plot(cp.asnumpy(self._envelope), 'r', label='Envelope')
        plt.plot([0, self._buff_len-1], [self._thresh, self._thresh], 'k--',
                 label='Threshold')
        plt.xlim([0, self._buff_len-1])
        plt.ylabel('Amplitude (dB)')
        plt.title('Received Signal')
        plt.legend()
        plt.show()

    def detect(self, x_in):
        # Compute the instantaneous power for the current buffer
        x_envelope = cp.abs(x_in)
        # Filter and decimate the envelope to a lower data rate
        self._envelope[:] = signal.upfirdn(self._win,
                                           x_envelope)[self._filter_roi]
        # Update threshold
        # Add summation of current envelope to the threshold fifo array
        self._bkg_sum_arr[self._fifo_index] = cp.sum(self._envelope)
        # Update fifo index for next detection window
        self._fifo_index = (self._fifo_index + 1) % self._fifo_len
        # Calculate avg background power level for the previous buffers in fifo
        bkg_avg = np.sum(self._bkg_sum_arr) / (self._fifo_len * self._buff_len)
        # Calculate new threshold value
        self._thresh = bkg_avg * self._thresh_offset

        # Calc index vector where power is above the threshold
        envelope_det_idx = self._envelope > self._thresh
        n_detections = cp.sum(envelope_det_idx)
        # Make sure at least samp_above_thresh are higher than the threshold
        if n_detections > self._samp_above_thresh:
            x_in[~envelope_det_idx] = 0  # Zero out samples below threshold
        else:
            x_in = None
        return x_in


def tx_task_fn(sdr, tx_stream, tx_sig, tx_buff_len):
    """ Transmit task that can be made a background process """
    rc = sdr.writeStream(tx_stream, [tx_sig], tx_buff_len)
    if rc.ret != tx_buff_len:
        raise IOError('Tx Error {}:{}'.format(
            rc.ret, SoapySDR.errToStr(rc.ret)))
    # print an asterisk when a signal is repeated
    print('*', end='', flush=True)


def main():
    pars = parse_command_line_arguments()

    #  Initialize the AIR-T receiver, set sample rate, gain, and frequency
    sdr = SoapySDR.Device()
    sdr.setSampleRate(SOAPY_SDR_RX, pars.channel, pars.samp_rate)
    if pars.rx_gain.lower() == 'agc':  # Turn on AGC
        sdr.setGainMode(SOAPY_SDR_RX, pars.channel, True)
    else:  # set manual gain
        sdr.setGain(SOAPY_SDR_RX, pars.channel, float(pars.rx_gain))
    sdr.setFrequency(SOAPY_SDR_RX, pars.channel, pars.freq)

    #  Initialize the AIR-T transmitter, set sample rate, gain, and frequency
    sdr.setSampleRate(SOAPY_SDR_TX, pars.channel, pars.samp_rate)
    sdr.setGain(SOAPY_SDR_TX, pars.channel, float(pars.tx_gain))
    sdr.setFrequency(SOAPY_SDR_TX, pars.channel, pars.freq)

    # Create SDR shared memory buffer, detector
    buff = cuda.mapped_array(pars.buff_len, dtype=cp.complex64)
    detr = PowerDetector(buff, pars.threshold)

    # Turn on the transmitter
    tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [pars.channel])
    sdr.activateStream(tx_stream)
    # Setup thread subclass to asynchronously execute transmit requests
    tx_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    # Turn on the receiver
    rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [pars.channel])
    sdr.activateStream(rx_stream)

    # Start processing Data
    print('Looking for signals to repeat. Press ctrl-c to exit.')
    while True:
        try:
            sr = sdr.readStream(rx_stream, [buff], pars.buff_len)  # Read data
            if sr.ret == SOAPY_SDR_OVERFLOW:  # Data was dropped
                print('O', end='', flush=True)
                continue
            detected_sig = detr.detect(buff)
            if detected_sig is not None:
                # AIR-T transmitter currently only accepts numpy arrays or lists
                tx_sig = cp.asnumpy(detected_sig)
                tx_executor.submit(tx_task_fn, sdr, tx_stream, tx_sig,
                                   pars.buff_len)
                detr.plot_envelope(buff)  # Plot the signal end envelope
        except KeyboardInterrupt:
            break
    sdr.closeStream(rx_stream)
    sdr.closeStream(tx_stream)


if __name__ == '__main__':
    main()
