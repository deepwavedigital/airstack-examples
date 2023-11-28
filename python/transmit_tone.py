#!/usr/bin/env python3
#
# Copyright 2020, Deepwave Digital, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""
Transmits a tone out of the AIR-T. The script will create a tone segment that
is infinitely repeatable without a phase discontinuity and with 8 samples per
period.

The TX LO of the AIR-T is set such that the baseband frequency of the generated
tone plus the LO frequency will transmit at the desired RF.

Compatibility: This example is compatible with AirStack 0.3 and later. Earlier
               versions of AirStack firmware do not have support for
               transmitting using the SoapySDR API.
"""

import sys
import numpy as np
import argparse
import SoapySDR
from SoapySDR import SOAPY_SDR_TX, SOAPY_SDR_CS16, errToStr


def make_tone(n, fcen, fs, phi=0.285):
    """
    Generates tone signal window with a frequency that is an integer
    multiple of the sample rate so it can be repeated without a phase
    discontinuity.
    """
    period = fs / fcen
    assert n % period == 0, 'Total samples is not an integer number of periods'
    a = 2**15
    # Make Complex Valued Tone Signal
    wt = np.array(2 * np.pi * fcen * np.arange(n) / fs)
    sig_cplx = np.exp(1j * (wt + phi))
    # Convert to interleaved int16 values
    sig_int16 = np.empty(2 * n, dtype=np.int16)
    sig_int16[0::2] = 32767 * sig_cplx.real
    sig_int16[1::2] = 32767 * sig_cplx.imag
    return sig_int16


def transmit_tone(freq, chan=0, fs=31.25e6, gain=-20, buff_len=16384):
    """ Transmit a tone out of the AIR-T """

    # Setup Radio
    sdr = SoapySDR.Device()  # Create AIR-T instance
    sdr.setSampleRate(SOAPY_SDR_TX, chan, fs)  # Set sample rate
    sdr.setGain(SOAPY_SDR_TX, chan, gain)

    fs_actual = sdr.getSampleRate(SOAPY_SDR_TX, chan)
    print(f"Requested Fs: {fs}, Actual Fs: {fs_actual}")

    # Generate tone buffer that can be repeated without phase discontunity
    bb_freq = fs_actual / 8  # baseband frequency of tone
    tx_buff = make_tone(buff_len, bb_freq, fs_actual)
    lo_freq = freq - bb_freq  # Calc LO freq to put tone at tone_rf

    # Tune the LO before radio setup so that calibrations run for the correct
    # transmit frequency of interest
    sdr.setFrequency(SOAPY_SDR_TX, chan, lo_freq)
    tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CS16, [chan])
    sdr.activateStream(tx_stream)  # this turns the radio on

    # Transmit
    print('Now Transmitting')
    while True:
        try:
            rc = sdr.writeStream(tx_stream, [tx_buff], buff_len)
            if rc.ret != buff_len:
                print('TX Error {}: {}'.format(rc.ret, errToStr(rc.ret)))
        except KeyboardInterrupt:
            break

    # Stop streaming
    sdr.deactivateStream(tx_stream)
    sdr.closeStream(tx_stream)


def parse_command_line_arguments():
    """ Create command line options for transmit function """
    help_formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description='Transmit a tone from the AIR-T',
                                     formatter_class=help_formatter)
    parser.add_argument('-f', type=float, required=False, dest='freq',
                        default=2400e6, help='TX Tone Frequency')
    parser.add_argument('-c', type=int, required=False, dest='chan',
                        default=0, help='TX Channel Number [0 or 1]')
    parser.add_argument('-s', type=float, required=False, dest='fs',
                        default=31.25e6, help='TX Sample Rate')
    parser.add_argument('-g', type=float, required=False, dest='gain',
                        default=0, help='TX gain')
    parser.add_argument('-n', type=int, required=False, dest='buff_len',
                        default=16384, help='TX Buffer Size')
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    pars = parse_command_line_arguments()
    transmit_tone(pars.freq, pars.chan, pars.fs, pars.gain, pars.buff_len)
