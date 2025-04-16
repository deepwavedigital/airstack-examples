#!/usr/bin/env python3
#
# Copyright 2020, Deepwave Digital, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# Import Packages
import sys
import argparse
import numpy as np
import os
from pathlib import Path
from matplotlib import pyplot as plt
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CS16, errToStr


_script_dir = Path(__file__).parent
results_dir = _script_dir / 'results'


def record(rec_dir, freq, rx_chan, fs, use_agc, N):
    # Data transfer settings
    timeout_us = int(5e6)

    # Recording Settings
    cplx_samples_per_file = 2048  # Complex samples per file
    nfiles = 6              # Number of files to record
    file_prefix = 'file'   # File prefix for each file

    # Receive Signal
    # File calculations and checks
    assert N % cplx_samples_per_file == 0, 'samples_per_file must be divisible by N'
    files_per_buffer = int(N / cplx_samples_per_file)
    real_samples_per_file = 2 * cplx_samples_per_file

    #  Initialize the AIR-T receiver using SoapyAIRT
    sdr = SoapySDR.Device(dict(driver="SoapyAIRT"))  # Create AIR-T instance
    sdr.setSampleRate(SOAPY_SDR_RX, 0, fs)           # Set sample rate
    sdr.setGainMode(SOAPY_SDR_RX, 0, use_agc)        # Set the gain mode
    sdr.setFrequency(SOAPY_SDR_RX, 0, freq)          # Tune the LO

    # Create data buffer and start streaming samples to it
    rx_buff = np.empty(2 * N, np.int16)  # Create memory buffer for data stream
    rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16, [
                                rx_chan])  # Setup data stream
    sdr.activateStream(rx_stream)  # this turns the radio on

    # Save the file names for plotting later. Remove if not plotting.
    file_names = []
    file_ctr = 0
    while file_ctr < nfiles:
        # Read the samples from the data buffer
        sr = sdr.readStream(rx_stream, [rx_buff], N, timeoutUs=timeout_us)

        # Make sure that the proper number of samples was read
        rc = sr.ret
        assert rc == N, 'Error {}: {}'.format(rc, errToStr(rc))

        # Write buffer to multiple files. Reshaping the rx_buffer allows for iteration
        for file_data in rx_buff.reshape(files_per_buffer, real_samples_per_file):
            # Define current file name
            file_name = os.path.join(
                rec_dir, '{}_{}.bin'.format(file_prefix, file_ctr))

            # Write signal to disk
            file_data.tofile(file_name)

            # Save file name for plotting later. Remove this if you are not going to plot.
            file_names.append(file_name)

            # Increment file write counter and see if we are done
            file_ctr += 1
            if file_ctr > nfiles:
                break

    # Stop streaming and close the connection to the radio
    sdr.deactivateStream(rx_stream)
    sdr.closeStream(rx_stream)

    # Plot Recorded Data
    nrow = 2
    ncol = np.ceil(float(nfiles) / float(nrow)).astype(int)
    fig, axs = plt.subplots(nrow, ncol, figsize=(
        11, 11), sharex='all', sharey='all')
    for ax, file_name in zip(axs.flatten(), file_names):
        # Read data from current file
        s_interleaved = np.fromfile(file_name, dtype=np.int16)

        # Convert interleaved shorts (received signal) to numpy.float32
        s_real = s_interleaved[::2].astype(np.float32)
        s_imag = s_interleaved[1::2].astype(np.float32)

        # Plot time domain signals
        ax.plot(s_real, 'k', label='I')
        ax.plot(s_imag, 'r', label='Q')
        ax.set_xlim([0, len(s_real)])
        ax.set_title(os.path.basename(file_name))

    plt.show()


def parse_command_line_arguments():
    """ Create command line options for transmit function """
    help_formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description='Record Signals using an AIR-T',
                                     formatter_class=help_formatter)
    parser.add_argument('-d', type=str, required=False, dest='results_dir',
                        default=None, help='Output directory for files')
    parser.add_argument('-f', type=float, required=False, dest='freq',
                        default=2400e6, help='Rx Frequency')
    parser.add_argument('-c', type=int, required=False, dest='chan',
                        default=0, help='Rx Channel Number [0 or 1]')
    parser.add_argument('-s', type=float, required=False, dest='fs',
                        default=31.25e6, help='Rx Sample Rate')
    parser.add_argument('-g', type=bool, required=False, dest='use_agc',
                        default=True, help='Use AGC')
    parser.add_argument('-n', type=int, required=False, dest='buff_len',
                        default=16384, help='Rx Buffer Size')
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    pars = parse_command_line_arguments()
    if pars.results_dir is None:
        pars.results_dir = results_dir
    os.makedirs(pars.results_dir, exist_ok=True)
    record(pars.results_dir, pars.freq, pars.chan,
           pars.fs, pars.use_agc, pars.buff_len)
