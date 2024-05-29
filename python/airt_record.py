#!/usr/bin/env python3
#
# Copyright 2020, Deepwave Digital, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# Import Packages
import numpy as np
import os
from matplotlib import pyplot as plt
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CS16, errToStr

########################################################################################
# Settings
########################################################################################
# Data transfer settings
rx_chan = 0               # RX1 = 0, RX2 = 1
N = 16384                  # Number of complex samples per transfer
fs = 31.25e6               # Radio sample Rate
freq = 2.4e9               # LO tuning frequency in Hz
use_agc = True           # Use or don't use the AGC
timeout_us = int(5e6)

# Recording Settings
cplx_samples_per_file = 2048  # Complex samples per file
nfiles = 6              # Number of files to record
rec_dir = os.path.join(os.getenv('HOME'), 'Desktop')  # Location of drive for recording
file_prefix = 'file'   # File prefix for each file

########################################################################################
# Receive Signal
########################################################################################
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
rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16, [rx_chan]) # Setup data stream
sdr.activateStream(rx_stream)  # this turns the radio on

file_names = []  # Save the file names for plotting later. Remove if not plotting.
file_ctr = 0
while file_ctr < nfiles:
    # Read the samples from the data buffer
    sr = sdr.readStream(rx_stream, [rx_buff], N, timeoutUs=timeout_us)

    # Make sure that the proper number of samples was read
    rc = sr.ret
    assert rc == N, 'Error {}: {}'.format(rc, errToStr(rc))     # I am getting OverFlow Error due to this line and can't able to rectify it. Can anyone send the modified Code.

    # Write buffer to multiple files. Reshaping the rx_buffer allows for iteration
    for file_data in rx_buff.reshape(files_per_buffer, real_samples_per_file):
        # Define current file name
        file_name = os.path.join(rec_dir, '{}_{}.bin'.format(file_prefix, file_ctr))

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

########################################################################################
# Plot Recorded Data
########################################################################################

nrow = 2
ncol = np.ceil(float(nfiles) / float(nrow)).astype(int)
fig, axs = plt.subplots(nrow, ncol, figsize=(11, 11), sharex='all', sharey='all')
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
