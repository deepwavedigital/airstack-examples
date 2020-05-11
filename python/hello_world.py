#!/usr/bin/env python3
#
# Copyright 2020, Deepwave Digital, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from matplotlib import pyplot as plt
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CS16, errToStr

############################################################################################
# Settings
############################################################################################
# Data transfer settings
rx_chan = 0             # RX1 = 0, RX2 = 1
N = 16384               # Number of complex samples per transfer
fs = 31.25e6              # Radio sample Rate
freq = 2.4e9            # LO tuning frequency in Hz
use_agc = True          # Use or don't use the AGC
timeout_us = int(5e6)
rx_bits = 16            # The AIR-T's ADC is 16 bits

############################################################################################
# Receive Signal
############################################################################################
#  Initialize the AIR-T receiver using SoapyAIRT
sdr = SoapySDR.Device(dict(driver="SoapyAIRT")) # Create AIR-T instance
sdr.setSampleRate(SOAPY_SDR_RX, 0, fs)          # Set sample rate
sdr.setGainMode(SOAPY_SDR_RX, 0, use_agc)       # Set the gain mode
sdr.setFrequency(SOAPY_SDR_RX, 0, freq)         # Tune the LO

# Create data buffer and start streaming samples to it
rx_buff = np.empty(2 * N, np.int16)                 # Create memory buffer for data stream
rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16, [rx_chan])  # Setup data stream
sdr.activateStream(rx_stream)  # this turns the radio on

# Read the samples from the data buffer
sr = sdr.readStream(rx_stream, [rx_buff], N, timeoutUs=timeout_us)
rc = sr.ret  # number of samples read or the error code
assert rc == N, 'Error {}: {}'.format(rc.ret, errToStr(rc.ret))

# Stop streaming
sdr.deactivateStream(rx_stream)
sdr.closeStream(rx_stream)

############################################################################################
# Plot Signal
############################################################################################
# Convert interleaved shorts (received signal) to numpy.complex64 normalized between [-1, 1]
s0 = rx_buff.astype(float) / np.power(2.0, rx_bits-1)
s = (s0[::2] + 1j*s0[1::2])

# Take the fourier transform of the signal and perform FFT Shift
S = np.fft.fftshift(np.fft.fft(s, N) / N)

# Time Domain Plot
plt.figure(num=1, figsize=(11, 8.5))
plt.subplot(211)
t_us = np.arange(N) / fs / 1e-6
plt.plot(t_us, s.real, 'k', label='I')
plt.plot(t_us, s.imag, 'r', label='Q')
plt.xlim(t_us[0], t_us[-1])
plt.xlabel('Time (us)')
plt.ylabel('Normalized Amplitude')

# Frequency Domain Plot
plt.subplot(212)
f_ghz = (freq + (np.arange(0, fs, fs/N) - (fs/2) + (fs/N))) / 1e9
plt.plot(f_ghz, 20*np.log10(np.abs(S)))
plt.xlim(f_ghz[0], f_ghz[-1])
plt.ylim(-140, 0)
plt.xlabel('Frequency (GHz)')
plt.ylabel('Amplitude (dBFS)')
plt.show()
