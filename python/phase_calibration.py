#!/usr/bin/env python3

"""
PhaseCal - RX Phase calibration utility for AIR-T devices.

Estimates and applies phase corrections across multiple receive channels
to enable coherent combining.
"""

import argparse
import sys
import time
import warnings
from itertools import combinations

import numpy as np
import scipy.signal
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_TX, SOAPY_SDR_CF32, SOAPY_SDR_HAS_TIME, errToStr


# Helper functions for various DSP computations
def analyze_channel_pairs(chans: list[int], buffs: list[np.ndarray], samp_rate: float):
    """Compute delay/phase results for every received channel pair.

    Returns a list where each element is a dictionary containing the reference
    channel, the test channel, and the parameters computed by
    estimate_pair_delay_phase().
    """
    pair_results = []
    for idx_ref, idx_test in combinations(range(len(chans)), 2):
        result = estimate_pair_delay_phase(buffs[idx_ref], buffs[idx_test], samp_rate)
        result['chan_ref'] = chans[idx_ref]
        result['chan_test'] = chans[idx_test]
        pair_results.append(result)
    return pair_results

def estimate_pair_delay_phase(sig_ref: np.ndarray, sig_test: np.ndarray, samp_rate: float):
    """Estimate relative delay and phase between two complex signals.

    Takes in the reference samples, test samples, and the sample rate of the
    signal. Returns a dictionary containing the computed delay (in samples and
    seconds) and the computed phase (in degrees).
    """
    max_lag_samples = int(round(100e-9 * samp_rate))
    sig_ref_zm = sig_ref.astype(np.complex128) - np.mean(sig_ref)
    sig_test_zm = sig_test.astype(np.complex128) - np.mean(sig_test)
    corr_full = scipy.signal.correlate(sig_test_zm, sig_ref_zm, method='fft')
    lag_idxs_full = np.arange(-(len(sig_ref) - 1), len(sig_test))
    lag_mask = np.abs(lag_idxs_full) <= max_lag_samples
    corr = corr_full[lag_mask]
    lag_idxs = lag_idxs_full[lag_mask]
    peak_idx = int(np.argmax(np.abs(corr)))
    delay_samples = int(lag_idxs[peak_idx])
    delay_seconds = delay_samples / samp_rate
    if delay_samples >= 0:
        ref_aligned = sig_ref[:len(sig_ref) - delay_samples]
        test_aligned = sig_test[delay_samples:]
    else:
        shift = -delay_samples
        ref_aligned = sig_ref[shift:]
        test_aligned = sig_test[:len(sig_test) - shift]

    n_overlap = min(len(ref_aligned), len(test_aligned))
    ref_aligned = ref_aligned[:n_overlap]
    test_aligned = test_aligned[:n_overlap]
    if n_overlap == 0:
        phase_deg = np.nan
    else:
        phase_est = np.vdot(ref_aligned, test_aligned)
        phase_deg = float(np.degrees(np.angle(phase_est)))

    return {
        'delay_samples': delay_samples,
        'delay_seconds': delay_seconds,
        'phase_deg': phase_deg,
    }


def compute_channel_phase_rotations(chans: list[int], pair_results: list[dict]):
    """Solve for one complex phase rotation per channel from pairwise
    measurements.

    Takes in the data structure produced by analyze_channel_pairs() and returns
    the computed phase rotations as a dictionary where the key is the
    corresponding reference channel.
    """
    ref_chan = chans[0]
    chan_to_col = {chan: idx for idx, chan in enumerate(chans[1:])}
    num_unknowns = len(chans) - 1

    if num_unknowns == 0:
        return {ref_chan: np.complex128(1.0 + 0.0j)}

    rows = []
    rhs = []
    for result in pair_results:
        phase_deg = result["phase_deg"]
        if not np.isfinite(phase_deg):
            continue

        m_ij = np.exp(1j * np.radians(phase_deg))
        row = np.zeros(num_unknowns, dtype=np.complex128)
        b = 0.0j
        test_chan = result["chan_test"]
        ref_pair_chan = result["chan_ref"]

        if test_chan == ref_chan:
            # 1 - m_ij * r_j ~= 0
            row[chan_to_col[ref_pair_chan]] = -m_ij
            b = -1.0
        elif ref_pair_chan == ref_chan:
            # r_i - m_ij * 1 ~= 0
            row[chan_to_col[test_chan]] = 1.0
            b = m_ij
        else:
            # r_i - m_ij * r_j ~= 0
            row[chan_to_col[test_chan]] = 1.0
            row[chan_to_col[ref_pair_chan]] = -m_ij

        rows.append(row)
        rhs.append(b)

    if not rows:
        return {chan: np.complex128(1.0 + 0.0j) for chan in chans}

    A = np.vstack(rows)
    b = np.asarray(rhs, dtype=np.complex128)
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    channel_rotations = {ref_chan: np.complex128(1.0 + 0.0j)}

    for chan, r in zip(chans[1:], sol):
        # Project back onto the unit circle
        mag = np.abs(r)
        channel_rotations[chan] = (
            np.conj(np.complex128(r / mag)) if mag > 0 else np.complex128(1.0 + 0.0j)
        )
    return channel_rotations


class PhaseCal:
    def __init__(self, device: SoapySDR.Device, tx_cal_src: int,
                 rx_cal_chans: list[int], master_clk_rate: float, freq: float):
        self._tx_stream = None
        self.rx_stream = None
        self.path_results = {"internal": [], "external": []}
        global_settings = device.getSettingInfo()
        info = next((x for x in global_settings if x.key == "phase_cal:mode"), None)
        if info is None:
            raise ValueError("Device does not support phase calibration!")
        self._device = device
        if self._device.getMasterClockRate() != master_clk_rate:
            # Note that it is much more efficient to set the master clock rate when
            # creating a new SDR device, so hopefully this warning is never printed.
            warnings.warn("SDR master clock rate not set previously, applying now...")
            self._device.setMasterClockRate(master_clk_rate)

        # Setup Rx
        if not isinstance(rx_cal_chans, list):
            rx_cal_chans = list(rx_cal_chans)
        self._rx_cal_chans = rx_cal_chans
        for chan in self._rx_cal_chans:
            self._device.setSampleRate(SOAPY_SDR_RX, chan, master_clk_rate / 2)
            self._device.setFrequency(SOAPY_SDR_RX, chan, freq)
            self._device.setGainMode(SOAPY_SDR_RX, chan, False)  # AGC off
        self.rx_stream = self._device.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, self._rx_cal_chans)
        print(f'\nReceiving on Channel(s): {self._rx_cal_chans}\n')

        # Setup Tx
        tx_fs = master_clk_rate
        bb_freq = 0
        tx_lo_freq = freq - bb_freq
        self._tx_cal_src = tx_cal_src
        self._device.setSampleRate(SOAPY_SDR_TX, self._tx_cal_src, tx_fs)
        self._device.setFrequency(SOAPY_SDR_TX, self._tx_cal_src, tx_lo_freq)
        self._tx_stream = self._device.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [self._tx_cal_src])

        # Activate noise generator on FPGA for calibration signal.
        self._device.activateStream(self._tx_stream)
        self._device.writeSetting("dds:frequency", bb_freq)
        print('Enabling noise generator for calibration...')
        self._device.writeSetting("dds:mode", "Noise")
        print(f'\nTransmitting on Channel: {self._tx_cal_src}\n')

    def __del__(self):
        self._device.writeSetting("dds:mode", "Off")

        if self._tx_stream is not None:
            self._device.closeStream(self._tx_stream)
            self._tx_stream = None

        if self.rx_stream is not None:
            self._device.closeStream(self.rx_stream)
            self.rx_stream = None

    def run(self, cal_type: str = "internal", num_captures: int = 5,
            delay: float = 1.0, buff_len: int = 4096, ext_gain: float = -15.0):
        """Configures the device for calibration and runs the calibration."""
        cal_type = cal_type.lower()
        if cal_type not in ('internal', 'external', 'both'):
            raise ValueError(f"Unsupported calibration type: {cal_type}")

        if cal_type in ('internal', 'both'):
            print("Performing Internal Calibration...")
            if self._rx_cal_chans == [0, 1] and self._tx_cal_src in self._rx_cal_chans:
                phase_cal_mode = 'Slot A'
            elif self._rx_cal_chans == [2, 3] and self._tx_cal_src in self._rx_cal_chans:
                phase_cal_mode = 'Slot B'
            else:
                phase_cal_mode = 'Four Channel'

            if cal_type == 'both':
                self._device.writeSetting('phase_cal:mode', 'Off')

            self._device.setGain(SOAPY_SDR_TX, self._tx_cal_src, 0.0)
            for chan in self._rx_cal_chans:
                self._device.setGain(SOAPY_SDR_RX, chan, 0.0)
            self._device.writeSetting('phase_cal:mode', phase_cal_mode)
            self._device.writeSetting('phase_cal:src_chan', self._tx_cal_src)
            phasors_internal = self._run_cal_path("Internal",
                               num_captures=num_captures,
                               delay=delay,
                               buff_len=buff_len)

        if cal_type in ('external', 'both'):
            print("Performing External Calibration...")
            self._device.writeSetting('phase_cal:mode', "Off")
            self._device.setGain(SOAPY_SDR_TX, self._tx_cal_src, -15.0)
            for chan in self._rx_cal_chans:
                self._device.setGain(SOAPY_SDR_RX, chan, -15.0)
            phasors_external = self._run_cal_path("External",
                               num_captures=num_captures,
                               delay=delay,
                               buff_len=buff_len,
                               ext_gain=ext_gain)

    def _run_cal_path(self, path_name: str = "Internal", num_captures: int = 5,
                      delay: float = 1.0, buff_len: int = 4096, ext_gain: float = -15.0):
        """Runs the requested calibration path and applies the phasor settings.

        Returns the results of each capture as a list, where each index contains
        applied phasor settings (i.e., the data structure returned from
        apply_channel_phase_rotations()) from a specific internal or external run.
        """
        samp_rate = self._device.getSampleRate(SOAPY_SDR_RX, self._rx_cal_chans[0])
        start_time_ns = int(time.time() * 1e9)
        self._device.setHardwareTime(start_time_ns, "now")
        repeat_time_ns = int(delay * 1e9)
        rx_time_ns = start_time_ns + int(1e9)

        for capture in range(num_captures):
            rx_buff = [np.zeros(buff_len, np.complex64) for _ in self._rx_cal_chans]
            self._device.activateStream(self.rx_stream,
                                        flags=SOAPY_SDR_HAS_TIME,
                                        timeNs=rx_time_ns)
            now = self._device.getHardwareTime("now")
            if now > rx_time_ns:
                print(f"capture: {capture} is late, skipping")
                print(f"now - rx_time_ns = {now - rx_time_ns} ns")
                self._device.deactivateStream(self.rx_stream)
                rx_time_ns += repeat_time_ns
                continue

            rc = self._device.readStream(self.rx_stream, rx_buff, buff_len,
                                         timeoutUs=int((rx_time_ns - now) / 1000) + 100000)
            if rc.ret != buff_len:
                print('capture: {}, RX Error {}: {}'.format(capture, rc.ret,
                                                            errToStr(rc.ret)))
                self._device.deactivateStream(self.rx_stream)
                break

            self._device.deactivateStream(self.rx_stream)
            pair_results = analyze_channel_pairs(self._rx_cal_chans, rx_buff, samp_rate)
            if capture == num_captures - 1:
                print('Results:')
                for result in pair_results:
                    print(
                        f"  CH{result['chan_test']} vs CH{result['chan_ref']} "
                        f"delay={result['delay_samples']} samples "
                        f"({result['delay_seconds'] * 1e9:.1f} ns), "
                        f"phase={result['phase_deg']:.2f} deg, "
                    )
                print()

            phase_rotations = compute_channel_phase_rotations(self._rx_cal_chans, pair_results)
            phasor_results = self._apply_channel_phase_rotations(phase_rotations)
            self.path_results[path_name.lower()].append(phasor_results)
            rx_time_ns += repeat_time_ns
        return self.path_results[path_name.lower()]

    def _apply_channel_phase_rotations(self, phase_rotations: dict[np.complex128]):
        """Accumulate and program per-channel RX calibration phasor settings.

        Takes in the computed per-channel phase rotations and returns the
        applied phasors as a per-channel dictionary. See below for contents of each
        index.
        """
        applied_phasors = {}
        for chan in self._rx_cal_chans:
            delta_phasor = phase_rotations.get(chan, np.complex128(1.0 + 0.0j))
            existing_real = float(self._device.readSetting(SOAPY_SDR_RX, chan, "cal_phasor_real"))
            existing_imag = float(self._device.readSetting(SOAPY_SDR_RX, chan, "cal_phasor_imag"))
            existing_phasor = np.complex128(existing_real + 1j * existing_imag)
            phasor = existing_phasor * delta_phasor

            phasor_mag = np.abs(phasor)
            if phasor_mag > 0:
                phasor /= phasor_mag
            else:
                phasor = np.complex128(1.0 + 0.0j)

            self._device.writeSetting(SOAPY_SDR_RX, chan, "cal_phasor_real", phasor.real)
            self._device.writeSetting(SOAPY_SDR_RX, chan, "cal_phasor_imag", phasor.imag)

            applied_phasors[chan] = {
                'delta_phasor': delta_phasor,
                'phasor': phasor,
                'delta_phase_deg': float(np.degrees(np.angle(delta_phasor))),
                'phase_deg': float(np.degrees(np.angle(phasor))),
            }
        return applied_phasors


class DefaultsRawTextHelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                                   argparse.RawTextHelpFormatter):
    """Preserve help-text newlines while showing argument default values."""


def parse_command_line_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Run RX Phase Calibration on an AIR-T Device.",
                                     formatter_class=DefaultsRawTextHelpFormatter)
    argument_definitions = (
        ('--freq', dict(type=float, default=1400e6, help='RX carrier frequency in Hz.')),
        ('--master-clock', dict(type=float, default=125e6, help='Master clock rate in Hz.')),
        ('--captures', dict(type=int, default=5, help='number of calibration captures.')),
        ('--delay', dict(type=float, default=1.0, help='delay between captures in seconds.')),
        ('--samples', dict(type=int, default=4096, help='samples per capture.')),
        ('--channels', dict(type=int, action='store', default=(0, 1, 2, 3),
                            nargs='+', help='space-separated RX channels.')),
        ('--tx-channel', dict(type=int, default=0,
                              help='TX channel used as the calibration source.')),
        ('--calibration', dict(choices=('internal', 'external', 'both'),
                               default='internal', help='calibration path to run.')),
        ('--ext-gain', dict(type=float, default=-15.0,
                             help=('external calibration gain in dB.\n'
                                   'NOTE: only relevant when --calibration is external or both.\n'
                                   'Ignored for internal calibration.'))),
    )
    for opt, kwargs in argument_definitions:
        parser.add_argument(opt, **kwargs)
    args = parser.parse_args(argv)
    command_line = argv if argv is not None else sys.argv[1:]
    ext_gain_specified = any(
        argument == '--ext-gain' or argument.startswith('--ext-gain=')
        for argument in command_line
    )
    if args.calibration == 'internal' and ext_gain_specified:
        warnings.warn('--ext-gain is ignored when --calibration is internal.',
                      UserWarning, stacklevel=2)
    return args

def print_configuration(args: argparse.ArgumentParser):
    print("\nRadio Configuration:\n")
    for label, value in (
            ('RX Channels', args.channels),
            ('TX Calibration Source', f'CH{args.tx_channel}'),
            ('Frequency', f'{args.freq / 1e6:.3f} MHz'),
            ('Master Clock Rate', f'{args.master_clock / 1e6:.6f} MHz'),
            ('Captures', args.captures),
            ('Delay', f'{args.delay:g} s'),
            ('Samples per Capture', args.samples),
            ('Calibration Path(s)', args.calibration)
    ):
        print(f'  {label}: {value}')
    print()

def main():
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_command_line_arguments()
    print_configuration(args)

    dev_args = dict(master_clock_rate=str(args.master_clock),
                    time_src="internal",
                    lock_timeout=str(10))
    sdr = SoapySDR.Device(dev_args)
    cal = PhaseCal(sdr, args.tx_channel, args.channels, args.master_clock, args.freq)
    cal.run(args.calibration, args.captures, args.delay, args.samples, args.ext_gain)

    # Note that after calibration is run, cal.rx_stream can be used to read
    # samples from the calibrated channels. The calibration will be valid so
    # long as the cal object is in scope and not garbage collected.

if __name__ == '__main__':
    main()
