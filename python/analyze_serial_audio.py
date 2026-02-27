#!/usr/bin/env python3
"""
Analyze mono I2S audio from a comma-separated serial log.

Usage:
    python3 analyze_serial_audio.py [logfile] [--plot-seconds N] [--print-vals] [--vals-count N]

Dependencies:
    pip install numpy scipy matplotlib
"""

import re
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
from datetime import datetime

SAMPLE_RATE = 44_100
Q_STEP = 1.7881393433e-7

def load(path):
    with open(path, "r") as f:
        text = f.read()
    values = [int(x) for x in re.findall(r"-?\d+", text)]
    return np.array(values, dtype=np.int64)

def top_frequencies(freqs, magnitudes, n=10):
    peaks, _ = find_peaks(magnitudes, height=np.max(magnitudes) * 0.01)
    if len(peaks) == 0:
        return []
    order = np.argsort(magnitudes[peaks])[::-1]
    return list(zip(freqs[peaks[order[:n]]], magnitudes[peaks[order[:n]]]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile",        nargs="?", default="serial_output.log")
    parser.add_argument("--plot-seconds", type=float, default=0.05)
    parser.add_argument("--print-vals",    action="store_true")
    args = parser.parse_args()

    time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    print(f"Loading: {args.logfile}")
    samples    = load(args.logfile)
    n_neg  = np.sum(samples < 0)
    n_pos  = np.sum(samples > 0)
    n_zero = np.sum(samples == 0)
    print("First 20 samples:", samples[:20]*Q_STEP)
    print(f"Positive: {n_pos:,}  Negative: {n_neg:,}  Zero: {n_zero:,}")
    n_samples  = len(samples)
    duration_s = n_samples / SAMPLE_RATE
    t          = np.arange(n_samples) / SAMPLE_RATE

    print(f"Samples  : {n_samples:,}  ({duration_s:.3f} s @ {SAMPLE_RATE} Hz)")
    print(f"Range    : {samples.min()*Q_STEP:,}  to  {samples.max()*Q_STEP:,}")
    print(f"DC offset: {samples.mean()*Q_STEP:.1f}")

    if n_samples < 256:
        sys.exit("ERROR: Not enough samples.")

    # ── Print values ─────────────────────────────────────────────────────────────
    if args.print_vals:
        print(f"\n{'─'*40}")
        print(f"{'Index':>8}  {'Value':>12}")
        print(f"{'─'*8}  {'─'*12}")
        for i in range(min(args.vals_count, n_samples)):
            print(f"{i:>8}  {samples[i]*Q_STEP:>12}")
        print(f"{'─'*40}\n")

    # ── FFT ───────────────────────────────────────────────────────────────────
    maxval  = 2**31 if samples.max() > 32767 else 2**15
    sf      = samples.astype(np.float64) / maxval

    window    = np.hanning(n_samples)
    fft_mag   = np.abs(rfft(sf * window))
    fft_freqs = rfftfreq(n_samples, d=1.0 / SAMPLE_RATE)
    fft_db    = 20 * np.log10(fft_mag / n_samples + 1e-10)

    # ── Stats ─────────────────────────────────────────────────────────────────
    rms  = np.sqrt(np.mean(sf ** 2)) * Q_STEP
    peak = np.max(np.abs(samples)) * Q_STEP
    print(f"\nRMS={rms:.4f}  peak={peak:,}  DC offset={samples.mean()*Q_STEP:.1f}")

    print("\nTop 10 frequencies:")
    for freq, mag in top_frequencies(fft_freqs, fft_mag):
        print(f"  {freq:8.1f} Hz   magnitude {mag:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    display_n = min(n_samples, int(args.plot_seconds * SAMPLE_RATE))

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(
        f"I2S Audio  |  {SAMPLE_RATE/1000:.1f} kHz mono  |  "
        f"{duration_s:.3f} s  |  {n_samples:,} samples",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.5)

    # Time domain
    ax_t = fig.add_subplot(gs[0])
    ax_t.plot(t[:display_n] * 1000, samples[:display_n]*Q_STEP, color="steelblue", linewidth=0.6)
    ax_t.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_t.set_title(f"Time Domain  (first {args.plot_seconds*1000:.0f} ms)")
    ax_t.set_xlabel("Time (ms)")
    ax_t.set_ylabel("Signal Vin")
    ax_t.grid(True, alpha=0.3)

    # FFT
    ax_f = fig.add_subplot(gs[1])
    ax_f.plot(fft_freqs / 1000, fft_db, color="steelblue", linewidth=0.7)
    ax_f.set_title("FFT Spectrum")
    ax_f.set_xlabel("Frequency (kHz)")
    ax_f.set_ylabel("Magnitude (dB)")
    ax_f.set_xlim(0, SAMPLE_RATE / 2000)
    ax_f.grid(True, alpha=0.3)
    for freq, mag in top_frequencies(fft_freqs, fft_mag, n=5):
        db = 20 * np.log10(mag / n_samples + 1e-10)
        ax_f.annotate(f"{freq:.0f} Hz", xy=(freq/1000, db),
                      xytext=(0, 8), textcoords="offset points",
                      fontsize=7, ha="center", color="steelblue",
                      arrowprops=dict(arrowstyle="-", color="steelblue", lw=0.5))

    plt.savefig(f"results/audio_analysis_{time_stamp}.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → results/audio_analysis_{time_stamp}.png")
    plt.show()

if __name__ == "__main__":
    main()