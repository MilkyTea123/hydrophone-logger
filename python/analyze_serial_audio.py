#!/usr/bin/env python3
"""
Analyze mono I2S audio from a comma-separated serial log.

Usage:
    python3 analyze_serial_audio.py [logfile] [--plot-seconds N] [--print-raw] [--raw-count N]

Dependencies:
    pip install numpy scipy matplotlib
"""

import re
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import welch, find_peaks
from scipy.fft import rfft, rfftfreq

SAMPLE_RATE = 44_100

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
    parser.add_argument("--print-raw",    action="store_true")
    parser.add_argument("--raw-count",    type=int, default=100)
    args = parser.parse_args()

    print(f"Loading: {args.logfile}")
    samples    = load(args.logfile)
    n_neg  = np.sum(samples < 0)
    n_pos  = np.sum(samples > 0)
    n_zero = np.sum(samples == 0)
    print("First 20 samples:", samples[:20])
    print(f"Positive: {n_pos:,}  Negative: {n_neg:,}  Zero: {n_zero:,}")
    n_samples  = len(samples)
    duration_s = n_samples / SAMPLE_RATE
    t          = np.arange(n_samples) / SAMPLE_RATE

    print(f"Samples  : {n_samples:,}  ({duration_s:.3f} s @ {SAMPLE_RATE} Hz)")
    print(f"Range    : {samples.min():,}  to  {samples.max():,}")
    print(f"DC offset: {samples.mean():.1f}")

    if n_samples < 256:
        sys.exit("ERROR: Not enough samples.")

    # ── Print raw ─────────────────────────────────────────────────────────────
    if args.print_raw:
        print(f"\n{'─'*40}")
        print(f"{'Index':>8}  {'Value':>12}")
        print(f"{'─'*8}  {'─'*12}")
        for i in range(min(args.raw_count, n_samples)):
            print(f"{i:>8}  {samples[i]:>12}")
        print(f"{'─'*40}\n")

    # ── FFT ───────────────────────────────────────────────────────────────────
    maxval  = 2**31 if samples.max() > 32767 else 2**15
    sf      = samples.astype(np.float64) / maxval

    window    = np.hanning(n_samples)
    fft_mag   = np.abs(rfft(sf * window))
    fft_freqs = rfftfreq(n_samples, d=1.0 / SAMPLE_RATE)
    fft_db    = 20 * np.log10(fft_mag / n_samples + 1e-10)

    f_w, psd = welch(sf, fs=SAMPLE_RATE, nperseg=min(4096, n_samples // 4))

    # ── Stats ─────────────────────────────────────────────────────────────────
    rms  = np.sqrt(np.mean(sf ** 2))
    peak = np.max(np.abs(samples))
    print(f"\nRMS={rms:.4f}  peak={peak:,}  DC offset={samples.mean():.1f}")

    print("\nTop 10 frequencies:")
    for freq, mag in top_frequencies(fft_freqs, fft_mag):
        print(f"  {freq:8.1f} Hz   magnitude {mag:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    display_n = min(n_samples, int(args.plot_seconds * SAMPLE_RATE))

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"I2S Audio  |  {SAMPLE_RATE/1000:.1f} kHz mono  |  "
        f"{duration_s:.3f} s  |  {n_samples:,} samples",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.5)

    # Time domain
    ax_t = fig.add_subplot(gs[0])
    ax_t.plot(t[:display_n] * 1000, samples[:display_n], color="steelblue", linewidth=0.6)
    ax_t.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_t.set_title(f"Time Domain  (raw ADC counts, first {args.plot_seconds*1000:.0f} ms)")
    ax_t.set_xlabel("Time (ms)")
    ax_t.set_ylabel("ADC value")
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

    # Welch PSD
    ax_p = fig.add_subplot(gs[2])
    ax_p.semilogy(f_w / 1000, psd, color="steelblue", linewidth=1)
    ax_p.set_title("Welch Power Spectral Density")
    ax_p.set_xlabel("Frequency (kHz)")
    ax_p.set_ylabel("PSD")
    ax_p.set_xlim(0, SAMPLE_RATE / 2000)
    ax_p.grid(True, which="both", alpha=0.3)

    plt.savefig("audio_analysis.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → audio_analysis.png")
    plt.show()

if __name__ == "__main__":
    main()