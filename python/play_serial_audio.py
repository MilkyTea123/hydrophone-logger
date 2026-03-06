#!/usr/bin/env python3

import re
import sys
import argparse
import numpy as np
from scipy.io.wavfile import write
import sounddevice as sd

SAMPLE_RATE = 44_100

def load(path):
    with open(path, "r") as f:
        text = f.read()
    values = [int(x) for x in re.findall(r"-?\d+", text)]
    return np.array(values, dtype=np.int64)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile",        nargs="?", default="serial_output.log")
    args = parser.parse_args()

    print(f"Loading: {args.logfile}")
    samples    = load(args.logfile)
    samples    = (np.iinfo(np.int16).max / np.iinfo(np.int32).max) * samples
    n_samples  = len(samples)
    duration_s = n_samples / SAMPLE_RATE

    print(f"Samples  : {n_samples:,}  ({duration_s:.3f} s @ {SAMPLE_RATE} Hz)")

    write(args.logfile.split(".")[0] + ".wav", SAMPLE_RATE, samples)

    print('playing sound')
    sd.play(samples, SAMPLE_RATE)
    sd.wait()

    if n_samples < 256:
        sys.exit("ERROR: Not enough samples.")



if __name__ == "__main__":
    main()