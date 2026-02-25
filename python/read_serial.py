#!/usr/bin/env python3
"""
Read from a serial port on macOS at 115200 baud and write output to a file.

Usage:
    python3 read_serial.py [port] [output_file]

Example:
    python3 read_serial.py /dev/tty.usbserial-0001 output.log

Dependencies:
    pip install pyserial
"""

import sys
import signal
import serial
from datetime import datetime

# --- Configuration ---
PORT        = sys.argv[1] if len(sys.argv) > 1 else "/dev/tty.usbserial-0001"
OUTPUT_FILE = sys.argv[2] if len(sys.argv) > 2 else "serial_output.log"
BAUD_RATE   = 115200
TIMEOUT     = 1  # seconds


def find_serial_ports():
    """List available serial ports to help the user pick one."""
    import glob
    ports = glob.glob("/dev/tty.*") + glob.glob("/dev/cu.*")
    return sorted(ports)


def main():
    print(f"Opening serial port : {PORT}")
    print(f"Baud rate           : {BAUD_RATE}")
    print(f"Writing output to   : {OUTPUT_FILE}")
    print("Press Ctrl+C to stop.\n")

    try:
        ser = serial.Serial(
            port=PORT,
            baudrate=BAUD_RATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=TIMEOUT,
        )
    except serial.SerialException as e:
        print(f"\nERROR: Could not open port '{PORT}': {e}")
        print("\nAvailable serial ports on this machine:")
        for p in find_serial_ports():
            print(f"  {p}")
        sys.exit(1)

    # Allow clean exit on Ctrl+C
    running = True
    def _stop(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _stop)

    bytes_written = 0
    with open(OUTPUT_FILE, "ab") as f:
        # Write a session header
        header = f"\n--- Session started {datetime.now().isoformat()} ---\n"
        f.write(header.encode())
        print(header.strip())

        while running:
            try:
                data = ser.read(1024)          # read up to 1 KB at a time
                if data:
                    f.write(data)
                    f.flush()                  # keep the file up to date
                    bytes_written += len(data)
                    sys.stdout.buffer.write(data)   # echo to terminal too
                    sys.stdout.flush()
            except serial.SerialException as e:
                print(f"\nSerial error: {e}")
                break

    ser.close()
    footer = f"\n--- Session ended {datetime.now().isoformat()} | {bytes_written} bytes written ---\n"
    with open(OUTPUT_FILE, "a") as f:
        f.write(footer)
    print(footer.strip())


if __name__ == "__main__":
    main()
