"""Quest screen recorder via adb screenrecord.

Usage:
    python record.py          # Start recording, press Enter to stop
    python record.py --time 30  # Record for 30 seconds then stop
"""
import subprocess
import sys
import os
import time
import argparse
from pathlib import Path

REMOTE_PATH = "/sdcard/quest_recording.mp4"


def main():
    parser = argparse.ArgumentParser(description="Record Quest screen via adb")
    parser.add_argument("--time", type=int, default=0, help="Max seconds (0=manual stop)")
    parser.add_argument("--output", "-o", type=str, default="", help="Output filename")
    args = parser.parse_args()

    out_dir = Path(__file__).parent / "recordings"
    out_dir.mkdir(exist_ok=True)

    if args.output:
        out_file = out_dir / args.output
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_file = out_dir / f"quest_{stamp}.mp4"

    # Check adb connection
    result = subprocess.run(["adb", "devices"], capture_output=True, text=True)
    if "device" not in result.stdout.split("\n", 1)[-1]:
        print("ERROR: No Quest device found. Check USB/WiFi adb connection.")
        sys.exit(1)

    # Build screenrecord command
    cmd = ["adb", "shell", "screenrecord", REMOTE_PATH]
    if args.time > 0:
        cmd.extend(["--time-limit", str(min(args.time, 180))])

    print(f"Recording Quest screen...")
    print(f"Output: {out_file}")
    if args.time > 0:
        print(f"Will stop after {args.time}s")
    else:
        print("Press Enter to stop recording.")

    # Start recording
    proc = subprocess.Popen(cmd)

    try:
        if args.time > 0:
            proc.wait(timeout=args.time + 5)
        else:
            input()  # Wait for Enter
            proc.terminate()
            proc.wait(timeout=5)
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait(timeout=5)

    # Give device a moment to finalize the file
    time.sleep(1)

    # Pull file from device
    print("Pulling recording from Quest...")
    pull = subprocess.run(["adb", "pull", REMOTE_PATH, str(out_file)], capture_output=True, text=True)
    if pull.returncode == 0:
        size_mb = out_file.stat().st_size / (1024 * 1024)
        print(f"Saved: {out_file} ({size_mb:.1f} MB)")
    else:
        print(f"ERROR pulling file: {pull.stderr}")
        sys.exit(1)

    # Clean up remote file
    subprocess.run(["adb", "shell", "rm", REMOTE_PATH], capture_output=True)


if __name__ == "__main__":
    main()
