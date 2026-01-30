#!/usr/bin/env python3
"""
Latency measurement tool for Quest pipeline.
Measures delay at each stage: capture, decode, display.
"""
import time
import cv2
import numpy as np

def main():
    print("Quest Pipeline Latency Measurement")
    print("=" * 50)

    # Import scrcpy components
    try:
        from adbutils import adb
        from myscrcpy.core import Session, VideoArgs
    except ImportError:
        print("ERROR: Install dependencies: pip install adbutils myscrcpy")
        return

    # Connect to device
    devices = adb.device_list()
    if not devices:
        print("ERROR: No ADB devices found")
        return

    device = devices[0]
    print(f"Device: {device.serial}")

    # Start session with minimal buffering
    print("Starting scrcpy session...")
    session = Session(
        device,
        video_args=VideoArgs(max_size=0, fps=60),
    )
    time.sleep(2)

    if session.va is None:
        print("ERROR: Video adapter not initialized")
        return

    print("Connected! Measuring latency...\n")

    # Timing stats
    capture_times = []
    total_times = []
    frame_count = 0
    last_frame = None

    cv2.namedWindow('Latency Test', cv2.WINDOW_NORMAL)

    print("Move your head - watch for delay between movement and display update")
    print("Press 'q' to quit\n")

    try:
        while True:
            t_start = time.perf_counter()

            # Capture (blocking call to scrcpy)
            frame = session.va.get_frame()
            t_capture = time.perf_counter()

            if frame is None:
                continue

            # Skip if same frame
            if frame is last_frame:
                time.sleep(0.001)
                continue
            last_frame = frame

            # Convert to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            t_convert = time.perf_counter()

            # Display
            cv2.imshow('Latency Test', frame_bgr)
            t_display = time.perf_counter()

            # Calculate times
            capture_ms = (t_capture - t_start) * 1000
            convert_ms = (t_convert - t_capture) * 1000
            display_ms = (t_display - t_convert) * 1000
            total_ms = (t_display - t_start) * 1000

            capture_times.append(capture_ms)
            total_times.append(total_ms)
            frame_count += 1

            # Print stats every 30 frames
            if frame_count % 30 == 0:
                avg_capture = np.mean(capture_times[-30:])
                avg_total = np.mean(total_times[-30:])
                fps = 30 / (sum(total_times[-30:]) / 1000) if total_times else 0

                print(f"\rCapture: {avg_capture:5.1f}ms | Convert: {convert_ms:4.1f}ms | "
                      f"Display: {display_ms:4.1f}ms | Total: {avg_total:5.1f}ms | "
                      f"FPS: {fps:5.1f}", end='', flush=True)

            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        session.disconnect()

        # Print summary
        print("\n\n" + "=" * 50)
        print("LATENCY SUMMARY")
        print("=" * 50)
        if capture_times:
            print(f"Capture (scrcpy): {np.mean(capture_times):6.1f}ms avg, {np.max(capture_times):6.1f}ms max")
            print(f"Total pipeline:   {np.mean(total_times):6.1f}ms avg, {np.max(total_times):6.1f}ms max")
            print(f"Frames: {frame_count}")

            # Identify bottleneck
            avg_capture = np.mean(capture_times)
            if avg_capture > 50:
                print("\n>>> BOTTLENECK: scrcpy capture is slow")
                print("    Try: Lower resolution, check USB cable/connection")
            elif avg_capture > 30:
                print("\n>>> Capture is moderate - some delay expected")
            else:
                print("\n>>> Capture is fast - delay may be in display or perception")


if __name__ == "__main__":
    main()
