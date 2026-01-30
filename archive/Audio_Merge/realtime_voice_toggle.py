import sounddevice as sd
import torch
import numpy as np
import queue
import threading
import sys
import select
import tty
import termios
from df.enhance import enhance, init_df

# Initialize model
print("Loading model...")
model, df_state, _ = init_df()
model.eval()
print("Model loaded!")

# DeepFilterNet expects 48kHz, so we'll use that and let sounddevice handle resampling if needed
fs = 48000
blocksize = int(fs * 0.1)  # 100 ms - larger blocksize reduces callback frequency

# Get device info
try:
    input_device = sd.query_devices(kind="input")
    output_device = sd.query_devices(kind="output")
    print(
        f"Input device: {input_device['name']} (max SR: {input_device.get('default_samplerate', 'unknown')} Hz)"
    )
    print(
        f"Output device: {output_device['name']} (max SR: {output_device.get('default_samplerate', 'unknown')} Hz)"
    )
    print(f"Using sample rate: {fs} Hz (DeepFilterNet requirement)\n")
except:
    print(f"Using sample rate: {fs} Hz\n")

# Queues for async processing
input_queue = queue.Queue(maxsize=3)  # Small buffer to prevent overflow
output_queue = queue.Queue(maxsize=3)
processing_lock = threading.Lock()

# Toggle state for voice isolation
voice_isolation_enabled = True  # Start with isolation enabled
toggle_lock = threading.Lock()


def process_audio_worker():
    """Worker thread that processes audio asynchronously"""
    processed_frames = 0
    while True:
        try:
            # Get audio from input queue
            audio_data = input_queue.get(timeout=0.1)
            if audio_data is None:  # Shutdown signal
                break

            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)

            # Process with DeepFilterNet
            try:
                with torch.no_grad():
                    enhanced = enhance(model, df_state, audio_tensor)

                # Put processed audio in output queue
                enhanced_np = enhanced.squeeze().cpu().numpy()
                processed_frames += 1

                try:
                    output_queue.put_nowait(enhanced_np)
                except queue.Full:
                    # Drop oldest if queue is full
                    try:
                        output_queue.get_nowait()
                        output_queue.put_nowait(enhanced_np)
                    except queue.Empty:
                        pass

                if processed_frames % 10 == 0:
                    print(f"Processed {processed_frames} frames")
            except Exception as e:
                print(f"Enhance error: {e}")
                import traceback

                traceback.print_exc()

            input_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Processing error: {e}")
            import traceback

            traceback.print_exc()
            continue


# Start processing worker thread
worker_thread = threading.Thread(target=process_audio_worker, daemon=True)
worker_thread.start()


def keyboard_listener():
    """Listen for keypresses to toggle voice isolation"""
    global voice_isolation_enabled

    # Save terminal settings
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())

        print("\n" + "=" * 60)
        print("Press SPACEBAR to toggle between:")
        print("  - Voice Isolation (noise suppression)")
        print("  - Full Audio (passthrough)")
        print("Press 'q' to quit")
        print("=" * 60 + "\n")

        while True:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1)

                if key == " ":  # Spacebar toggles
                    with toggle_lock:
                        voice_isolation_enabled = not voice_isolation_enabled
                        status = (
                            "VOICE ISOLATION"
                            if voice_isolation_enabled
                            else "FULL AUDIO"
                        )
                        print(f"\n🔄 Toggled to: {status}")
                elif key == "q" or key == "\x1b":  # 'q' or ESC to quit
                    print("\n👋 Quitting...")
                    sys.exit(0)

    except KeyboardInterrupt:
        pass
    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


# Start keyboard listener thread
keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
keyboard_thread.start()


passthrough_count = 0
processed_count = 0


callback_count = 0


def audio_callback(indata, outdata, frames, time, status):
    global passthrough_count, processed_count, callback_count

    callback_count += 1

    if status:
        print(f"Status: {status}")

    # Debug: Print on first few callbacks to verify it's being called
    if callback_count <= 3:
        print(
            f"Callback #{callback_count}: frames={frames}, input shape={indata.shape}, output shape={outdata.shape}"
        )
        print(f"  Input range: [{indata.min():.4f}, {indata.max():.4f}]")

    # Always start with passthrough
    audio_out = indata[:, 0].copy()

    # Check if voice isolation is enabled
    with toggle_lock:
        isolation_enabled = voice_isolation_enabled

    if isolation_enabled:
        # Voice isolation mode - try to use processed audio
        # Quickly put input audio in queue (non-blocking)
        try:
            input_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            # If queue is full, skip this frame to prevent overflow
            pass

        # Try to get processed audio from output queue
        try:
            enhanced = output_queue.get_nowait()
            # Ensure we have enough samples
            if len(enhanced) >= frames:
                audio_out = enhanced[:frames]
                processed_count += 1
                if processed_count % 50 == 0:  # Print every 50 processed frames
                    print(
                        f"✓ Voice isolation active: {processed_count} frames processed"
                    )
            else:
                # Pad if needed (shouldn't happen, but safety)
                audio_out = np.pad(
                    enhanced, (0, frames - len(enhanced)), mode="constant"
                )
        except queue.Empty:
            # If no processed audio available, use passthrough (processing is slow)
            passthrough_count += 1
            if passthrough_count == 1:
                print(
                    "⚠ Voice isolation enabled but processing slow - using passthrough temporarily"
                )
            elif passthrough_count % 100 == 0:  # Print every 100 passthrough frames
                print(
                    f"⚠ Still waiting for processed audio: {passthrough_count} (processed: {processed_count})"
                )
    else:
        # Full audio mode - always use passthrough, don't process
        # (audio_out is already set to passthrough above, no processing needed)
        pass

    # CRITICAL: Always output audio (either processed or passthrough)
    # This ensures you always hear something
    outdata[:, 0] = audio_out

    # Verify output was written
    if callback_count <= 3:
        print(f"  Output range: [{outdata.min():.4f}, {outdata.max():.4f}]")


# Instructions will be printed by keyboard_listener
print("\n" + "=" * 60)
print("🎤 Real-time Voice Isolation System")
print("=" * 60)

# Duplex stream (mic → speakers)
try:
    with sd.Stream(
        samplerate=fs,
        blocksize=blocksize,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        latency="low",
    ):
        print("🎤 Audio stream started!")
        print("Voice isolation is ENABLED by default.")
        print("Use SPACEBAR to toggle, 'q' to quit.\n")
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("\nStopping...")
            input_queue.put(None)  # Signal worker to stop
except Exception as e:
    print(f"Stream error: {e}")
    import traceback

    traceback.print_exc()
