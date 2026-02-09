"""Voice-enabled webcam test client — exercises voice agent pipeline.

Usage:
    # Start server with SAM3 + voice agent
    python -m server.main --device cuda --pipeline sam3

    # Start voice-enabled webcam client
    python -m server.test_client_voice

Architecture:
    FrameGrabber thread  → webcam capture
    AudioGrabber thread  → microphone PCM16 capture
    sender loop          → sends frames + audio via protobuf WebSocket
    receiver loop        → displays server responses + voice agent state

Voice UX:
    1. Say "Hey Vibe" to start recording
    2. Make your request (e.g., "blur the laptop")
    3. Say "Thank you" to execute

The server dashboard shows:
    - Active prompts (objects being segmented)
    - Voice agent status (listening/recording)
    - Last command + response
    - Real-time mask overlays
"""

import asyncio
import argparse
import time
import sys
import threading
import queue

import cv2
import numpy as np
import websockets

sys.path.insert(0, ".")
from server.proto import socialsense_pb2 as pb


class FrameGrabber:
    """Captures webcam frames in background thread."""

    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        self._cap = cv2.VideoCapture(camera_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            time.sleep(0.01)

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        self._cap.release()


class AudioGrabber:
    """Captures microphone audio in background thread.

    Records PCM16 mono at 16kHz and queues chunks for sending.
    """

    def __init__(self, sample_rate=16000, chunk_duration=0.5):
        self._sr = sample_rate
        self._chunk_duration = chunk_duration
        self._chunk_samples = int(sample_rate * chunk_duration)
        self._queue = queue.Queue()
        self._running = True
        self._thread = None
        self._available = False

    def start(self):
        """Start audio capture thread."""
        try:
            import sounddevice as sd
            self._available = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            print(f"Audio capture started: {self._sr}Hz, {self._chunk_duration}s chunks")
        except ImportError:
            print("sounddevice not installed — audio capture disabled")
            print("  Install with: pip install sounddevice")

    def _run(self):
        import sounddevice as sd

        def callback(indata, frames, time_info, status):
            if status:
                print(f"Audio status: {status}")
            # Convert float32 [-1,1] to PCM16 int16
            pcm16 = (indata[:, 0] * 32767).astype(np.int16)
            self._queue.put(pcm16.tobytes())

        with sd.InputStream(
            samplerate=self._sr,
            channels=1,
            dtype="float32",
            blocksize=self._chunk_samples,
            callback=callback,
        ):
            while self._running:
                time.sleep(0.1)

    def get_audio_chunk(self):
        """Get next audio chunk (non-blocking). Returns None if queue empty."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self._running = False

    @property
    def available(self):
        return self._available


async def send_loop(ws, frame_grabber, audio_grabber, target_fps=30):
    """Send frames + audio to server."""
    frame_id = 0
    frame_interval = 1.0 / target_fps

    while True:
        t0 = time.perf_counter()

        # Send frame
        frame = frame_grabber.get_frame()
        if frame is not None:
            frame_id += 1
            h, w = frame.shape[:2]

            # Encode JPEG
            _, jpeg_bytes = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

            msg = pb.ClientMessage()
            msg.frame_id = frame_id
            msg.timestamp_ms = time.time() * 1000
            msg.frame.jpeg_data = jpeg_bytes.tobytes()
            msg.frame.width = w
            msg.frame.height = h
            msg.frame.quality = 85

            try:
                await ws.send(msg.SerializeToString())
            except Exception as e:
                print(f"Send frame error: {e}")
                break

        # Send audio chunks (non-blocking)
        if audio_grabber.available:
            for _ in range(10):  # drain queue
                audio_chunk = audio_grabber.get_audio_chunk()
                if audio_chunk is None:
                    break

                audio_msg = pb.ClientMessage()
                audio_msg.frame_id = frame_id
                audio_msg.timestamp_ms = time.time() * 1000
                audio_msg.audio.pcm16_data = audio_chunk
                audio_msg.audio.sample_rate = audio_grabber._sr
                audio_msg.audio.num_samples = len(audio_chunk) // 2  # PCM16 = 2 bytes/sample

                try:
                    await ws.send(audio_msg.SerializeToString())
                except Exception as e:
                    print(f"Send audio error: {e}")
                    break

        # Maintain target FPS
        elapsed = time.perf_counter() - t0
        if elapsed < frame_interval:
            await asyncio.sleep(frame_interval - elapsed)


async def receive_loop(ws):
    """Receive server responses and display voice agent state."""
    last_print_time = 0.0
    print_interval = 2.0  # seconds

    async for raw_msg in ws:
        try:
            msg = pb.ServerMessage()
            msg.ParseFromString(raw_msg)

            now = time.time()
            if now - last_print_time >= print_interval:
                last_print_time = now

                # Print voice agent state
                if msg.conversation.voice_agent.listening or msg.conversation.voice_agent.recording:
                    if msg.conversation.voice_agent.recording:
                        status = "🔴 RECORDING"
                        partial = msg.conversation.voice_agent.partial_transcript
                        if partial:
                            print(f"{status}: \"{partial}\"")
                        else:
                            print(status)
                    else:
                        status = "🎤 Listening (say 'Hey Vibe')"
                        print(status)

                    if msg.conversation.voice_agent.last_response:
                        print(f"  Response: {msg.conversation.voice_agent.last_response}")

                # Print active effects
                effects_summary = []
                for seg in msg.segments:
                    if seg.effect.effect_type and seg.effect.effect_type != "none":
                        effects_summary.append(f"{seg.label}:{seg.effect.effect_type}")

                if effects_summary:
                    print(f"  Active effects: {', '.join(effects_summary)}")

                # Full-screen filter
                if msg.conversation.voice_agent.full_screen_filter.filter_type != "none":
                    fs = msg.conversation.voice_agent.full_screen_filter
                    print(f"  Full-screen: {fs.filter_type} ({fs.intensity:.1f})")

                print(f"  Segments: {len(msg.segments)}, FPS: {1000/msg.processing_time_ms:.0f}")
                print()

        except Exception as e:
            print(f"Receive error: {e}")
            break


async def main():
    parser = argparse.ArgumentParser(description="Voice-enabled webcam test client")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio capture")
    args = parser.parse_args()

    print("=" * 60)
    print("Voice-Enabled Webcam Test Client")
    print("=" * 60)
    print(f"Server: ws://{args.host}:{args.port}")
    print(f"Camera: {args.camera} ({args.width}x{args.height} @ {args.fps}fps)")
    print()
    print("Voice Commands:")
    print("  1. Say 'Hey Vibe' to start")
    print("  2. Make your request (e.g., 'blur the laptop')")
    print("  3. Say 'Thank you' to execute")
    print()
    print("Examples:")
    print("  - 'Hey Vibe, blur the monitor, thank you'")
    print("  - 'Hey Vibe, it's too bright, dim it, thank you'")
    print("  - 'Hey Vibe, stop blurring the laptop, thank you'")
    print()
    print("Watch the server dashboard window for visual feedback!")
    print("=" * 60)
    print()

    # Start frame grabber
    frame_grabber = FrameGrabber(args.camera, args.width, args.height, args.fps)
    frame_grabber.start()
    print("✓ Webcam started")

    # Start audio grabber
    audio_grabber = AudioGrabber()
    if not args.no_audio:
        audio_grabber.start()
        if audio_grabber.available:
            print("✓ Microphone started")
        else:
            print("⚠ Audio unavailable (install sounddevice)")
    else:
        print("⚠ Audio disabled (--no-audio)")

    print()
    print("Connecting to server...")

    try:
        uri = f"ws://{args.host}:{args.port}"
        async with websockets.connect(uri, max_size=10*1024*1024) as ws:
            print(f"✓ Connected to {uri}")
            print()

            # Run send + receive concurrently
            await asyncio.gather(
                send_loop(ws, frame_grabber, audio_grabber, args.fps),
                receive_loop(ws),
            )

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        frame_grabber.stop()
        audio_grabber.stop()


if __name__ == "__main__":
    asyncio.run(main())
