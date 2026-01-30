#!/usr/bin/env python3
"""
Quest Passthrough Camera Pipeline Server

Receives raw passthrough frames from Quest via TCP (over ADB USB),
processes them through your YOLO pipeline, and sends back the processed frames.

SETUP:
1. Connect Quest via USB
2. Run: adb reverse tcp:9090 tcp:9090
3. Run this server: python quest_server.py
4. Launch the Quest app

The Quest app will connect to localhost:9090 (forwarded via ADB to the Quest).
"""
import socket
import struct
import time
import cv2
import numpy as np
from typing import Optional



class QuestPipelineServer:
    """TCP server that receives Quest frames, processes them, and sends back results."""

    def __init__(self, host: str = '0.0.0.0', port: int = 9090):
        self.host = host
        self.port = port

        # Initialize YOLO model with TensorRT for best performance
        print("Initializing YOLO model...", flush=True)
        from ultralytics import YOLO
        import torch
        import os

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}", flush=True)

        # Try TensorRT engine first, fall back to PyTorch
        script_dir = os.path.dirname(os.path.abspath(__file__))
        engine_path = os.path.join(script_dir, 'yolov8n-seg.engine')

        if os.path.exists(engine_path) and self.device == 'cuda':
            print(f"Loading TensorRT engine: {engine_path}", flush=True)
            self.model = YOLO(engine_path)
        else:
            print("Loading PyTorch model (TensorRT not available)", flush=True)
            self.model = YOLO('yolov8n-seg.pt')

        print("YOLO model loaded!", flush=True)

        # Server state
        self.server_socket: Optional[socket.socket] = None
        self.client_socket: Optional[socket.socket] = None
        self.running = False

        # Stats
        self.frames_processed = 0
        self.start_time = time.time()
        self.last_stats_time = time.time()
        self.fps = 0

    def start(self):
        """Start the server."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)

        print(f"\n{'='*50}")
        print(f"Quest Pipeline Server")
        print(f"{'='*50}")
        print(f"Listening on {self.host}:{self.port}")
        print(f"")
        print(f"SETUP STEPS:")
        print(f"1. Connect Quest via USB cable")
        print(f"2. Run: adb reverse tcp:{self.port} tcp:{self.port}")
        print(f"3. Launch Quest app")
        print(f"{'='*50}\n")

        self.running = True

        while self.running:
            try:
                print("Waiting for Quest to connect...")
                self.client_socket, addr = self.server_socket.accept()
                self.client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print(f"Quest connected from {addr}")

                self._handle_client()

            except KeyboardInterrupt:
                print("\nShutting down...")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)

        self.stop()

    def _handle_client(self):
        """Handle connected client - receive frames, process, send back."""
        try:
            while self.running:
                # Receive frame size (4 bytes, little-endian)
                size_data = self._recv_all(4)
                if not size_data:
                    print("Client disconnected")
                    break

                frame_size = struct.unpack('<I', size_data)[0]

                # Receive frame data
                frame_data = self._recv_all(frame_size)
                if not frame_data:
                    print("Failed to receive frame")
                    break

                # Decode JPEG
                frame = cv2.imdecode(
                    np.frombuffer(frame_data, np.uint8),
                    cv2.IMREAD_COLOR
                )

                if frame is None:
                    print("Failed to decode frame")
                    continue

                # Process frame through your pipeline
                processed = self._process_frame(frame)

                # Show preview on PC
                cv2.imshow('Quest Preview', processed)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Preview closed by user")
                    break

                # Encode result as JPEG
                _, encoded = cv2.imencode(
                    '.jpg', processed,
                    [cv2.IMWRITE_JPEG_QUALITY, 85]
                )

                # Send back: size + data
                result_data = encoded.tobytes()
                self.client_socket.sendall(struct.pack('<I', len(result_data)))
                self.client_socket.sendall(result_data)

                # Update stats
                self.frames_processed += 1
                self._update_stats()

        except Exception as e:
            print(f"Client error: {e}")
        finally:
            cv2.destroyAllWindows()
            if self.client_socket:
                self.client_socket.close()
                self.client_socket = None

    def _recv_all(self, size: int) -> Optional[bytes]:
        """Receive exactly `size` bytes from socket."""
        data = bytearray()
        while len(data) < size:
            try:
                chunk = self.client_socket.recv(size - len(data))
                if not chunk:
                    return None
                data.extend(chunk)
            except:
                return None
        return bytes(data)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame (mono or stereo) and return processed result.

        Automatically detects stereo (side-by-side) frames based on aspect ratio.
        Stereo frames have width ~2x height, mono frames have width ~1.33x height.
        """
        h, w = frame.shape[:2]
        aspect = w / h

        # Detect stereo: side-by-side frames have aspect ~2.67 (2x 4:3)
        # Mono frames have aspect ~1.33 (4:3)
        is_stereo = aspect > 2.0

        if is_stereo:
            return self._process_stereo_frame(frame)
        else:
            return self._process_mono_frame(frame)

    def _process_mono_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process mono frame and return processed result."""
        h, w = frame.shape[:2]

        # Convert BGR to RGB for YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLO on single frame
        results = self.model(
            frame_rgb,
            verbose=False,
            classes=[0]  # person class only
        )

        # Extract mask
        mask = None
        if results and len(results) > 0 and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            if len(masks) > 0:
                combined = np.zeros((h, w), dtype=np.float32)
                for m in masks:
                    resized = cv2.resize(m, (w, h))
                    combined = np.maximum(combined, resized)
                mask = (combined * 255).astype(np.uint8)

        # Apply effect
        return self._apply_effect(frame, mask)

    def _process_stereo_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process stereo (side-by-side) frame with batched inference."""
        h, w = frame.shape[:2]
        half_w = w // 2

        # Split into left and right
        left_frame = frame[:, :half_w]
        right_frame = frame[:, half_w:]

        # Convert BGR to RGB
        left_rgb = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)

        # Batch both frames together for TensorRT (batch=2)
        results = self.model(
            [left_rgb, right_rgb],
            verbose=False,
            classes=[0]  # person class only
        )

        # Process results for each eye
        left_processed = self._apply_mask_from_result(left_frame, results[0] if len(results) > 0 else None)
        right_processed = self._apply_mask_from_result(right_frame, results[1] if len(results) > 1 else None)

        # Combine back side-by-side
        return np.hstack([left_processed, right_processed])

    def _apply_mask_from_result(self, frame: np.ndarray, result) -> np.ndarray:
        """Apply effect using YOLO result."""
        h, w = frame.shape[:2]
        mask = None

        if result is not None and result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            if len(masks) > 0:
                combined = np.zeros((h, w), dtype=np.float32)
                for m in masks:
                    resized = cv2.resize(m, (w, h))
                    combined = np.maximum(combined, resized)
                mask = (combined * 255).astype(np.uint8)

        return self._apply_effect(frame, mask)

    def _apply_effect(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply grayscale background, color person effect."""
        if mask is not None and mask.max() > 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            mask_3ch = mask[:, :, np.newaxis].astype(np.float32) / 255.0
            result = (frame.astype(np.float32) * mask_3ch +
                     gray_bgr.astype(np.float32) * (1 - mask_3ch))
            return result.astype(np.uint8)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _update_stats(self):
        """Print stats periodically."""
        now = time.time()
        if now - self.last_stats_time >= 2.0:
            elapsed = now - self.last_stats_time
            self.fps = self.frames_processed / elapsed if elapsed > 0 else 0
            print(f"Processing: {self.fps:.1f} fps | Total frames: {self.frames_processed}")
            self.frames_processed = 0
            self.last_stats_time = now

    def stop(self):
        """Stop the server."""
        self.running = False

        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()

        print("Server stopped")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Quest Pipeline Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=9090, help='Port to listen on')
    args = parser.parse_args()

    # Create and start server
    server = QuestPipelineServer(
        host=args.host,
        port=args.port
    )

    try:
        server.start()
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        server.stop()


if __name__ == '__main__':
    main()
