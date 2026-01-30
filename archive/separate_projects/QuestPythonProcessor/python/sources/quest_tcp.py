"""
Quest TCP source - receives passthrough frames from Unity over TCP.

Works with Unity's StereoPipelineClient.cs which sends JPEG frames
over ADB reverse port forwarding.

SETUP:
1. Connect Quest via USB
2. Run: adb reverse tcp:9090 tcp:9090
3. Run main.py with --source quest_tcp
4. Launch Unity Quest app
"""
import socket
import struct
import threading
import time
from typing import Optional, Tuple
import numpy as np
import cv2
import lz4.block

from .base import BaseSource

# Profiling
PROFILE = True


class QuestTCPSource(BaseSource):
    """TCP source that receives frames from Quest Unity app.

    Runs a TCP server that accepts connection from Quest (via ADB reverse).
    Frames are received as: [4-byte size][JPEG data]
    """

    def __init__(self, config=None):
        self.config = config
        self.host = '0.0.0.0'
        self.port = getattr(config, 'tcp_port', 9090) if config else 9090

        # Server state
        self.server_socket: Optional[socket.socket] = None
        self.client_socket: Optional[socket.socket] = None
        self.running = False
        self.connected = False

        # Frame buffer (thread-safe)
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_resolution = (0, 0)
        self._is_stereo = True  # Quest sends stereo

        # Background thread for receiving
        self.recv_thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """Start TCP server and wait for Quest connection."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.settimeout(1.0)  # Allow checking running flag
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)

            print(f"\n{'='*50}")
            print(f"Quest TCP Source")
            print(f"{'='*50}")
            print(f"Listening on {self.host}:{self.port}")
            print(f"")
            print(f"SETUP:")
            print(f"1. Connect Quest via USB")
            print(f"2. Run: adb reverse tcp:{self.port} tcp:{self.port}")
            print(f"3. Launch Quest app")
            print(f"{'='*50}\n")

            self.running = True

            # Start accept thread
            self.recv_thread = threading.Thread(target=self._accept_and_receive, daemon=True)
            self.recv_thread.start()

            return True

        except Exception as e:
            print(f"Failed to start TCP server: {e}")
            return False

    def stop(self) -> None:
        """Stop server and cleanup."""
        self.running = False
        self.connected = False

        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass

        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass

        if self.recv_thread:
            self.recv_thread.join(timeout=2)

        print("Quest TCP source stopped")

    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest received frame (non-blocking)."""
        with self.lock:
            return self.latest_frame

    @property
    def resolution(self) -> Tuple[int, int]:
        """Get frame resolution."""
        with self.lock:
            return self.frame_resolution

    @property
    def is_stereo(self) -> bool:
        """Quest sends stereo side-by-side frames."""
        return self._is_stereo

    def get_device_info(self) -> dict:
        """Get device info."""
        return {
            "device": "Quest (TCP)",
            "connected": self.connected,
            "port": self.port,
        }

    def send_processed_frame(self, frame: np.ndarray, quality: int = 85, use_raw: bool = False) -> bool:
        """Send processed frame back to Quest.

        This is called by QuestTCPUI to send results back.

        Args:
            frame: BGR frame to send
            quality: JPEG quality (1-100), ignored if use_raw=True
            use_raw: If True, send raw RGB24 pixels instead of JPEG (faster encode, larger transfer)

        Returns:
            True if successful
        """
        if not self.connected:
            print(f"[TCP-OUT] Cannot send: not connected")
            return False
        if not self.client_socket:
            print(f"[TCP-OUT] Cannot send: no client socket")
            return False

        try:
            t0 = time.time()

            if use_raw:
                # Send raw BGR pixels directly (skip BGR→RGB conversion, Unity will swap)
                # This saves ~7ms per frame vs converting to RGB
                # Flip vertically for Unity (OpenCV origin top-left, Unity texture bottom-left)
                frame_flipped = np.flipud(frame)
                h, w = frame_flipped.shape[:2]

                # Header: [4-byte magic 'BGR\0'][4-byte width][4-byte height][pixels]
                # Using 'BGR\0' magic tells Unity to swap R/B channels
                header = struct.pack('<4sII', b'BGR\x00', w, h)
                data = header + frame_flipped.tobytes()
            else:
                # Encode as JPEG
                _, encoded = cv2.imencode(
                    '.jpg', frame,
                    [cv2.IMWRITE_JPEG_QUALITY, quality]
                )
                data = encoded.tobytes()

            t1 = time.time()

            # Send: [4-byte size][data]
            self.client_socket.sendall(struct.pack('<I', len(data)))
            self.client_socket.sendall(data)

            t2 = time.time()

            if PROFILE and hasattr(self, '_send_count'):
                self._encode_times.append((t1 - t0) * 1000)
                self._net_send_times.append((t2 - t1) * 1000)
                self._send_count += 1
                self._out_sizes.append(len(data) // 1024)

                if time.time() - self._last_send_profile >= 2.0:
                    avg_enc = sum(self._encode_times) / len(self._encode_times) if self._encode_times else 0
                    avg_net = sum(self._net_send_times) / len(self._net_send_times) if self._net_send_times else 0
                    avg_size = sum(self._out_sizes) / len(self._out_sizes) if self._out_sizes else 0
                    print(f"[TCP-OUT] encode:{avg_enc:.1f}ms net:{avg_net:.1f}ms | size:{avg_size:.0f}KB")
                    self._encode_times.clear()
                    self._net_send_times.clear()
                    self._out_sizes.clear()
                    self._last_send_profile = time.time()
            elif PROFILE:
                # Initialize profiling vars
                self._encode_times = []
                self._net_send_times = []
                self._out_sizes = []
                self._send_count = 0
                self._last_send_profile = time.time()

            return True

        except Exception as e:
            print(f"Send error: {e}")
            self.connected = False
            return False

    def _accept_and_receive(self) -> None:
        """Background thread: accept connection and receive frames."""
        while self.running:
            # Wait for connection
            if not self.connected:
                try:
                    print("Waiting for Quest to connect...")
                    self.client_socket, addr = self.server_socket.accept()
                    self.client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
                    self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)
                    self.client_socket.settimeout(5.0)
                    self.connected = True
                    print(f"Quest connected from {addr}")
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"Accept error: {e}")
                    continue

            # Receive frames
            try:
                self._receive_loop()
            except Exception as e:
                if self.running:
                    print(f"Connection lost: {e}")
                self.connected = False
                if self.client_socket:
                    try:
                        self.client_socket.close()
                    except:
                        pass
                    self.client_socket = None

    def _receive_loop(self) -> None:
        """Receive frames from connected client."""
        frame_count = 0
        last_profile_time = time.time()
        wait_times = []  # Time waiting for Unity to send
        transfer_times = []  # Actual network transfer time
        decode_times = []
        raw_mode_logged = False

        while self.running and self.connected:
            t0 = time.time()

            # Receive frame size (4 bytes) - this includes wait time
            size_data = self._recv_all(4)
            if not size_data:
                raise ConnectionError("Client disconnected")

            t_header = time.time()  # Header received, frame transfer starting

            frame_size = struct.unpack('<I', size_data)[0]

            # Receive frame data - this is actual transfer time
            frame_data = self._recv_all(frame_size)
            if not frame_data:
                raise ConnectionError("Failed to receive frame")

            t1 = time.time()

            # Separate wait time vs transfer time
            wait_ms = (t_header - t0) * 1000  # Waiting for Unity
            transfer_ms = (t1 - t_header) * 1000  # Actual network transfer

            # Check frame type (first byte)
            frame_type = frame_data[0]

            if frame_type == 0x04:  # RGBA32 (native GPU format, fastest readback)
                width = frame_data[1] | (frame_data[2] << 8)
                height = frame_data[3] | (frame_data[4] << 8)
                pixels = frame_data[5:]
                expected_size = width * height * 4

                if not raw_mode_logged:
                    print(f"[TCP-IN] RGBA32 MODE: {width}x{height} = {len(pixels)//1024}KB (expected {expected_size//1024}KB)")
                    raw_mode_logged = True

                if len(pixels) != expected_size:
                    print(f"[TCP-IN] ERROR: Size mismatch! Got {len(pixels)}, expected {expected_size}")
                    continue

                # Unity RGBA32 is actually RGBA byte order
                # Convert to RGB (drop alpha)
                rgba = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 4))
                frame_rgb = rgba[:, :, :3].copy()  # RGB, drop alpha, make contiguous

            elif frame_type == 0x03:  # RGB565 (2 bytes per pixel, 33% smaller)
                width = frame_data[1] | (frame_data[2] << 8)
                height = frame_data[3] | (frame_data[4] << 8)
                pixels = frame_data[5:]

                if not raw_mode_logged:
                    print(f"[TCP-IN] RGB565 MODE: {width}x{height} = {len(pixels)//1024}KB (33% smaller!)")
                    raw_mode_logged = True

                # FAST RGB565 to RGB using OpenCV (optimized C code)
                rgb565_img = np.frombuffer(pixels, dtype=np.uint16).reshape((height, width))
                frame_rgb = cv2.cvtColor(rgb565_img, cv2.COLOR_BGR5652RGB)

            elif frame_type == 0x02:  # LZ4 compressed RGB24
                width = frame_data[1] | (frame_data[2] << 8)
                height = frame_data[3] | (frame_data[4] << 8)
                compressed = bytes(frame_data[5:])
                uncompressed_size = width * height * 3
                pixels = lz4.block.decompress(compressed[4:], uncompressed_size=uncompressed_size)

                if not raw_mode_logged:
                    print(f"[TCP-IN] LZ4 MODE: {width}x{height}")
                    raw_mode_logged = True

                frame_rgb = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 3))

            elif frame_type == 0x01:  # RAW RGB24 (uncompressed)
                width = frame_data[1] | (frame_data[2] << 8)
                height = frame_data[3] | (frame_data[4] << 8)
                pixels = frame_data[5:]

                if not raw_mode_logged:
                    print(f"[TCP-IN] RAW MODE: {width}x{height} = {len(pixels)//1024}KB")
                    raw_mode_logged = True

                frame_rgb = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 3))

            else:  # JPEG (type 0x00)
                jpeg_data = frame_data[1:]  # Skip type byte
                frame = cv2.imdecode(
                    np.frombuffer(jpeg_data, np.uint8),
                    cv2.IMREAD_COLOR
                )
                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    continue

            t2 = time.time()

            if frame_rgb is not None:
                # Only flip JPEG frames - RAW frames from Unity are already correctly oriented
                # (JPEG encoder does an implicit flip that RAW doesn't)
                if frame_type == 0x00:  # JPEG only
                    frame_rgb = np.flipud(frame_rgb).copy()

                with self.lock:
                    self.latest_frame = frame_rgb
                    self.frame_resolution = (frame_rgb.shape[1], frame_rgb.shape[0])

                    # Detect stereo from aspect ratio
                    aspect = frame_rgb.shape[1] / frame_rgb.shape[0]
                    self._is_stereo = aspect > 2.0

            # Profile
            if PROFILE:
                decode_ms = (t2 - t1) * 1000
                wait_times.append(wait_ms)
                transfer_times.append(transfer_ms)
                decode_times.append(decode_ms)
                frame_count += 1

                if time.time() - last_profile_time >= 2.0:
                    avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0
                    avg_transfer = sum(transfer_times) / len(transfer_times) if transfer_times else 0
                    avg_decode = sum(decode_times) / len(decode_times) if decode_times else 0
                    fps = frame_count / (time.time() - last_profile_time)
                    mode = {0x00: "JPEG", 0x01: "RAW24", 0x02: "LZ4", 0x03: "RGB565", 0x04: "RGBA32"}.get(frame_type, "???")

                    # Calculate actual transfer bandwidth (excluding wait time)
                    actual_bandwidth_mbps = (frame_size / (avg_transfer / 1000)) * 8 / (1024 * 1024) if avg_transfer > 0 else 0

                    # Identify bottleneck
                    bottleneck = ""
                    if avg_wait > avg_transfer and avg_wait > 20:
                        bottleneck = " << UNITY FRAME PRODUCTION SLOW"
                    elif avg_transfer > 20:
                        bottleneck = " << USB TRANSFER SLOW"
                    elif avg_decode > 20:
                        bottleneck = " << DECODE SLOW"

                    print(f"\n[TCP-IN] {mode} wait:{avg_wait:.1f}ms transfer:{avg_transfer:.1f}ms decode:{avg_decode:.1f}ms | {fps:.1f}fps | {frame_size//1024}KB | {actual_bandwidth_mbps:.0f}Mbps actual{bottleneck}")
                    wait_times.clear()
                    transfer_times.clear()
                    decode_times.clear()
                    frame_count = 0
                    last_profile_time = time.time()

    def _recv_all(self, size: int) -> Optional[bytes]:
        """Receive exactly `size` bytes."""
        data = bytearray()
        while len(data) < size:
            try:
                chunk = self.client_socket.recv(size - len(data))
                if not chunk:
                    return None
                data.extend(chunk)
            except socket.timeout:
                if not self.running:
                    return None
                continue
            except:
                return None
        return bytes(data)
