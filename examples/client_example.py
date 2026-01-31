"""
SocialSenseAR Python Client SDK Example

This example demonstrates how to connect to the SocialSenseAR server
from an AR headset or other client device.

Usage:
    python python_example.py --api-key YOUR_API_KEY --server-url ws://SERVER_IP:8000
"""

import asyncio
import json
import base64
import time
import argparse
from typing import Optional
import numpy as np
import websockets
from loguru import logger


class SocialSenseARClient:
    """Client for connecting to SocialSenseAR streaming server."""

    def __init__(self, server_url: str, api_key: str, session_id: Optional[str] = None):
        """
        Initialize client.

        Args:
            server_url: WebSocket server URL (e.g., ws://SERVER_IP:8000)
            api_key: API key for authentication
            session_id: Optional session ID (generated if not provided)
        """
        self.server_url = server_url
        self.api_key = api_key
        self.session_id = session_id or f"client_{int(time.time())}"

        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.is_connected = False

        self.frame_count = 0
        self.last_latency_ms = 0.0

        logger.info(f"SocialSenseAR Client initialized for session: {self.session_id}")

    async def connect(self):
        """Connect to the streaming server."""
        try:
            # Build WebSocket URL with session ID
            ws_url = f"{self.server_url}/ws/{self.session_id}"

            # Connect with API key in header
            extra_headers = {"X-API-Key": self.api_key}

            self.websocket = await websockets.connect(ws_url, extra_headers=extra_headers)
            self.is_connected = True

            logger.success(f"Connected to server: {self.server_url}")

            # Start heartbeat task
            asyncio.create_task(self._heartbeat_loop())

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise

    async def disconnect(self):
        """Disconnect from server."""
        self.is_connected = False

        if self.websocket:
            await self.websocket.close()
            logger.info("Disconnected from server")

    async def send_frame(self, frame: np.ndarray):
        """
        Send a video frame to the server for processing.

        Args:
            frame: Video frame as numpy array (H x W x 3, BGR format)
        """
        if not self.is_connected or not self.websocket:
            logger.warning("Not connected to server")
            return

        try:
            timestamp = time.time()

            # Serialize frame (in production, would use H.264 encoding)
            import io

            buffer = io.BytesIO()
            np.save(buffer, frame, allow_pickle=False)
            frame_bytes = buffer.getvalue()

            # Encode as base64
            frame_b64 = base64.b64encode(frame_bytes).decode("utf-8")

            # Send frame message
            message = {
                "type": "frame",
                "data": frame_b64,
                "timestamp": timestamp,
            }

            await self.websocket.send(json.dumps(message))
            self.frame_count += 1

            logger.debug(f"Sent frame #{self.frame_count}")

        except Exception as e:
            logger.error(f"Error sending frame: {e}")

    async def send_command(self, command: str):
        """
        Send a text command to the server.

        Args:
            command: Command text (e.g., "blur the background")
        """
        if not self.is_connected or not self.websocket:
            logger.warning("Not connected to server")
            return

        try:
            message = {"type": "command", "command": command}

            await self.websocket.send(json.dumps(message))
            logger.info(f"Sent command: {command}")

        except Exception as e:
            logger.error(f"Error sending command: {e}")

    async def receive_frame(self) -> Optional[np.ndarray]:
        """
        Receive a processed frame from the server.

        Returns:
            Processed frame as numpy array, or None if no frame available
        """
        if not self.is_connected or not self.websocket:
            return None

        try:
            message_str = await asyncio.wait_for(self.websocket.recv(), timeout=0.1)
            message = json.loads(message_str)

            if message.get("type") == "frame":
                # Decode frame
                frame_b64 = message.get("data")
                server_timestamp = message.get("timestamp")
                latency_ms = message.get("latency_ms", 0)

                self.last_latency_ms = latency_ms

                # Decode base64 and decompress (in production, would use H.264 decoding)
                # For this example, we're using numpy serialization
                logger.debug(
                    f"Received processed frame (latency: {latency_ms:.1f}ms)"
                )

                # In a real implementation, you would decode the H.264 data here
                # For now, just acknowledge receipt
                return None

            elif message.get("type") == "pong":
                logger.debug("Heartbeat acknowledged")

        except asyncio.TimeoutError:
            pass  # No frame available
        except Exception as e:
            logger.error(f"Error receiving frame: {e}")

        return None

    async def _heartbeat_loop(self):
        """Send periodic heartbeat to keep connection alive."""
        while self.is_connected:
            try:
                if self.websocket:
                    await self.websocket.send(json.dumps({"type": "ping"}))
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                break


async def example_usage(server_url: str, api_key: str):
    """
    Example usage of the SocialSenseAR client.

    This simulates an AR headset sending frames and receiving processed results.
    """
    client = SocialSenseARClient(server_url, api_key)

    try:
        # Connect to server
        await client.connect()

        # Simulate sending frames
        for i in range(10):
            # Create a dummy frame (in reality, this would come from camera)
            frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

            # Send frame for processing
            await client.send_frame(frame)

            # Try to receive processed frame
            processed = await client.receive_frame()

            # Send a command every 5 frames
            if i == 4:
                await client.send_command("blur the background")

            # Simulate frame rate (30 FPS)
            await asyncio.sleep(1.0 / 30.0)

        logger.success(
            f"Completed sending {client.frame_count} frames. "
            f"Last latency: {client.last_latency_ms:.1f}ms"
        )

    finally:
        await client.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SocialSenseAR Python Client Example")
    parser.add_argument(
        "--server-url",
        default="ws://localhost:8000",
        help="WebSocket server URL",
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="API key for authentication",
    )

    args = parser.parse_args()

    # Run example
    asyncio.run(example_usage(args.server_url, args.api_key))
