#!/usr/bin/env python3
"""Standalone PersonaPlex test — connects directly to Moshi, sends mic audio, prints responses.

Usage:
    # Against Modal deployment:
    python tools/test_personaplex.py --url ws://localhost:8998/api/chat

    # With custom prompt:
    python tools/test_personaplex.py --url ws://... --prompt "You are a helpful assistant."

Say "Hey Vibe, blur the laptop" and watch for [COMMAND:...] tags in text output.
Press Ctrl+C to stop.
"""

import argparse
import asyncio
import logging
import sys
import time
from urllib.parse import urlencode

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_personaplex")

# PersonaPlex protocol
MSG_HANDSHAKE = 0x00
MSG_AUDIO = 0x01
MSG_TEXT = 0x02

SAMPLE_RATE = 24000  # Moshi's native rate


SYSTEM_PROMPT = """You are Vibe, an AR assistant. You respond when the user says "Hey Vibe".
When the user requests a visual effect, include a command tag like [COMMAND:blur:laptop].
Keep responses short."""


async def run(url: str, prompt: str, use_mic: bool):
    import aiohttp
    import sphn

    params = {
        "text_prompt": prompt,
        "voice_prompt": "NATF0.pt",
    }
    full_url = url + ("&" if "?" in url else "?") + urlencode(params)

    logger.info(f"Connecting to {url}")
    session = aiohttp.ClientSession()
    ws = await session.ws_connect(full_url, max_msg_size=10 * 1024 * 1024, heartbeat=20)
    logger.info("Connected, waiting for handshake...")

    ready = False
    opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)
    opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)
    pcm_buffer = np.empty(0, dtype=np.float32)
    frame_size = 960  # 40ms at 24kHz

    text_accum = ""
    audio_sent = 0
    audio_recv = 0
    text_recv = 0
    t_ready = None

    async def recv_loop():
        nonlocal ready, text_accum, audio_recv, text_recv, t_ready
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                data = msg.data
                if not data:
                    continue
                kind = data[0]
                if kind == MSG_HANDSHAKE:
                    ready = True
                    t_ready = time.time()
                    logger.info("=== HANDSHAKE RECEIVED — Moshi ready ===")
                elif kind == MSG_AUDIO:
                    audio_recv += 1
                    opus_reader.append_bytes(data[1:])
                    pcm = opus_reader.read_pcm()
                    if pcm.shape[-1] > 0 and audio_recv <= 3:
                        rms = np.sqrt(np.mean(pcm ** 2))
                        logger.info(f"  Audio response #{audio_recv}: {pcm.shape[-1]} samples, rms={rms:.4f}")
                elif kind == MSG_TEXT:
                    text_recv += 1
                    token = data[1:].decode("utf-8", errors="replace")
                    text_accum += token
                    sys.stdout.write(token)
                    sys.stdout.flush()
                else:
                    logger.warning(f"Unknown message kind: {kind}")
            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                logger.info(f"WebSocket closed: {msg.type}")
                break

    async def send_mic():
        nonlocal audio_sent, pcm_buffer
        try:
            import sounddevice as sd
        except ImportError:
            logger.error("sounddevice not installed. pip install sounddevice")
            return

        logger.info(f"Opening mic at {SAMPLE_RATE}Hz...")
        q = asyncio.Queue()

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Mic status: {status}")
            q.put_nowait(indata[:, 0].copy())

        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                blocksize=frame_size, callback=callback)
        stream.start()
        logger.info("Mic started. Speak now...")

        try:
            while True:
                pcm_chunk = await q.get()
                if not ready:
                    continue

                pcm_buffer = np.concatenate([pcm_buffer, pcm_chunk])
                while len(pcm_buffer) >= frame_size:
                    frame = np.ascontiguousarray(pcm_buffer[:frame_size], dtype=np.float32)
                    pcm_buffer = pcm_buffer[frame_size:]
                    opus_writer.append_pcm(frame)
                    opus_bytes = opus_writer.read_bytes()
                    if len(opus_bytes) > 0:
                        await ws.send_bytes(bytes([MSG_AUDIO]) + opus_bytes)
                        audio_sent += 1
                        if audio_sent == 1:
                            logger.info(f"  First Opus frame sent ({len(opus_bytes)}B)")
        finally:
            stream.stop()
            stream.close()

    async def send_sine():
        """Send a sine wave tone (440Hz) if no mic available — tests protocol only."""
        nonlocal audio_sent, pcm_buffer
        logger.info("No mic — sending 440Hz sine wave to test protocol...")
        t = 0.0
        while True:
            if not ready:
                await asyncio.sleep(0.1)
                continue
            # Generate 40ms of 440Hz sine
            n = frame_size
            samples = 0.3 * np.sin(2 * np.pi * 440 * (np.arange(n, dtype=np.float32) + t) / SAMPLE_RATE).astype(np.float32)
            t += n
            opus_writer.append_pcm(samples)
            opus_bytes = opus_writer.read_bytes()
            if len(opus_bytes) > 0:
                await ws.send_bytes(bytes([MSG_AUDIO]) + opus_bytes)
                audio_sent += 1
                if audio_sent == 1:
                    logger.info(f"  First sine Opus frame sent ({len(opus_bytes)}B)")
            await asyncio.sleep(0.04)  # ~real-time

    async def status_loop():
        while True:
            await asyncio.sleep(5.0)
            elapsed = f"{time.time() - t_ready:.1f}s" if t_ready else "N/A"
            logger.info(f"Status: sent={audio_sent} | recv_audio={audio_recv} recv_text={text_recv} | "
                        f"elapsed={elapsed} | text='{text_accum[-80:]}'")

    send_fn = send_mic if use_mic else send_sine
    try:
        await asyncio.gather(recv_loop(), send_fn(), status_loop())
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        await ws.close()
        await session.close()
        logger.info(f"\nFinal: sent={audio_sent} | recv_audio={audio_recv} recv_text={text_recv}")
        if text_accum:
            logger.info(f"Full text: {text_accum}")
        else:
            logger.info("NO TEXT TOKENS RECEIVED — Moshi produced no text output")


def main():
    parser = argparse.ArgumentParser(description="Test PersonaPlex directly")
    parser.add_argument("--url", default="ws://localhost:8998/api/chat")
    parser.add_argument("--prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--no-mic", action="store_true", help="Send sine wave instead of mic audio")
    args = parser.parse_args()
    asyncio.run(run(args.url, args.prompt, use_mic=not args.no_mic))


if __name__ == "__main__":
    main()
