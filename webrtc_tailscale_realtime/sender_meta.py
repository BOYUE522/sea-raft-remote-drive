import argparse
import asyncio
import json
import time
from fractions import Fraction

import aiohttp
import cv2
from av import VideoFrame
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCRtpSender,
    VideoStreamTrack,
)


class StaticTelemetry:
    def __init__(self, speed, steer):
        self.speed = speed
        self.steer = steer

    def get(self):
        return self.speed, self.steer


class UdpTelemetryProtocol(asyncio.DatagramProtocol):
    def __init__(self, state):
        self.state = state

    def datagram_received(self, data, addr):
        text = data.decode("utf-8", errors="ignore").strip()
        if not text:
            return
        try:
            payload = json.loads(text)
            speed = float(payload.get("speed", self.state.speed))
            steer = float(payload.get("steer", self.state.steer))
        except ValueError:
            return
        except json.JSONDecodeError:
            try:
                parts = [float(x) for x in text.split(",")]
                if len(parts) >= 2:
                    speed, steer = parts[0], parts[1]
                elif len(parts) == 1:
                    speed, steer = parts[0], self.state.steer
                else:
                    return
            except ValueError:
                return
        self.state.speed = speed
        self.state.steer = steer


class UdpTelemetryState:
    def __init__(self, speed, steer):
        self.speed = speed
        self.steer = steer

    def get(self):
        return self.speed, self.steer


def parse_camera(value):
    if value.isdigit():
        return int(value)
    return value


def set_preferred_codec(transceiver, codec_name):
    if not codec_name:
        return False
    codec_name = codec_name.lower()
    caps = RTCRtpSender.getCapabilities("video").codecs
    preferred = [c for c in caps if c.mimeType.lower() == f"video/{codec_name}"]
    if not preferred:
        return False
    transceiver.setCodecPreferences(preferred)
    return True


class CameraVideoTrack(VideoStreamTrack):
    def __init__(self, source, width, height, fps, telemetry, meta_sender):
        super().__init__()
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {source}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.fps = fps if fps > 0 else 30
        self.time_base = Fraction(1, self.fps)
        self.frame_id = 0
        self.telemetry = telemetry
        self.meta_sender = meta_sender

    async def recv(self):
        self.frame_id += 1
        ok, frame = self.cap.read()
        if not ok:
            await asyncio.sleep(0.01)
            return await self.recv()

        speed, steer = self.telemetry.get()
        ts_ms = int(time.time() * 1000)
        self.meta_sender.send(self.frame_id, ts_ms, speed, steer, self.fps)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = self.frame_id
        video_frame.time_base = self.time_base
        return video_frame


class MetaSender:
    def __init__(self):
        self.channel = None

    def attach(self, channel):
        self.channel = channel

    def send(self, frame_id, ts_ms, speed, steer, fps):
        if self.channel is None or self.channel.readyState != "open":
            return
        payload = {
            "frame_id": frame_id,
            "ts_ms": ts_ms,
            "speed": float(speed),
            "steer": float(steer),
            "fps": float(fps),
        }
        self.channel.send(json.dumps(payload))


async def wait_for_ice(pc):
    if pc.iceGatheringState == "complete":
        return
    done = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def check_state():
        if pc.iceGatheringState == "complete":
            done.set()

    await done.wait()


async def setup_telemetry(args):
    if args.telemetry_mode == "udp":
        state = UdpTelemetryState(args.speed, args.steer)
        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(
            lambda: UdpTelemetryProtocol(state),
            local_addr=(args.telemetry_udp_host, args.telemetry_udp_port),
        )
        return state
    return StaticTelemetry(args.speed, args.steer)


async def run(args):
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
    meta_sender = MetaSender()

    channel = pc.createDataChannel("telemetry")
    meta_sender.attach(channel)

    telemetry = await setup_telemetry(args)
    source = parse_camera(args.camera)
    track = CameraVideoTrack(source, args.width, args.height, args.fps, telemetry, meta_sender)

    transceiver = pc.addTransceiver(track, direction="sendonly")
    if args.codec and not set_preferred_codec(transceiver, args.codec):
        print(f"Requested codec not available: {args.codec}")

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    await wait_for_ice(pc)

    async with aiohttp.ClientSession() as session:
        payload = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        async with session.post(args.signal, json=payload) as resp:
            resp.raise_for_status()
            answer = await resp.json()

    await pc.setRemoteDescription(RTCSessionDescription(**answer))
    print("WebRTC connected. Press Ctrl+C to stop.")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await pc.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", required=True, help="Signaling URL, e.g. http://ip:8080/offer")
    parser.add_argument("--camera", default="0", help="Camera index or device path")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--codec", type=str, default="vp8", help="vp8 or h264")
    parser.add_argument("--telemetry_mode", choices=["static", "udp"], default="static")
    parser.add_argument("--speed", type=float, default=0.0)
    parser.add_argument("--steer", type=float, default=0.0)
    parser.add_argument("--telemetry_udp_host", default="127.0.0.1")
    parser.add_argument("--telemetry_udp_port", type=int, default=5001)
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
