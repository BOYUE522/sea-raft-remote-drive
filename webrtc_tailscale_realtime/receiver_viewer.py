import argparse
import asyncio
import json
import queue
import threading
import time

import cv2
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration


class ViewerState:
    def __init__(self, display=True, queue_size=120):
        self.display = display
        self.queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread = None
        self._frame_count = 0
        self._last_fps_wall = time.time()
        if self.display:
            self._thread = threading.Thread(target=self._display_loop, daemon=True)
            self._thread.start()

    def on_frame(self, frame):
        try:
            bgr = frame.to_ndarray(format="bgr24")
        except Exception:
            return
        if not isinstance(bgr, np.ndarray) or bgr.ndim != 3:
            return
        self._frame_count += 1
        now = time.time()
        if now - self._last_fps_wall >= 1.0:
            print(f"[viewer] video fps ~ {self._frame_count}")
            self._frame_count = 0
            self._last_fps_wall = now
        if not self.display:
            return
        try:
            self.queue.put_nowait(bgr)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(bgr)
            except queue.Full:
                pass

    def _display_loop(self):
        cv2.namedWindow("webrtc_viewer", cv2.WINDOW_NORMAL)
        placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "waiting for frames...", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("webrtc_viewer", placeholder)
        cv2.waitKey(1)
        while not self._stop.is_set():
            frame = None
            try:
                while True:
                    frame = self.queue.get_nowait()
            except queue.Empty:
                pass
            if frame is not None:
                cv2.imshow("webrtc_viewer", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._stop.set()
                    break
            time.sleep(0.005)

    def stop(self):
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        cv2.destroyAllWindows()


pcs = set()


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
    pcs.add(pc)
    state = request.app["state"]

    @pc.on("track")
    def on_track(track):
        print(f"[viewer] track: {track.kind}")
        if track.kind == "video":
            asyncio.ensure_future(consume_video(track, state))

    @pc.on("datachannel")
    def on_datachannel(channel):
        print(f"[viewer] datachannel: {channel.label}")

        @channel.on("message")
        def on_message(message):
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                return
            if "ts_ms" in payload:
                print("[viewer] telemetry sample:", payload)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})


async def consume_video(track, state):
    while True:
        frame = await track.recv()
        state.on_frame(frame)


async def on_shutdown(app):
    state = app.get("state")
    if state is not None:
        state.stop()
    coros = [pc.close() for pc in pcs]
    if coros:
        await asyncio.gather(*coros)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no_display", action="store_true")
    args = parser.parse_args()

    app = web.Application()
    app["state"] = ViewerState(display=not args.no_display)
    app.on_shutdown.append(on_shutdown)
    app.router.add_post("/offer", offer)

    web.run_app(app, host=args.listen, port=args.port)


if __name__ == "__main__":
    main()
