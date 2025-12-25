import argparse
import asyncio
import json
import queue
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration

import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'core'))

from config.parser import parse_args
from raft import RAFT
from utils.utils import load_ckpt, coords_grid, bilinear_sampler


class PredictionSequence:
    def __init__(
        self,
        base_ts,
        base_speed,
        base_frame,
        flow_forward,
        interval_ms,
        base_interval_ms,
        horizon,
        eps,
        use_speed_scale,
    ):
        self.base_ts = base_ts
        self.base_speed = base_speed
        self.base_frame = base_frame
        self.flow_forward = flow_forward
        self.interval_ms = interval_ms
        self.base_interval_ms = base_interval_ms
        self.horizon = horizon
        self.eps = eps
        self.use_speed_scale = use_speed_scale

    def in_range(self, ts_ms):
        step = (ts_ms - self.base_ts) / self.base_interval_ms
        return 0.0 < step <= float(self.horizon)

    def make(self, ts_ms, speed_target):
        step = (ts_ms - self.base_ts) / self.base_interval_ms
        step = max(0.0, min(float(self.horizon), step))
        if self.use_speed_scale:
            scale = speed_target / max(self.eps, self.base_speed)
        else:
            scale = 1.0
        flow_scaled = self.flow_forward * (scale * step)
        return warp_frame(self.base_frame, flow_scaled)


def tensor_from_bgr(bgr, device):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1)[None].to(device)
    return tensor


def tensor_to_bgr(tensor):
    img = tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def warp_frame(image, flow):
    _, _, h, w = image.shape
    coords = coords_grid(1, h, w, device=image.device)
    coords = coords + flow
    coords = coords.permute(0, 2, 3, 1)
    return bilinear_sampler(image, coords)


class ReceiverState:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU.")
            self.device = torch.device("cpu")

        self.model = None
        if not args.raw_only:
            self.model = RAFT(args).to(self.device)
            load_ckpt(self.model, args.path)
            self.model.eval()

        self.interval_ms = 1000.0 / args.fps
        self.display_interval = 1.0 / max(1.0, args.fps)
        self.gap_threshold = args.gap_threshold
        self.low_fps_threshold = args.low_fps_threshold
        self.window_size = args.window_size
        self.horizon = args.horizon
        self.speed_eps = args.speed_eps
        self.use_speed_scale = args.speed_scale
        self.meta_timeout = args.meta_timeout
        self.display = not args.no_display
        self.raw_only = args.raw_only

        self.meta_queue = deque(maxlen=args.queue_size)
        self.frame_queue = deque(maxlen=args.queue_size)
        self.pair_queue = queue.Queue(maxsize=args.queue_size)
        self.display_queue = queue.Queue(maxsize=args.queue_size)
        self.window = []
        self.window_created = False
        self.sequences = deque(maxlen=10)
        self.seq_lock = threading.Lock()
        self.raft_queue = queue.Queue(maxsize=args.queue_size)
        self.last_real = None
        self.have_meta = False
        self.first_video_wall = None
        self._printed_no_meta = False
        self._printed_meta = False
        self._video_count = 0
        self._last_fps_wall = time.time()
        self.input_fps = None
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._process_loop, daemon=True)
        self._worker.start()
        self._raft_worker = None
        if not self.raw_only:
            self._raft_worker = threading.Thread(target=self._raft_loop, daemon=True)
            self._raft_worker.start()

    def on_meta(self, payload):
        if not self.have_meta:
            self.have_meta = True
            self.first_video_wall = None
            self.last_real = None
            self.window = []
            self.window_created = False
            with self.seq_lock:
                self.sequences.clear()
            self.meta_queue.clear()
            self.frame_queue.clear()
            if not self._printed_meta:
                print("[receiver] telemetry connected")
                self._printed_meta = True
        self.meta_queue.append(payload)
        self._drain_pairs()

    def on_video(self, frame):
        try:
            bgr = frame.to_ndarray(format="bgr24")
        except Exception:
            return
        if not isinstance(bgr, np.ndarray) or bgr.ndim != 3:
            return
        self._video_count += 1
        now = time.time()
        if now - self._last_fps_wall >= 1.0:
            self.input_fps = float(self._video_count)
            print(f"[receiver] video fps ~ {self._video_count}")
            self._video_count = 0
            self._last_fps_wall = now
        if not self.have_meta:
            now = time.time()
            if self.first_video_wall is None:
                self.first_video_wall = now
            if (now - self.first_video_wall) >= self.meta_timeout:
                if not self._printed_no_meta:
                    print("[receiver] no telemetry yet, showing video only")
                    self._printed_no_meta = True
                dummy = {"ts_ms": now * 1000.0, "speed": 0.0, "_no_meta": True}
                try:
                    self.pair_queue.put_nowait((bgr, dummy))
                except queue.Full:
                    try:
                        self.pair_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.pair_queue.put_nowait((bgr, dummy))
                    except queue.Full:
                        pass
                return
        self.frame_queue.append(bgr)
        self._drain_pairs()

    def _drain_pairs(self):
        while self.meta_queue and self.frame_queue:
            meta = self.meta_queue.popleft()
            frame = self.frame_queue.popleft()
            try:
                self.pair_queue.put_nowait((frame, meta))
            except queue.Full:
                try:
                    self.pair_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.pair_queue.put_nowait((frame, meta))
                except queue.Full:
                    pass

    def _process_loop(self):
        while not self._stop_event.is_set():
            try:
                frame, meta = self.pair_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.process_real(frame, meta)
            finally:
                self.pair_queue.task_done()

    def stop(self):
        self._stop_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        if self._raft_worker is not None and self._raft_worker.is_alive():
            self._raft_worker.join(timeout=2.0)

    def process_real(self, bgr, meta):
        if meta.get("_no_meta"):
            self.show_frame(bgr, is_pred=False)
            return
        if self.raw_only:
            self.show_frame(bgr, is_pred=False)
            return
        ts_ms = float(meta["ts_ms"])
        speed = float(meta.get("speed", 0.0))

        if self.last_real is not None:
            last_ts, last_speed = self.last_real
            low_fps = self.input_fps is not None and self.input_fps < self.low_fps_threshold
            gap = ts_ms - last_ts
            if low_fps:
                target_ratio = max(1.0, (self.args.fps / max(self.input_fps, 1.0)))
                missing = max(0, int(round(target_ratio)) - 1)
                missing = min(self.horizon, missing)
            else:
                missing = 0
                if gap > self.gap_threshold * self.interval_ms:
                    missing = int(round(gap / self.interval_ms)) - 1
            if missing > 0:
                for k in range(1, missing + 1):
                    ts_m = last_ts + k * self.interval_ms
                    if gap > 0:
                        alpha = (ts_m - last_ts) / gap
                    else:
                        alpha = 0.0
                    speed_m = last_speed + (speed - last_speed) * alpha
                    pred = self.get_predicted(ts_m, speed_m)
                    if pred is not None:
                        print(f"[receiver] insert pred ts_ms={ts_m:.0f} speed={speed_m:.2f}")
                        self.show_frame(pred, is_pred=True)

        self.show_frame(bgr, is_pred=False)

        self.last_real = (ts_ms, speed)
        self.add_to_window(bgr, ts_ms, speed)

    def add_to_window(self, bgr, ts_ms, speed):
        if self.raw_only or self.model is None:
            return
        self.window.append((bgr, ts_ms, speed))
        if len(self.window) == 2 and not self.window_created:
            frame0, ts0, speed0 = self.window[0]
            frame1, ts1, speed1 = self.window[1]
            try:
                self.raft_queue.put_nowait((frame0, frame1, ts0, ts1, speed1))
            except queue.Full:
                try:
                    self.raft_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.raft_queue.put_nowait((frame0, frame1, ts0, ts1, speed1))
                except queue.Full:
                    pass
            self.window_created = True
        if len(self.window) >= self.window_size:
            self.window = []
            self.window_created = False

    def _raft_loop(self):
        while not self._stop_event.is_set():
            try:
                frame0, frame1, ts0, ts1, speed1 = self.raft_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if not isinstance(frame0, np.ndarray) or not isinstance(frame1, np.ndarray):
                    continue
                base_frame = tensor_from_bgr(frame1, self.device)
                prev_frame = tensor_from_bgr(frame0, self.device)
                with torch.no_grad():
                    flow_forward = self.calc_flow_forward(base_frame, prev_frame)
                base_interval = float(ts1 - ts0)
                if base_interval <= 0:
                    base_interval = self.interval_ms
                seq = PredictionSequence(
                    base_ts=ts1,
                    base_speed=speed1,
                    base_frame=base_frame,
                    flow_forward=flow_forward,
                    interval_ms=self.interval_ms,
                    base_interval_ms=max(1.0, base_interval),
                    horizon=self.horizon,
                    eps=self.speed_eps,
                    use_speed_scale=self.use_speed_scale,
                )
                with self.seq_lock:
                    self.sequences.append(seq)
            finally:
                self.raft_queue.task_done()

    def calc_flow_forward(self, frame_curr, frame_prev):
        if self.model is None:
            raise RuntimeError("RAFT is not initialized")
        img1 = F.interpolate(frame_curr, scale_factor=2 ** self.args.scale, mode="bilinear", align_corners=False)
        img2 = F.interpolate(frame_prev, scale_factor=2 ** self.args.scale, mode="bilinear", align_corners=False)
        output = self.model(img1, img2, iters=self.args.iters, test_mode=True)
        flow = output["flow"][-1]
        flow = F.interpolate(flow, scale_factor=0.5 ** self.args.scale, mode="bilinear", align_corners=False)
        flow = flow * (0.5 ** self.args.scale)
        return flow

    def get_predicted(self, ts_ms, speed_target):
        with self.seq_lock:
            for seq in reversed(self.sequences):
                if seq.in_range(ts_ms):
                    pred = seq.make(ts_ms, speed_target)
                    return tensor_to_bgr(pred)
        return None

    def show_frame(self, bgr, is_pred):
        if not self.display:
            return
        if is_pred:
            cv2.putText(bgr, "PRED", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        try:
            self.display_queue.put_nowait(bgr)
        except queue.Full:
            try:
                self.display_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.display_queue.put_nowait(bgr)
            except queue.Full:
                pass


pcs = set()


async def display_loop(app):
    state = app["state"]
    if not state.display:
        return
    cv2.namedWindow("webrtc_realtime", cv2.WINDOW_NORMAL)
    placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, "waiting for frames...", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    last_frame = placeholder
    cv2.imshow("webrtc_realtime", last_frame)
    cv2.waitKey(1)
    while True:
        if state._stop_event.is_set():
            break
        try:
            frame = state.display_queue.get_nowait()
        except queue.Empty:
            frame = None
        if frame is not None:
            last_frame = frame
        cv2.imshow("webrtc_realtime", last_frame)
        cv2.waitKey(1)
        await asyncio.sleep(state.display_interval)


async def on_startup(app):
    app["display_task"] = asyncio.create_task(display_loop(app))


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
    pcs.add(pc)
    state = request.app["state"]

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            asyncio.ensure_future(consume_video(track, state))

    @pc.on("datachannel")
    def on_datachannel(channel):
        if channel.label != "telemetry":
            return

        @channel.on("message")
        def on_message(message):
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                return
            if "frame_id" not in payload or "ts_ms" not in payload:
                return
            state.on_meta(payload)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )


async def consume_video(track, state):
    while True:
        frame = await track.recv()
        state.on_video(frame)


async def on_shutdown(app):
    state = app.get("state")
    if state is not None:
        state.stop()
    task = app.get("display_task")
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    coros = [pc.close() for pc in pcs]
    if coros:
        await asyncio.gather(*coros)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no_display", action="store_true")
    parser.add_argument("--cfg", required=True, type=str)
    parser.add_argument("--path", required=True, type=str)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--gap_threshold", type=float, default=1.5)
    parser.add_argument("--low_fps_threshold", type=float, default=20.0)
    parser.add_argument("--speed_eps", type=float, default=1e-3)
    parser.add_argument("--speed_scale", action="store_true")
    parser.add_argument("--meta_timeout", type=float, default=2.0)
    parser.add_argument("--queue_size", type=int, default=120)
    parser.add_argument("--raw_only", action="store_true")
    args = parse_args(parser)

    app = web.Application()
    app["state"] = ReceiverState(args)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    app.router.add_post("/offer", offer)

    web.run_app(app, host=args.listen, port=args.port)


if __name__ == "__main__":
    main()
