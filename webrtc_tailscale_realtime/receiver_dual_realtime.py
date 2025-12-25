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
sys.path.append(os.path.join(ROOT_DIR, "core"))

from config.parser import parse_args
from raft import RAFT
from utils.utils import load_ckpt, coords_grid, bilinear_sampler


# ================== GUI Tunables (match car_gui_sender) ==================
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_FPS = 60

SIDE_MARGIN_W = 560
MARGIN_COLOR = (0, 0, 0)

CENTER_BRIGHT_ALPHA = 1.18
CENTER_BRIGHT_BETA = 12

MIRROR_ROI_W_RATIO = 0.15
MIRROR_ROI_VRATIO = 0.40
MIRROR_ROI_VCENTER = 0.70

THUMB_H_FRAC = 0.95
THUMB_MARGIN = 14

LOGO_SKY_CANDS = ["sky_logo.png", "sky_logo.jpg", "sky.png", "sky.jpg"]
LOGO_UW_CANDS = [
    "uw_logo.png",
    "uw_logo.jpg",
    "uw.png",
    "uw.jpg",
    "wisconsin.png",
    "wisconsin.jpg",
]

BG_BGR = (0, 0, 0)
FG_BGR = (255, 255, 255)
ACCENT_BGR = (255, 255, 255)
BALL_RED = (0, 0, 255)

BTN_COLORS = {
    "A": (0, 200, 0),
    "B": (0, 0, 255),
    "Y": (0, 200, 255),
    "X": (255, 128, 0),
}
BTN_LETTER_COLOR = (255, 255, 255)

MS_TO_MPH = 2.23694
GEAR_MAP = {0: "NONE", 1: "P", 2: "R", 3: "N", 4: "D", 5: "LOW", 15: "CALIBRATE"}


class ThreadSafeValue:
    def __init__(self, init=None):
        self._lock = threading.Lock()
        self._val = init

    def set(self, v):
        with self._lock:
            self._val = v

    def get(self):
        with self._lock:
            return self._val


speed_mps_cache = ThreadSafeValue(0.0)
gear_value_cache = ThreadSafeValue(0)
steer_deg_cache = ThreadSafeValue(0.0)


def put_text(img, text, org, scale, color, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


STEER_MAX_ABS_DEG = 600.0
STEER_BAR_W = 360
STEER_BAR_H = 12
STEER_BALL_R = 8


def _draw_steer_readout_and_bar(canvas, x, y):
    ang = float(steer_deg_cache.get() or 0.0)
    put_text(canvas, f"Steering: {ang:.1f} deg", (x, y), 0.95, FG_BGR, 2)
    y += 26

    bar_x, bar_y = x, y
    bar_w, bar_h = STEER_BAR_W, STEER_BAR_H
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (0, 0, 0), -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), FG_BGR, 2, cv2.LINE_AA)
    cx_mid = bar_x + bar_w // 2
    cv2.line(canvas, (cx_mid, bar_y - 6), (cx_mid, bar_y + bar_h + 6), FG_BGR, 1, cv2.LINE_AA)

    norm = max(-1.0, min(1.0, -ang / STEER_MAX_ABS_DEG))
    margin = 6
    movable = (bar_w // 2 - margin)
    cx = int(cx_mid + norm * movable)
    cy = bar_y + bar_h // 2
    cv2.circle(canvas, (cx, cy), STEER_BALL_R, BALL_RED, -1, cv2.LINE_AA)
    return y + bar_h + 20


def draw_left_hud_overlay(canvas):
    x = 18
    y = 34
    line = 36

    put_text(canvas, "Velocity", (x, y), 0.8, ACCENT_BGR, 2)
    y += int(line * 1.0)

    v_mps = float(speed_mps_cache.get() or 0.0)
    v_mph = v_mps * MS_TO_MPH
    put_text(canvas, f"{v_mps:.2f} m/s", (x, y + 20), 1.4, FG_BGR, 3)
    y += int(line * 1.5)
    put_text(canvas, f"{v_mph:.2f} mph", (x, y + 18), 1.0, FG_BGR, 2)
    y += int(line * 1.3)

    gear_val = int(gear_value_cache.get() or 0)
    gear_txt = GEAR_MAP.get(gear_val, str(gear_val))
    put_text(canvas, f"Gear: {gear_txt}", (x, y + 40), 2.0, FG_BGR, 4)
    y += int(line * 2.0)

    hints = [("A", "Drive"), ("B", "Reverse"), ("X", "Park"), ("Y", "Neutral")]
    spacing = 50
    for i, (btn, label) in enumerate(hints):
        cy = y + i * spacing
        cv2.circle(canvas, (x + 22, cy - 10), 18, BTN_COLORS[btn], -1, cv2.LINE_AA)
        put_text(canvas, btn, (x + 14, cy - 4), 0.70, BTN_LETTER_COLOR, 2)
        put_text(canvas, f"= {label}", (x + 56, cy + 4), 0.95, FG_BGR, 2)
    y += spacing * len(hints) + 16

    _draw_steer_readout_and_bar(canvas, x, y)


def crop_mirror_roi(bottom_frame, side="left"):
    h, w = bottom_frame.shape[:2]
    crop_h = int(h * MIRROR_ROI_VRATIO)
    crop_h = max(1, min(h, crop_h))
    cy = int(h * MIRROR_ROI_VCENTER)
    top = max(0, min(h - crop_h, cy - crop_h // 2))
    bot = top + crop_h
    cw = int(w * MIRROR_ROI_W_RATIO)
    cw = max(1, min(w, cw))
    if side == "left":
        x0, x1 = 0, cw
    else:
        x0, x1 = w - cw, w
    return bottom_frame[top:bot, x0:x1]


def make_panel_cover(img, target_w, target_h):
    if img is None or img.size == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    ih, iw = img.shape[:2]
    scale = max(target_w / float(iw), target_h / float(ih))
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh))
    x0 = max(0, (nw - target_w) // 2)
    y0 = max(0, (nh - target_h) // 2)
    crop = resized[y0 : y0 + target_h, x0 : x0 + target_w].copy()
    cv2.rectangle(crop, (0, 0), (target_w - 1, target_h - 1), (255, 255, 255), 3, cv2.LINE_AA)
    return crop


def paste(dst, src, x, y):
    H, W = dst.shape[:2]
    h, w = src.shape[:2]
    if x >= W or y >= H:
        return
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    cx = max(0, -x)
    cy = max(0, -y)
    dst_y1, dst_y2 = max(0, y), y2
    dst_x1, dst_x2 = max(0, x), x2
    src_y1, src_y2 = cy, cy + (dst_y2 - dst_y1)
    src_x1, src_x2 = cx, cx + (dst_x2 - dst_x1)
    if dst_y1 < dst_y2 and dst_x1 < dst_x2:
        dst[dst_y1:dst_y2, dst_x1:dst_x2] = src[src_y1:src_y2, src_x1:src_x2]


def resolve_logo(candidates):
    from pathlib import Path

    script_dir = Path(__file__).resolve().parent
    search_dirs = [script_dir, Path.cwd()]
    for name in candidates:
        for base in search_dirs:
            p = (base / name).expanduser().resolve()
            if p.exists():
                return str(p)
    return None


def load_logo_auto(candidates, max_w, max_h):
    path = resolve_logo(candidates)
    if path is None:
        print(f"[Logo] Not found: {candidates}")
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[Logo] Failed to read: {path}")
        return None
    ih, iw = img.shape[:2]
    scale = min(max_w / float(iw), max_h / float(ih))
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    img = cv2.resize(img, (nw, nh))
    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3:4] / 255.0
        black = np.zeros_like(bgr)
        img = (bgr * alpha + black * (1 - alpha)).astype(np.uint8)
    print(f"[Logo] Loaded: {path}")
    return img


BTN_SIZE = (140, 46)
BTN_LABEL = "Swap"
BTN_TEXT = (30, 30, 30)
BTN_GRAD_TOP = (245, 245, 245)
BTN_GRAD_BOTTOM = (190, 190, 190)
BTN_BORDER = (80, 80, 80)
BTN_RADIUS = 12

SWAP_FLAG = {"on": False}
BTN_RECT = None
WINDOW_NAME = "Multi-View (Mirrors Focus)"


def _rounded_mask(w, h, r):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1, cv2.LINE_AA)
    cv2.rectangle(mask, (0, r), (w, h - r), 255, -1, cv2.LINE_AA)
    cv2.circle(mask, (r, r), r, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, (w - r - 1, r), r, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, (r, h - r - 1), r, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, (w - r - 1, h - r - 1), r, 255, -1, cv2.LINE_AA)
    return mask


def _draw_swap_icon(dst, cx, cy, active=False):
    radius = 10
    color_fill = (230, 230, 230) if active else (210, 210, 210)
    cv2.circle(dst, (cx, cy), radius + 6, color_fill, -1, cv2.LINE_AA)
    pts_up = np.array([[cx, cy - 8], [cx - 8, cy], [cx + 8, cy]], np.int32)
    cv2.polylines(dst, [pts_up], True, (60, 60, 60), 2, cv2.LINE_AA)
    pts_dn = np.array([[cx, cy + 8], [cx - 8, cy], [cx + 8, cy]], np.int32)
    cv2.polylines(dst, [pts_dn], True, (60, 60, 60), 2, cv2.LINE_AA)


def draw_swap_button(canvas, sidebar_right_x, y_start):
    global BTN_RECT
    bw, bh = BTN_SIZE
    bx = sidebar_right_x + THUMB_MARGIN
    by = y_start

    H, W = canvas.shape[:2]
    x2, y2 = min(W, bx + bw), min(H, by + bh)
    if x2 <= bx or y2 <= by:
        BTN_RECT = None
        return by
    roi = canvas[by:y2, bx:x2].copy()

    for i in range(roi.shape[0]):
        t = i / max(1, roi.shape[0] - 1)
        r = int(BTN_GRAD_TOP[0] * (1 - t) + BTN_GRAD_BOTTOM[0] * t)
        g = int(BTN_GRAD_TOP[1] * (1 - t) + BTN_GRAD_BOTTOM[1] * t)
        b = int(BTN_GRAD_TOP[2] * (1 - t) + BTN_GRAD_BOTTOM[2] * t)
        roi[i, :, :] = (r, g, b)

    mask = _rounded_mask(roi.shape[1], roi.shape[0], BTN_RADIUS)
    inv_mask = cv2.bitwise_not(mask)
    base_roi = canvas[by:y2, bx:x2]
    blended = cv2.bitwise_and(roi, roi, mask=mask) + cv2.bitwise_and(base_roi, base_roi, mask=inv_mask)
    canvas[by:y2, bx:x2] = blended

    cv2.rectangle(canvas, (bx + BTN_RADIUS, by), (bx + bw - BTN_RADIUS - 1, by + bh - 1), BTN_BORDER, 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (bx, by + BTN_RADIUS), (bx + bw - 1, by + bh - BTN_RADIUS - 1), BTN_BORDER, 2, cv2.LINE_AA)
    cv2.ellipse(canvas, (bx + BTN_RADIUS, by + BTN_RADIUS), (BTN_RADIUS, BTN_RADIUS), 180, 0, 90, BTN_BORDER, 2)
    cv2.ellipse(
        canvas, (bx + bw - BTN_RADIUS - 1, by + BTN_RADIUS), (BTN_RADIUS, BTN_RADIUS), 270, 0, 90, BTN_BORDER, 2
    )
    cv2.ellipse(
        canvas, (bx + BTN_RADIUS, by + bh - BTN_RADIUS - 1), (BTN_RADIUS, BTN_RADIUS), 90, 0, 90, BTN_BORDER, 2
    )
    cv2.ellipse(
        canvas,
        (bx + bw - BTN_RADIUS - 1, by + bh - BTN_RADIUS - 1),
        (BTN_RADIUS, BTN_RADIUS),
        0,
        0,
        90,
        BTN_BORDER,
        2,
    )

    icon_cx = bx + 20
    icon_cy = by + bh // 2
    _draw_swap_icon(canvas, icon_cx, icon_cy, active=SWAP_FLAG["on"])

    label = f"{BTN_LABEL}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)
    tx = bx + 40
    ty = by + (bh + th) // 2 - 1
    cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.70, BTN_TEXT, 2, cv2.LINE_AA)

    BTN_RECT = (bx, by, bw, bh)
    return by + bh


def swap_on_mouse(event, x, y, flags, param):
    global BTN_RECT
    if event == cv2.EVENT_LBUTTONDOWN and BTN_RECT is not None:
        bx, by, bw, bh = BTN_RECT
        if bx <= x <= bx + bw and by <= y <= by + bh:
            SWAP_FLAG["on"] = not SWAP_FLAG["on"]


def maybe_swap_frames(frames):
    if SWAP_FLAG["on"] and len(frames) >= 2:
        return frames[::-1]
    return frames


WINE_BORDER = (32, 0, 128)


def draw_rounded_panel(canvas, x, y, w, h, radius=16, border=WINE_BORDER, fill=(255, 255, 255), border_thick=2):
    H, W = canvas.shape[:2]
    if w <= 0 or h <= 0 or x >= W or y >= H:
        return
    x2, y2 = min(W, x + w), min(H, y + h)
    roi = canvas[y:y2, x:x2]
    mask = _rounded_mask(roi.shape[1], roi.shape[0], radius)
    inv_mask = cv2.bitwise_not(mask)
    fill_img = np.full_like(roi, fill, dtype=np.uint8)
    blended = cv2.bitwise_and(fill_img, fill_img, mask=mask) + cv2.bitwise_and(roi, roi, mask=inv_mask)
    canvas[y:y2, x:x2] = blended
    cv2.line(canvas, (x + radius, y), (x + w - radius - 1, y), border, border_thick, cv2.LINE_AA)
    cv2.line(canvas, (x + radius, y + h - 1), (x + w - radius - 1, y + h - 1), border, border_thick, cv2.LINE_AA)
    cv2.line(canvas, (x, y + radius), (x, y + h - radius - 1), border, border_thick, cv2.LINE_AA)
    cv2.line(canvas, (x + w - 1, y + radius), (x + w - 1, y + h - radius - 1), border, border_thick, cv2.LINE_AA)
    cv2.ellipse(canvas, (x + radius, y + radius), (radius, radius), 180, 0, 90, border, border_thick)
    cv2.ellipse(canvas, (x + w - radius - 1, y + radius), (radius, radius), 270, 0, 90, border, border_thick)
    cv2.ellipse(canvas, (x + radius, y + h - radius - 1), (radius, radius), 90, 0, 90, border, border_thick)
    cv2.ellipse(
        canvas, (x + w - radius - 1, y + h - radius - 1), (radius, radius), 0, 0, 90, border, border_thick
    )


def resize_by_height(frame, target_h=TARGET_HEIGHT):
    h, w = frame.shape[:2]
    scale = float(target_h) / float(h)
    return cv2.resize(frame, (int(w * scale), target_h))


def pad_to_same_width(frames, pad_value=0):
    max_w = max(f.shape[1] for f in frames)
    out = []
    for f in frames:
        h, w = f.shape[:2]
        if w < max_w:
            pad = np.full((h, max_w - w, 3), pad_value, dtype=f.dtype)
            f = np.hstack([f, pad])
        out.append(f)
    return out


def annotate_frame(frame, label, fps_value):
    out = frame.copy()
    if fps_value is None:
        text = f"{label}"
    else:
        text = f"{label}  {fps_value:.0f}fps"
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ================== Optical Flow Utils ==================
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


class StreamState:
    def __init__(self, name, receiver):
        self.name = name
        self.receiver = receiver
        self.meta_queue = deque(maxlen=receiver.queue_size)
        self.frame_queue = deque(maxlen=receiver.queue_size)
        self.pair_queue = queue.Queue(maxsize=receiver.queue_size)
        self.window = []
        self.window_created = False
        self.sequences = deque(maxlen=10)
        self.seq_lock = threading.Lock()
        self.last_real = None
        self.have_meta = False
        self.first_video_wall = None
        self._printed_no_meta = False
        self._fps_count = 0
        self._fps_last_wall = time.time()
        self.input_fps = None
        self.display_queue = queue.Queue(maxsize=receiver.queue_size)
        self.last_frame = None
        self._worker = threading.Thread(target=self._process_loop, daemon=True)
        self._worker.start()

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
            if self.receiver.debug:
                print(f"[receiver] telemetry connected: {self.name}")
        self.meta_queue.append(payload)
        self._drain_pairs()

    def on_video(self, frame):
        try:
            bgr = frame.to_ndarray(format="bgr24")
        except Exception:
            return
        if not isinstance(bgr, np.ndarray) or bgr.ndim != 3:
            return

        self._fps_count += 1
        now = time.time()
        if now - self._fps_last_wall >= 1.0:
            self.input_fps = float(self._fps_count)
            if self.receiver.debug:
                print(f"[receiver] {self.name} fps ~ {self._fps_count}")
            self._fps_count = 0
            self._fps_last_wall = now

        if not self.have_meta:
            if self.first_video_wall is None:
                self.first_video_wall = now
            if (now - self.first_video_wall) >= self.receiver.meta_timeout:
                if not self._printed_no_meta:
                    print(f"[receiver] {self.name} no telemetry yet, showing video only")
                    self._printed_no_meta = True
                self.show_frame(bgr, is_pred=False)
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
        while not self.receiver.stop_event.is_set():
            try:
                frame, meta = self.pair_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.process_real(frame, meta)
            finally:
                self.pair_queue.task_done()

    def add_sequence(self, seq):
        with self.seq_lock:
            self.sequences.append(seq)

    def get_predicted(self, ts_ms, speed_target):
        with self.seq_lock:
            for seq in reversed(self.sequences):
                if seq.in_range(ts_ms):
                    pred = seq.make(ts_ms, speed_target)
                    return tensor_to_bgr(pred)
        return None

    def show_frame(self, bgr, is_pred):
        frame = bgr.copy()
        if is_pred:
            cv2.putText(frame, "PRED", (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        self._push_display(frame)

    def _push_display(self, frame):
        self.last_frame = frame
        try:
            self.display_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.display_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.display_queue.put_nowait(frame)
            except queue.Full:
                pass

    def get_display_frame(self):
        try:
            return self.display_queue.get_nowait()
        except queue.Empty:
            return self.last_frame

    def process_real(self, bgr, meta):
        ts_ms = float(meta["ts_ms"])
        speed = float(meta.get("speed", 0.0))

        if self.last_real is not None:
            last_ts, last_speed = self.last_real
            low_fps = self.input_fps is not None and self.input_fps < self.receiver.low_fps_threshold
            gap = ts_ms - last_ts
            if low_fps:
                target_ratio = max(1.0, (self.receiver.args.fps / max(self.input_fps, 1.0)))
                missing = max(0, int(round(target_ratio)) - 1)
                missing = min(self.receiver.horizon, missing)
            else:
                missing = 0
                if gap > self.receiver.gap_threshold * self.receiver.interval_ms:
                    missing = int(round(gap / self.receiver.interval_ms)) - 1

            if missing > 0:
                for k in range(1, missing + 1):
                    ts_m = last_ts + k * self.receiver.interval_ms
                    if gap > 0:
                        alpha = (ts_m - last_ts) / gap
                    else:
                        alpha = 0.0
                    speed_m = last_speed + (speed - last_speed) * alpha
                    pred = self.get_predicted(ts_m, speed_m)
                    if pred is not None:
                        print(f"[receiver] insert pred {self.name} ts_ms={ts_m:.0f} speed={speed_m:.2f}")
                        self.show_frame(pred, is_pred=True)

        self.show_frame(bgr, is_pred=False)
        self.last_real = (ts_ms, speed)
        self.add_to_window(bgr, ts_ms, speed)

    def add_to_window(self, bgr, ts_ms, speed):
        if self.receiver.raw_only or self.receiver.model is None:
            return
        self.window.append((bgr, ts_ms, speed))
        if len(self.window) == 2 and not self.window_created:
            frame0, ts0, speed0 = self.window[0]
            frame1, ts1, speed1 = self.window[1]
            self.receiver.enqueue_raft(self.name, frame0, frame1, ts0, ts1, speed1)
            self.window_created = True
        if len(self.window) >= self.receiver.window_size:
            self.window = []
            self.window_created = False


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
        self.queue_size = args.queue_size
        self.raw_only = args.raw_only
        self.debug = args.debug
        self.stop_event = threading.Event()

        self.streams = {
            "cam0": StreamState("cam0", self),
            "cam1": StreamState("cam1", self),
        }

        self.raft_queue = queue.Queue(maxsize=args.queue_size)
        self._raft_worker = None
        if not args.raw_only:
            self._raft_worker = threading.Thread(target=self._raft_loop, daemon=True)
            self._raft_worker.start()

    def enqueue_raft(self, stream_name, frame0, frame1, ts0, ts1, speed1):
        try:
            self.raft_queue.put_nowait((stream_name, frame0, frame1, ts0, ts1, speed1))
        except queue.Full:
            try:
                self.raft_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.raft_queue.put_nowait((stream_name, frame0, frame1, ts0, ts1, speed1))
            except queue.Full:
                pass

    def _raft_loop(self):
        while not self.stop_event.is_set():
            try:
                stream_name, frame0, frame1, ts0, ts1, speed1 = self.raft_queue.get(timeout=0.1)
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
                self.streams[stream_name].add_sequence(seq)
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

    def stop(self):
        self.stop_event.set()
        if self._raft_worker is not None and self._raft_worker.is_alive():
            self._raft_worker.join(timeout=2.0)


pcs = set()


async def display_loop(app):
    state = app["state"]
    if state.args.no_display:
        return

    logo_uw = load_logo_auto(LOGO_UW_CANDS, SIDE_MARGIN_W - 2 * THUMB_MARGIN, 240)
    logo_sky = load_logo_auto(LOGO_SKY_CANDS, SIDE_MARGIN_W - 2 * THUMB_MARGIN, 120)
    if logo_uw is not None:
        h, w = logo_uw.shape[:2]
        logo_uw = cv2.resize(logo_uw, (max(1, int(w * 0.9)), max(1, int(h * 0.9))))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, swap_on_mouse)

    placeholder = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
    cv2.putText(placeholder, "waiting for frames...", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow(WINDOW_NAME, placeholder)
    cv2.waitKey(1)

    while True:
        if state.stop_event.is_set():
            break

        frame0 = state.streams["cam0"].last_frame
        frame1 = state.streams["cam1"].last_frame
        if frame0 is None:
            frame0 = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
        if frame1 is None:
            frame1 = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)

        frame0 = annotate_frame(frame0, "cam0", state.streams["cam0"].input_fps)
        frame1 = annotate_frame(frame1, "cam1", state.streams["cam1"].input_fps)
        frames_gui = [resize_by_height(frame0, TARGET_HEIGHT), resize_by_height(frame1, TARGET_HEIGHT)]
        frames_gui = maybe_swap_frames(frames_gui)
        frames_gui = pad_to_same_width(frames_gui, pad_value=0)
        video_stack_full = frames_gui[0] if len(frames_gui) == 1 else cv2.vconcat(frames_gui)

        H, W_full = video_stack_full.shape[:2]
        center_col = cv2.convertScaleAbs(video_stack_full, alpha=CENTER_BRIGHT_ALPHA, beta=CENTER_BRIGHT_BETA)

        left_margin = np.full((H, SIDE_MARGIN_W, 3), MARGIN_COLOR, dtype=center_col.dtype)
        right_margin = np.full((H, SIDE_MARGIN_W, 3), MARGIN_COLOR, dtype=center_col.dtype)
        canvas = np.hstack([left_margin, center_col, right_margin])

        draw_left_hud_overlay(canvas)

        right_sidebar_x0 = canvas.shape[1] - SIDE_MARGIN_W
        y_right = THUMB_MARGIN
        y_right = draw_swap_button(canvas, right_sidebar_x0, y_right)
        y_right += 12

        panel_x = right_sidebar_x0 + THUMB_MARGIN
        panel_w = SIDE_MARGIN_W - 2 * THUMB_MARGIN
        pad_y = 10
        pad_between = 20

        h_uw = logo_uw.shape[0] if logo_uw is not None else 0
        h_sky = logo_sky.shape[0] if logo_sky is not None else 0
        logos_h = h_uw + (pad_between if (logo_uw is not None and logo_sky is not None) else 0) + h_sky
        panel_h = max(2 * pad_y + logos_h, 2 * pad_y + 40)

        panel_y = y_right
        draw_rounded_panel(canvas, panel_x, panel_y, panel_w, panel_h, radius=16, border=WINE_BORDER, fill=(255, 255, 255))

        cur_y = panel_y + pad_y
        if logo_uw is not None:
            x_uw = panel_x + (panel_w - logo_uw.shape[1]) // 2
            paste(canvas, logo_uw, x_uw, cur_y)
            cur_y += h_uw + (pad_between if logo_sky is not None else 0)
        if logo_sky is not None:
            x_sky = panel_x + (panel_w - logo_sky.shape[1]) // 2
            paste(canvas, logo_sky, x_sky, cur_y)

        y_right = panel_y + panel_h + 12

        bottom_frame = frames_gui[-1]
        roi_left = crop_mirror_roi(bottom_frame, "left")
        roi_right = crop_mirror_roi(bottom_frame, "right")

        thumb_w = SIDE_MARGIN_W - 2 * THUMB_MARGIN
        thumb_h = int(TARGET_HEIGHT * THUMB_H_FRAC)
        thumb_left = make_panel_cover(roi_left, thumb_w, thumb_h)
        thumb_right = make_panel_cover(roi_right, thumb_w, thumb_h)

        N = len(frames_gui)
        y_base = (N - 1) * TARGET_HEIGHT + THUMB_MARGIN
        paste(canvas, thumb_left, THUMB_MARGIN, y_base)
        paste(canvas, thumb_right, canvas.shape[1] - SIDE_MARGIN_W + THUMB_MARGIN, y_base)

        cv2.putText(canvas, "Left Mirror", (THUMB_MARGIN + 4, y_base - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "Right Mirror",
            (canvas.shape[1] - SIDE_MARGIN_W + THUMB_MARGIN + 4, y_base - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        await asyncio.sleep(state.display_interval)


async def on_startup(app):
    app["display_task"] = asyncio.create_task(display_loop(app))


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
    pcs.add(pc)
    state = request.app["state"]
    track_map = {"video": []}

    @pc.on("track")
    def on_track(track):
        if track.kind != "video":
            return
        index = len(track_map["video"])
        track_map["video"].append(track)
        stream_name = "cam0" if index == 0 else "cam1"
        asyncio.ensure_future(consume_video(track, state, stream_name))

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
            stream_id = payload.get("stream_id")
            if stream_id not in state.streams:
                return
            speed_mps_cache.set(float(payload.get("speed", 0.0)))
            steer_deg_cache.set(float(payload.get("steer", 0.0)))
            gear_value_cache.set(int(payload.get("gear", 0)))
            state.streams[stream_id].on_meta(payload)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})


async def consume_video(track, state, stream_name):
    while True:
        frame = await track.recv()
        state.streams[stream_name].on_video(frame)


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
    parser.add_argument("--debug", action="store_true")
    args = parse_args(parser)

    app = web.Application()
    app["state"] = ReceiverState(args)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    app.router.add_post("/offer", offer)

    web.run_app(app, host=args.listen, port=args.port)


if __name__ == "__main__":
    main()
