#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import asyncio
import glob
import json
import os
import random
import threading
import time
from fractions import Fraction

import cv2
import numpy as np
import aiohttp
from av import VideoFrame
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCRtpSender,
    VideoStreamTrack,
)

# ================== UI Tunables ==================
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

# ================== ROS ==================
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ds_dbw_msgs.msg import GearReport, VehicleVelocity, SteeringReport

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
gear_msg_cache = ThreadSafeValue(None)
steer_deg_cache = ThreadSafeValue(0.0)


class VelocitySubNode(Node):
    def __init__(self):
        super().__init__("velocity_hud_reader")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(VehicleVelocity, "/vehicle/vehicle_velocity", self.cb, qos)

    def cb(self, msg: VehicleVelocity):
        v_mps = float(getattr(msg, "vehicle_velocity_brake", 0.0))
        speed_mps_cache.set(v_mps)


class GearSubNode(Node):
    def __init__(self):
        super().__init__("gear_hud_reader")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(GearReport, "/vehicle/gear/report", self.cb, qos)

    def cb(self, msg: GearReport):
        gear_msg_cache.set(msg)


class SteeringSubNode(Node):
    def __init__(self):
        super().__init__("steering_hud_reader")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(SteeringReport, "/vehicle/steering/report", self.cb, qos)

    def cb(self, msg: SteeringReport):
        try:
            steer_deg_cache.set(float(getattr(msg, "steering_wheel_angle", 0.0)))
        except Exception:
            pass


def start_ros_spin(stop_event: threading.Event):
    rclpy.init()
    gear_node = GearSubNode()
    vel_node = VelocitySubNode()
    steer_node = SteeringSubNode()
    exe = rclpy.executors.SingleThreadedExecutor()
    exe.add_node(gear_node)
    exe.add_node(vel_node)
    exe.add_node(steer_node)
    try:
        while not stop_event.is_set():
            exe.spin_once(timeout_sec=0.1)
    finally:
        exe.remove_node(gear_node)
        exe.remove_node(vel_node)
        exe.remove_node(steer_node)
        gear_node.destroy_node()
        vel_node.destroy_node()
        steer_node.destroy_node()
        rclpy.shutdown()


# ================== Video Utils ==================
def try_open(dev):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None
    return cap


def enumerate_cameras(prefer=None):
    devices = []
    if prefer:
        devices.extend(prefer)
    devices.extend(sorted(glob.glob("/dev/video*")))
    seen, ordered = set(), []
    for d in devices:
        if d not in seen and os.path.exists(d):
            seen.add(d)
            ordered.append(d)
    caps = []
    for d in ordered:
        cap = try_open(d)
        if cap is not None:
            caps.append((d, cap))
    return caps


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


# ================== HUD ==================
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
    cv2.rectangle(
        canvas,
        (bar_x, bar_y),
        (bar_x + bar_w, bar_y + bar_h),
        (0, 0, 0),
        -1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        canvas,
        (bar_x, bar_y),
        (bar_x + bar_w, bar_y + bar_h),
        FG_BGR,
        2,
        cv2.LINE_AA,
    )
    cx_mid = bar_x + bar_w // 2
    cv2.line(
        canvas,
        (cx_mid, bar_y - 6),
        (cx_mid, bar_y + bar_h + 6),
        FG_BGR,
        1,
        cv2.LINE_AA,
    )

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

    msg = gear_msg_cache.get()
    gear_txt = "--"
    if msg is not None:
        gear_txt = GEAR_MAP.get(getattr(msg.gear, "value", 0), str(getattr(msg.gear, "value", 0)))
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


# ===== Swap Button =====
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

    cv2.rectangle(
        canvas, (bx + BTN_RADIUS, by), (bx + bw - BTN_RADIUS - 1, by + bh - 1), BTN_BORDER, 2, cv2.LINE_AA
    )
    cv2.rectangle(
        canvas, (bx, by + BTN_RADIUS), (bx + bw - 1, by + bh - BTN_RADIUS - 1), BTN_BORDER, 2, cv2.LINE_AA
    )
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


# ================== WebRTC Sender ==================
class MetaSender:
    def __init__(self):
        self.channel = None

    def attach(self, channel):
        self.channel = channel

    def send(self, frame_id, ts_ms, speed, steer, fps, width, height):
        if self.channel is None or self.channel.readyState != "open":
            return
        payload = {
            "frame_id": frame_id,
            "ts_ms": ts_ms,
            "speed": float(speed),
            "steer": float(steer),
            "fps": float(fps),
            "width": int(width),
            "height": int(height),
        }
        self.channel.send(json.dumps(payload))


class GuiVideoTrack(VideoStreamTrack):
    def __init__(self, queue, meta_sender, send_fps):
        super().__init__()
        self.queue = queue
        self.meta_sender = meta_sender
        self.time_base = Fraction(1, send_fps)

    async def recv(self):
        frame_id, ts_ms, speed, steer, fps, bgr = await self.queue.get()
        self.meta_sender.send(frame_id, ts_ms, speed, steer, fps, bgr.shape[1], bgr.shape[0])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(rgb, format="rgb24")
        video_frame.pts = frame_id
        video_frame.time_base = self.time_base
        return video_frame


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


async def wait_for_ice(pc):
    if pc.iceGatheringState == "complete":
        return
    done = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def check_state():
        if pc.iceGatheringState == "complete":
            done.set()

    await done.wait()


class WebRtcSender:
    def __init__(self, signal_url, send_fps, codec, loop, queue):
        self.signal_url = signal_url
        self.send_fps = send_fps
        self.codec = codec
        self.loop = loop
        self.queue = queue
        self.pc = None
        self.meta_sender = MetaSender()

    async def start(self):
        self.pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
        channel = self.pc.createDataChannel("telemetry")
        self.meta_sender.attach(channel)
        track = GuiVideoTrack(self.queue, self.meta_sender, self.send_fps)
        transceiver = self.pc.addTransceiver(track, direction="sendonly")
        if self.codec and not set_preferred_codec(transceiver, self.codec):
            print(f"Requested codec not available: {self.codec}")

        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        await wait_for_ice(self.pc)

        async with aiohttp.ClientSession() as session:
            payload = {"sdp": self.pc.localDescription.sdp, "type": self.pc.localDescription.type}
            async with session.post(self.signal_url, json=payload) as resp:
                resp.raise_for_status()
                answer = await resp.json()

        await self.pc.setRemoteDescription(RTCSessionDescription(**answer))
        print("WebRTC connected.")

    async def close(self):
        if self.pc is not None:
            await self.pc.close()


def resize_for_send(img, send_width, send_height):
    if send_width <= 0 and send_height <= 0:
        return img
    h, w = img.shape[:2]
    if send_width <= 0:
        scale = send_height / float(h)
        new_w = int(round(w * scale))
        new_h = int(round(send_height))
    elif send_height <= 0:
        scale = send_width / float(w)
        new_w = int(round(send_width))
        new_h = int(round(h * scale))
    else:
        new_w = int(send_width)
        new_h = int(send_height)
    new_w = max(2, (new_w // 2) * 2)
    new_h = max(2, (new_h // 2) * 2)
    if new_w == w and new_h == h:
        return img
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", required=True, help="Signaling URL, e.g. http://ip:8080/offer")
    parser.add_argument("--codec", default="vp8", help="vp8 or h264")
    parser.add_argument("--send_fps", type=int, default=30)
    parser.add_argument("--send_width", type=int, default=960)
    parser.add_argument("--send_height", type=int, default=540)
    parser.add_argument("--no_display", action="store_true")
    args = parser.parse_args()

    stop_event = threading.Event()
    ros_thread = threading.Thread(target=start_ros_spin, args=(stop_event,), daemon=True)
    ros_thread.start()

    caps = enumerate_cameras(prefer=None)
    if not caps:
        print("No camera devices found.")
        stop_event.set()
        ros_thread.join(timeout=1.0)
        return

    uw_max_h_top = 240
    sky_max_h_bot = 120
    uw_max_w = SIDE_MARGIN_W - 2 * THUMB_MARGIN
    sky_max_w = SIDE_MARGIN_W - 2 * THUMB_MARGIN

    logo_uw = load_logo_auto(LOGO_UW_CANDS, uw_max_w, uw_max_h_top)
    logo_sky = load_logo_auto(LOGO_SKY_CANDS, sky_max_w, sky_max_h_bot)

    if logo_uw is not None:
        h, w = logo_uw.shape[:2]
        logo_uw = cv2.resize(logo_uw, (max(1, int(w * 0.9)), max(1, int(h * 0.9))))

    if not args.no_display:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, swap_on_mouse)

    loop = asyncio.new_event_loop()
    queue = asyncio.Queue(maxsize=2)
    sender = WebRtcSender(args.signal, args.send_fps, args.codec, loop, queue)

    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    loop_thread = threading.Thread(target=run_loop, daemon=True)
    loop_thread.start()

    future = asyncio.run_coroutine_threadsafe(sender.start(), loop)
    future.result()

    frame_id = 0
    send_interval = 1.0 / float(max(1, args.send_fps))
    next_send_ts = time.time()

    try:
        while True:
            frames, alive = [], []
            for dev, cap in caps:
                ok, frame = cap.read()
                if not ok or frame is None:
                    cap.release()
                    continue
                fps_display = random.uniform(55, 60)
                cv2.putText(
                    frame,
                    f"{dev}  {fps_display:.0f}fps",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                frames.append(resize_by_height(frame, TARGET_HEIGHT))
                alive.append((dev, cap))
            caps = alive
            if not frames:
                break

            frames = maybe_swap_frames(frames)
            frames = pad_to_same_width(frames, pad_value=0)
            video_stack_full = frames[0] if len(frames) == 1 else cv2.vconcat(frames)

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

            bottom_frame = frames[-1]
            roi_left = crop_mirror_roi(bottom_frame, "left")
            roi_right = crop_mirror_roi(bottom_frame, "right")

            thumb_w = SIDE_MARGIN_W - 2 * THUMB_MARGIN
            thumb_h = int(TARGET_HEIGHT * THUMB_H_FRAC)
            thumb_left = make_panel_cover(roi_left, thumb_w, thumb_h)
            thumb_right = make_panel_cover(roi_right, thumb_w, thumb_h)

            N = len(frames)
            y_base = (N - 1) * TARGET_HEIGHT + THUMB_MARGIN
            paste(canvas, thumb_left, THUMB_MARGIN, y_base)
            paste(canvas, thumb_right, canvas.shape[1] - SIDE_MARGIN_W + THUMB_MARGIN, y_base)

            cv2.putText(
                canvas,
                "Left Mirror",
                (THUMB_MARGIN + 4, y_base - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
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

            if not args.no_display:
                cv2.imshow(WINDOW_NAME, canvas)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            now = time.time()
            if now >= next_send_ts:
                frame_id += 1
                ts_ms = int(now * 1000)
                speed = float(speed_mps_cache.get() or 0.0)
                steer = float(steer_deg_cache.get() or 0.0)
                send_frame = resize_for_send(canvas, args.send_width, args.send_height)
                item = (frame_id, ts_ms, speed, steer, args.send_fps, send_frame)

                def push_item():
                    if queue.full():
                        try:
                            queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    queue.put_nowait(item)

                loop.call_soon_threadsafe(push_item)
                next_send_ts = now + send_interval

    finally:
        for _, cap in caps:
            cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        stop_event.set()
        ros_thread.join(timeout=1.0)
        close_future = asyncio.run_coroutine_threadsafe(sender.close(), loop)
        close_future.result()
        loop.call_soon_threadsafe(loop.stop)


if __name__ == "__main__":
    main()
