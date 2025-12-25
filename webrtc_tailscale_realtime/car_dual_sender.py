#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import asyncio
import json
import random
import time
import threading
from fractions import Fraction

import cv2
import numpy as np
import aiohttp
from av import VideoFrame
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCRtpSender, VideoStreamTrack

import car_gui_sender as gui


class MetaSender:
    def __init__(self):
        self.channel = None

    def attach(self, channel):
        self.channel = channel

    def send(self, payload):
        if self.channel is None or self.channel.readyState != "open":
            return
        self.channel.send(payload)


class RawVideoTrack(VideoStreamTrack):
    def __init__(self, queue, meta_sender, send_fps, stream_id):
        super().__init__()
        self.queue = queue
        self.meta_sender = meta_sender
        self.time_base = Fraction(1, send_fps)
        self.stream_id = stream_id

    async def recv(self):
        frame_id, ts_ms, speed, steer, gear, fps, bgr = await self.queue.get()
        payload = {
            "stream_id": self.stream_id,
            "frame_id": int(frame_id),
            "ts_ms": float(ts_ms),
            "speed": float(speed),
            "steer": float(steer),
            "gear": int(gear),
            "fps": float(fps),
            "width": int(bgr.shape[1]),
            "height": int(bgr.shape[0]),
        }
        self.meta_sender.send(json.dumps(payload))
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
    def __init__(self, signal_url, send_fps, codec, queue0, queue1):
        self.signal_url = signal_url
        self.send_fps = send_fps
        self.codec = codec
        self.queue0 = queue0
        self.queue1 = queue1
        self.pc = None
        self.meta_sender = MetaSender()

    async def start(self):
        self.pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
        channel = self.pc.createDataChannel("telemetry")
        self.meta_sender.attach(channel)

        track0 = RawVideoTrack(self.queue0, self.meta_sender, self.send_fps, "cam0")
        track1 = RawVideoTrack(self.queue1, self.meta_sender, self.send_fps, "cam1")

        transceiver0 = self.pc.addTransceiver(track0, direction="sendonly")
        transceiver1 = self.pc.addTransceiver(track1, direction="sendonly")

        if self.codec:
            if not set_preferred_codec(transceiver0, self.codec):
                print(f"Requested codec not available: {self.codec}")
            if not set_preferred_codec(transceiver1, self.codec):
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


def pick_two_cameras(prefer=None):
    caps = gui.enumerate_cameras(prefer=prefer)
    if len(caps) < 2:
        for _, cap in caps:
            cap.release()
        return []
    selected = caps[:2]
    for _, cap in caps[2:]:
        cap.release()
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", required=True, help="Signaling URL, e.g. http://ip:8080/offer")
    parser.add_argument("--codec", default="vp8", help="vp8 or h264")
    parser.add_argument("--send_fps", type=int, default=30)
    parser.add_argument("--send_width", type=int, default=960)
    parser.add_argument("--send_height", type=int, default=540)
    parser.add_argument("--burst_send", type=int, default=0, help="send N frames then drop M frames")
    parser.add_argument("--burst_drop", type=int, default=0, help="send N frames then drop M frames")
    parser.add_argument("--no_display", action="store_true")
    parser.add_argument("--cams", default=None, help="Optional comma-separated camera device paths")
    args = parser.parse_args()

    prefer = None
    if args.cams:
        prefer = [c.strip() for c in args.cams.split(",") if c.strip()]

    stop_event = threading.Event()
    ros_thread = threading.Thread(target=gui.start_ros_spin, args=(stop_event,), daemon=True)
    ros_thread.start()

    caps = []
    last_scan = 0.0

    uw_max_h_top = 240
    sky_max_h_bot = 120
    uw_max_w = gui.SIDE_MARGIN_W - 2 * gui.THUMB_MARGIN
    sky_max_w = gui.SIDE_MARGIN_W - 2 * gui.THUMB_MARGIN

    logo_uw = gui.load_logo_auto(gui.LOGO_UW_CANDS, uw_max_w, uw_max_h_top)
    logo_sky = gui.load_logo_auto(gui.LOGO_SKY_CANDS, sky_max_w, sky_max_h_bot)
    if logo_uw is not None:
        h, w = logo_uw.shape[:2]
        logo_uw = cv2.resize(logo_uw, (max(1, int(w * 0.9)), max(1, int(h * 0.9))))

    if not args.no_display:
        cv2.namedWindow(gui.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(gui.WINDOW_NAME, gui.swap_on_mouse)

    loop = asyncio.new_event_loop()
    queue0 = asyncio.Queue(maxsize=2)
    queue1 = asyncio.Queue(maxsize=2)
    sender = WebRtcSender(args.signal, args.send_fps, args.codec, queue0, queue1)

    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    loop_thread = threading.Thread(target=run_loop, daemon=True)
    loop_thread.start()

    future = asyncio.run_coroutine_threadsafe(sender.start(), loop)
    future.result()

    frame_id0 = 0
    frame_id1 = 0
    cycle_index = 0
    send_interval = 1.0 / float(max(1, args.send_fps))
    next_send_ts = time.time()

    try:
        while True:
            now = time.time()
            if len(caps) < 2 and (now - last_scan) > 0.5:
                caps = pick_two_cameras(prefer)
                last_scan = now
                if len(caps) < 2:
                    time.sleep(0.05)
                    continue

            frames_raw = []
            frames_gui = []
            alive = []
            for dev, cap in caps:
                ok, frame = cap.read()
                if not ok or frame is None:
                    cap.release()
                    continue
                frames_raw.append(frame)
                disp = frame.copy()
                fps_display = random.uniform(55, 60)
                cv2.putText(
                    disp,
                    f"{dev}  {fps_display:.0f}fps",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                frames_gui.append(gui.resize_by_height(disp, gui.TARGET_HEIGHT))
                alive.append((dev, cap))
            caps = alive
            if len(frames_raw) < 2:
                for _, cap in caps:
                    cap.release()
                caps = []
                continue

            frames_gui = gui.maybe_swap_frames(frames_gui)
            frames_gui = gui.pad_to_same_width(frames_gui, pad_value=0)
            video_stack_full = frames_gui[0] if len(frames_gui) == 1 else cv2.vconcat(frames_gui)

            H, W_full = video_stack_full.shape[:2]
            center_col = cv2.convertScaleAbs(video_stack_full, alpha=gui.CENTER_BRIGHT_ALPHA, beta=gui.CENTER_BRIGHT_BETA)

            left_margin = np.full((H, gui.SIDE_MARGIN_W, 3), gui.MARGIN_COLOR, dtype=center_col.dtype)
            right_margin = np.full((H, gui.SIDE_MARGIN_W, 3), gui.MARGIN_COLOR, dtype=center_col.dtype)
            canvas = np.hstack([left_margin, center_col, right_margin])

            gui.draw_left_hud_overlay(canvas)

            right_sidebar_x0 = canvas.shape[1] - gui.SIDE_MARGIN_W
            y_right = gui.THUMB_MARGIN
            y_right = gui.draw_swap_button(canvas, right_sidebar_x0, y_right)
            y_right += 12

            panel_x = right_sidebar_x0 + gui.THUMB_MARGIN
            panel_w = gui.SIDE_MARGIN_W - 2 * gui.THUMB_MARGIN
            pad_y = 10
            pad_between = 20

            h_uw = logo_uw.shape[0] if logo_uw is not None else 0
            h_sky = logo_sky.shape[0] if logo_sky is not None else 0
            logos_h = h_uw + (pad_between if (logo_uw is not None and logo_sky is not None) else 0) + h_sky
            panel_h = max(2 * pad_y + logos_h, 2 * pad_y + 40)

            panel_y = y_right
            gui.draw_rounded_panel(canvas, panel_x, panel_y, panel_w, panel_h, radius=16, border=gui.WINE_BORDER, fill=(255, 255, 255))

            cur_y = panel_y + pad_y
            if logo_uw is not None:
                x_uw = panel_x + (panel_w - logo_uw.shape[1]) // 2
                gui.paste(canvas, logo_uw, x_uw, cur_y)
                cur_y += h_uw + (pad_between if logo_sky is not None else 0)
            if logo_sky is not None:
                x_sky = panel_x + (panel_w - logo_sky.shape[1]) // 2
                gui.paste(canvas, logo_sky, x_sky, cur_y)

            y_right = panel_y + panel_h + 12

            bottom_frame = frames_gui[-1]
            roi_left = gui.crop_mirror_roi(bottom_frame, "left")
            roi_right = gui.crop_mirror_roi(bottom_frame, "right")

            thumb_w = gui.SIDE_MARGIN_W - 2 * gui.THUMB_MARGIN
            thumb_h = int(gui.TARGET_HEIGHT * gui.THUMB_H_FRAC)
            thumb_left = gui.make_panel_cover(roi_left, thumb_w, thumb_h)
            thumb_right = gui.make_panel_cover(roi_right, thumb_w, thumb_h)

            N = len(frames_gui)
            y_base = (N - 1) * gui.TARGET_HEIGHT + gui.THUMB_MARGIN
            gui.paste(canvas, thumb_left, gui.THUMB_MARGIN, y_base)
            gui.paste(canvas, thumb_right, canvas.shape[1] - gui.SIDE_MARGIN_W + gui.THUMB_MARGIN, y_base)

            cv2.putText(canvas, "Left Mirror", (gui.THUMB_MARGIN + 4, y_base - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Right Mirror", (canvas.shape[1] - gui.SIDE_MARGIN_W + gui.THUMB_MARGIN + 4, y_base - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            if not args.no_display:
                cv2.imshow(gui.WINDOW_NAME, canvas)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            now = time.time()
            if now >= next_send_ts:
                send_allowed = True
                if args.burst_send > 0 and args.burst_drop > 0:
                    cycle_len = args.burst_send + args.burst_drop
                    if cycle_index >= args.burst_send:
                        send_allowed = False
                    cycle_index = (cycle_index + 1) % cycle_len

                speed = float(gui.speed_mps_cache.get() or 0.0)
                steer = float(gui.steer_deg_cache.get() or 0.0)
                gear_msg = gui.gear_msg_cache.get()
                gear_val = 0
                if gear_msg is not None:
                    gear_val = int(getattr(gear_msg.gear, "value", 0))
                ts_ms = int(now * 1000)

                if send_allowed:
                    frame_id0 += 1
                    frame_id1 += 1
                    send0 = resize_for_send(frames_raw[0], args.send_width, args.send_height)
                    send1 = resize_for_send(frames_raw[1], args.send_width, args.send_height)

                    item0 = (frame_id0, ts_ms, speed, steer, gear_val, args.send_fps, send0)
                    item1 = (frame_id1, ts_ms, speed, steer, gear_val, args.send_fps, send1)

                    def push_items():
                        if queue0.full():
                            try:
                                queue0.get_nowait()
                            except asyncio.QueueEmpty:
                                pass
                        if queue1.full():
                            try:
                                queue1.get_nowait()
                            except asyncio.QueueEmpty:
                                pass
                        queue0.put_nowait(item0)
                        queue1.put_nowait(item1)

                    loop.call_soon_threadsafe(push_items)
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
