# Tailscale WebRTC + Telemetry (Real-time Fill)

This project receives a car camera stream over Tailscale WebRTC and uses per-frame
timestamp + speed metadata to detect missing frames and fill them locally.

## Requirements
```
pip install aiortc aiohttp av opencv-python torch torchvision
```

## How It Works
- **Sender (car)**: sends video over WebRTC and a DataChannel message per frame:
  `{frame_id, ts_ms, speed, steer}`.
- **Receiver (local)**: aligns metadata with video frames, detects gaps by `ts_ms`,
  and fills missing frames using a cached 5-frame window prediction.

Prediction policy (as confirmed):
- Every **5 real frames** (30 fps window), use the **1st and 2nd frame** to predict
  the **next 8 frames**. Store these predictions.
- If a gap is detected, use the cached predictions whose timestamps match the
  missing frame times. When a real frame arrives, switch back immediately.
- Flow scaling uses **speed ratio**: `scale = v_target / v_current`.

## Run
Local receiver (your machine, Tailscale IP `100.78.251.61`):
```
python receiver_realtime.py \
  --listen 0.0.0.0 --port 8080 \
  --cfg /home/boyue/SEA/SEA-RAFT-main/config/eval/kitti-M.json \
  --path /home/boyue/SEA/SEA-RAFT-main/weight/Tartan-C-T-TSKH-kitti432x960-M.pth \
  --fps 30 --device cuda
```

Car sender (Tailscale IP `100.65.223.76`):
```
python sender_meta.py \
  --signal http://100.78.251.61:8080/offer \
  --camera 0 --width 1280 --height 720 --fps 30 \
  --telemetry_mode static --speed 3.0 --steer 0.0
```

Later, switch telemetry to `udp` once you provide the speed source:
```
--telemetry_mode udp --telemetry_udp_port 5001
```

## Notes
- `ts_ms` comes from the sender clock (`time.time()`).
- If you want to run headless (no display), add `--no_display`.
- Lower resolution/fps reduces latency.

## G920 Wheel Control (optional)
This script lets you control the car using a Logitech G920 over Tailscale + ROSBridge.

Path:
- `webrtc_tailscale_realtime/g920_tailscale_control.py`

Install dependencies:
```
pip install roslibpy inputs
```

Edit the car IP at the top of the script (`CAR_TAILSCALE_IP`), then run:
```
python g920_tailscale_control.py
```

## 总说明书（电脑端 + 车端完整流程）
先在电脑端接好 G920 方向盘与显示器。

步骤 1（电脑端，启动接收与补帧）：
```
conda run -n sea-raft python webrtc_tailscale_realtime/receiver_realtime.py \
  --listen 0.0.0.0 --port 8080 \
  --cfg config/eval/kitti-M.json \
  --path weight/Tartan-C-T-TSKH-kitti432x960-M.pth \
  --fps 30 --device cuda
```

步骤 2（车端，启动 ROS/DBW 相关）：
```
uw@uw-Nuvo:~/ros2_ws/src/boyue$ ./start_g920_alienware.sh
```

步骤 3（车端，启动双摄像头发送）：
```
uw@uw-Nuvo:~/ros2_ws/src/boyue/webrtc_tailscale_realtime$ python3 car_dual_sender.py \
  --signal http://100.78.251.61:8080/offer \
  --send_fps 30 --send_width 960 --send_height 540 --codec vp8
```

步骤 4（电脑端，启动方向盘控制）：
```
python /home/boyue/SEA/SEA-RAFT-main/webrtc_tailscale_realtime/g920_tailscale_control.py
```
