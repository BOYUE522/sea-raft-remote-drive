# sea-raft-remote-drive

## 项目用途
- 通过 Tailscale + WebRTC 把车端双摄像头视频低延迟传到电脑端，同时把车速/方向等元数据同步过来，实现远程观察与控制。
- 当网络丢帧时，电脑端用 SEA-RAFT 做光流补帧，尽量保持画面连续，减少卡顿。
- 电脑端可接 Logitech G920 方向盘，通过 ROSBridge 远程控制车。

## 核心原理
- 车端用 WebRTC 发送两路原始相机视频，并通过 DataChannel 发送每帧的时间戳、车速、方向等信息。
- 电脑端接收视频与元数据，按时间戳检测缺帧；若发现缺帧，则用相邻两帧计算光流并预测中间帧进行补齐。
- 画面显示端按实时帧率渲染，若有预测帧就插入到显示流中；一旦新的真实帧到来，立即切回真实帧。

## 总说明书（电脑端 + 车端完整流程）
先在电脑端接好 G920 方向盘与显示器，并给电脑端接上非校园网（例如手机热点）。

步骤 1（电脑端，启动接收与补帧）：
```bash
conda run -n sea-raft python webrtc_tailscale_realtime/receiver_realtime.py \
  --listen 0.0.0.0 --port 8080 \
  --cfg config/eval/kitti-M.json \
  --path weight/Tartan-C-T-TSKH-kitti432x960-M.pth \
  --fps 30 --device cuda
```

步骤 2（车端，启动 ROS/DBW 相关）：
```bash
~/ros2_ws/src/boyue$ ./start_g920_alienware.sh
```

步骤 3（车端，打开另一个 terminal，启动双摄像头发送）：
```bash
~/ros2_ws/src/boyue/webrtc_tailscale_realtime$ python3 car_dual_sender.py \
  --signal http://100.78.251.61:8080/offer \
  --send_fps 30 --send_width 960 --send_height 540 --codec vp8
```

步骤 4（电脑端，打开另一个 terminal，启动方向盘控制）：
```bash
python /home/boyue/SEA/SEA-RAFT-main/webrtc_tailscale_realtime/g920_tailscale_control.py
```

## 原理详解（学术版）
### 1) 系统架构与同步机制
- 车端通过 WebRTC 发送两路视频流，使用 DataChannel 同步每帧元数据（时间戳 ts_ms、车速 v、方向 steer 等）。
- 电脑端用 Tailscale 作为安全覆盖网络，保证跨网段、NAT 场景下仍可建立 P2P 或中继链路。
- 接收端将视频帧与元数据按时间顺序对齐，形成时序样本序列。

### 2) 缺帧检测模型
- 设目标帧率为 fps，期望间隔 Δt = 1000/fps ms。
- 若相邻两帧时间戳间隔 ΔT > κ·Δt（κ 为阈值），判定存在缺帧，缺帧数量约为 m = round(ΔT/Δt) - 1。
- 对于低帧率输入，按 fps_target / fps_input 估计补帧数量。

### 3) 光流估计与插值/外推
- 使用 SEA-RAFT 估计密集光流 F（像素级位移场），捕捉局部运动。
- 在“局部恒速”假设下，中间时刻位移可视为线性缩放：F_α = α · F，其中 α = Δ/Δt。
- 通过反向/双线性重采样进行图像变换（warping）：Î(t) = W(I_ref, F_α)，其中 W(·) 为采样算子，I_ref 为参考帧。
- 若引入速度信息，可做速度比例修正：F_α = (v_target / v_ref) · α · F，用于近似加速/减速的运动幅度变化。

### 4) 多相机并行与可视化
- 双相机各自独立维护窗口与光流序列，分别补帧。
- 电脑端将两路结果融合到本地 GUI（与车端一致），显示仪表与视野。

## 致谢与参考
以下内容来自 SEA-RAFT 项目与论文（原作者信息保留）：

[New!] Please also check WAFT, our new efficient state-of-the-art method.
SEA-RAFT

[Paper][Slides]

We introduce SEA-RAFT, a more simple, efficient, and accurate RAFT for optical flow. Compared with RAFT, SEA-RAFT is trained with a new loss (mixture of Laplace). It directly regresses an initial flow for faster convergence in iterative refinements and introduces rigid-motion pre-training to improve generalization. SEA-RAFT achieves state-of-the-art accuracy on the Spring benchmark with a 3.69 endpoint-error (EPE) and a 0.36 1-pixel outlier rate (1px), representing 22.9% and 17.8% error reduction from best-published results. In addition, SEA-RAFT obtains the best cross-dataset generalization on KITTI and Spring. With its high efficiency, SEA-RAFT operates at least 2.3x faster than existing methods while maintaining competitive performance.

If you find SEA-RAFT useful for your work, please consider citing our academic paper:
SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow

Yihan Wang, Lahav Lipson, Jia Deng

```bibtex
@article{wang2024sea,
  title={SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow},
  author={Wang, Yihan and Lipson, Lahav and Deng, Jia},
  journal={arXiv preprint arXiv:2405.14793},
  year={2024}
}
```

Requirements

Our code is developed with pytorch 2.2.0, CUDA 12.2 and python 3.10.

```
conda create --name SEA-RAFT python=3.10.13
conda activate SEA-RAFT
pip install -r requirements.txt
```

Model Zoo

Google Drive: link.

HuggingFace: link.

Custom Usage

We provide an example in custom.py. By default, this file will take two RGB images as the input and provide visualizations of the optical flow and the uncertainty. You can load your model by providing the path:

```
python custom.py --cfg config/eval/spring-M.json --path models/Tartan-C-T-TSKH-spring540x960-M.pth
```

or load our models through HuggingFace (make sure you have installed huggingface-hub):

```
python custom.py --cfg config/eval/spring-M.json --url MemorySlices/Tartan-C-T-TSKH-spring540x960-M
```

Datasets

To evaluate/train SEA-RAFT, you will need to download the required datasets: FlyingChairs, FlyingThings3D, Sintel, KITTI, HD1K, TartanAir, and Spring.

By default datasets.py will search for the datasets in these locations. You can create symbolic links to wherever the datasets were downloaded in the datasets folder. Please check RAFT for more details.

```
├── datasets
    ├── Sintel
    ├── KITTI
    ├── FlyingChairs/FlyingChairs_release
    ├── FlyingThings3D
    ├── HD1K
    ├── spring
        ├── test
        ├── train
        ├── val
    ├── tartanair
```

Training, Evaluation, and Submission
Please refer to scripts/train.sh, scripts/eval.sh, and scripts/submission.sh for more details.
