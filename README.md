[New!] Please also check [WAFT](https://github.com/princeton-vl/WAFT), our new efficient state-of-the-art method.

# SEA-RAFT

[[Paper](https://arxiv.org/abs/2405.14793)][[Slides](https://docs.google.com/presentation/d/1xZn-NowHuPqfdLDAaQwKyzYvP4HzGmT7/edit?usp=sharing&ouid=118125745783453356964&rtpof=true&sd=true)]

We introduce SEA-RAFT, a more simple, efficient, and accurate [RAFT](https://github.com/princeton-vl/RAFT) for optical flow. Compared with RAFT, SEA-RAFT is trained with a new loss (mixture of Laplace). It directly regresses an initial flow for faster convergence in iterative refinements and introduces rigid-motion pre-training to improve generalization. SEA-RAFT achieves state-of-the-art accuracy on the [Spring benchmark](https://spring-benchmark.org/) with a 3.69 endpoint-error (EPE) and a 0.36 1-pixel outlier rate (1px), representing 22.9\% and 17.8\% error reduction from best-published results. In addition, SEA-RAFT obtains the best cross-dataset generalization on KITTI and Spring. With its high efficiency, SEA-RAFT operates at least 2.3x faster than existing methods while maintaining competitive performance.

<img src="assets/visualization.png" width='1000'>

If you find SEA-RAFT useful for your work, please consider citing our academic paper:

<h3 align="center">
    <a href="https://arxiv.org/abs/2405.14793">
        SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow
    </a>
</h3>
<p align="center">
    <a href="https://memoryslices.github.io/">Yihan Wang</a>,
    <a href="https://www.lahavlipson.com/">Lahav Lipson</a>,
    <a href="http://www.cs.princeton.edu/~jiadeng">Jia Deng</a><br>
</p>

```
@article{wang2024sea,
  title={SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow},
  author={Wang, Yihan and Lipson, Lahav and Deng, Jia},
  journal={arXiv preprint arXiv:2405.14793},
  year={2024}
}
```

## Requirements
Our code is developed with pytorch 2.2.0, CUDA 12.2 and python 3.10.
```Shell
conda create --name SEA-RAFT python=3.10.13
conda activate SEA-RAFT
pip install -r requirements.txt
```

## Model Zoo

Google Drive: [link](https://drive.google.com/drive/folders/1YLovlvUW94vciWvTyLf-p3uWscbOQRWW?usp=sharing).

HuggingFace: [link](https://huggingface.co/papers/2405.14793).

## Custom Usage

We provide an example in `custom.py`. By default, this file will take two RGB images as the input and provide visualizations of the optical flow and the uncertainty. You can load your model by providing the path:
```Shell
python custom.py --cfg config/eval/spring-M.json --path models/Tartan-C-T-TSKH-spring540x960-M.pth
```
or load our models through HuggingFace🤗 (make sure you have installed huggingface-hub):
```Shell
python custom.py --cfg config/eval/spring-M.json --url MemorySlices/Tartan-C-T-TSKH-spring540x960-M
```

## Datasets
To evaluate/train SEA-RAFT, you will need to download the required datasets: [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs), [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [Sintel](http://sintel.is.tue.mpg.de/), [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow), [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/), [TartanAir](https://theairlab.org/tartanair-dataset/), and [Spring](https://spring-benchmark.org/).

By default `datasets.py` will search for the datasets in these locations. You can create symbolic links to wherever the datasets were downloaded in the `datasets` folder. Please check [RAFT](https://github.com/princeton-vl/RAFT) for more details.

```Shell
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

## Training, Evaluation, and Submission

Please refer to [scripts/train.sh](scripts/train.sh), [scripts/eval.sh](scripts/eval.sh), and [scripts/submission.sh](scripts/submission.sh) for more details.

## 实时低延迟传输与插帧（Tailscale + WebRTC）

下面是本项目在“车端 → 电脑端”实时传输 GUI 画面并进行丢帧补帧的完整使用说明。

### 功能概述
- 车端采集相机并渲染 GUI（HUD/Logo/镜像），本地显示同时通过 WebRTC 发送到电脑端。
- 通过 DataChannel 发送每帧元数据：`ts_ms`、`speed`、`steer`、`fps`、`width`、`height`。
- 电脑端检测丢帧（基于 `ts_ms` 间隔）并用 SEA-RAFT 预测插帧，预测帧会显示 `PRED` 标记。
- 支持“仅显示不插帧”的调试模式。

### 相关脚本
- 车端发送：`webrtc_tailscale_realtime/car_gui_sender.py`
- 电脑端仅显示（调试）：`webrtc_tailscale_realtime/receiver_viewer.py`
- 电脑端插帧：`webrtc_tailscale_realtime/receiver_realtime.py`

### 环境准备（电脑端）
推荐在 `sea-raft` 环境中运行，并固定以下版本以避免不兼容问题：
- `numpy=1.26.4`
- `scipy=1.11.4`
- `opencv=4.8.1`（conda-forge）

```Shell
conda activate sea-raft
conda install -y "numpy=1.26.4" "scipy=1.11.4"
conda install -y -c conda-forge "opencv=4.8.1"
pip install aiortc aiohttp av
```

### 车端操作（发送端）
1) 确认 Tailscale 已连通（能 ping 通电脑端 IP）。
2) 运行发送脚本：

```Shell
python3 webrtc_tailscale_realtime/car_gui_sender.py \
  --signal http://<PC_TAILSCALE_IP>:8080/offer \
  --send_fps 30 \
  --send_width 960 --send_height 540 \
  --codec vp8
```

常用参数：
- `--send_fps`：发送帧率
- `--send_width/--send_height`：发送分辨率（降低可减少带宽和延迟）
- `--codec`：`vp8`（兼容好）或 `h264`
- `--no_display`：车端不显示窗口（仅发送）

说明：
- 车端本地会显示 GUI（默认）。
- 速度/方向盘角度来自 ROS2 话题（示例在脚本内），不可用时会用 0。

### 电脑端操作（仅显示/调试）
1) 启动调试接收器（不做插帧）：

```Shell
conda run -n sea-raft python webrtc_tailscale_realtime/receiver_viewer.py \
  --listen 0.0.0.0 --port 8080
```

预期输出：
- 终端看到 `[viewer] track: video`
- 终端每秒打印 `[viewer] video fps ~ ...`
- 窗口出现实时画面

### 电脑端操作（插帧）
1) 启动插帧接收器：

```Shell
conda run -n sea-raft python webrtc_tailscale_realtime/receiver_realtime.py \
  --listen 0.0.0.0 --port 8080 \
  --cfg config/eval/kitti-M.json \
  --path weight/Tartan-C-T-TSKH-kitti432x960-M.pth \
  --fps 30 --device cuda
```

插帧规则：
- 用 `ts_ms` 的间隔判断是否丢帧（默认阈值为 1.5 倍帧间隔）。
- 每 5 帧构建一次预测序列，基于前 2 帧推后 8 帧。
- 预测流会根据速度比例进行缩放（速度越大，光流越长）。
- 预测帧显示 `PRED` 标记。

常用参数：
- `--fps`：期望帧率（影响丢帧判断）
- `--gap_threshold`：丢帧阈值系数（默认 1.5）
- `--window_size`：构建预测序列的窗口大小（默认 5）
- `--horizon`：每次预测的帧数（默认 8）
- `--raw_only`：只显示，不插帧

### 常见问题
- 窗口不显示：请确认不是在纯 SSH 无图形环境；需要本地桌面环境或 X11 转发。
- 画面空白：先用 `receiver_viewer.py` 验证链路；若无视频帧，检查车端摄像头是否打开。
- 性能不足：降低 `--send_width/--send_height` 或发送帧率。

## Acknowledgements

This project relies on code from existing repositories: [RAFT](https://github.com/princeton-vl/RAFT), [unimatch](https://github.com/autonomousvision/unimatch/tree/master), [Flowformer](https://github.com/drinkingcoder/FlowFormer-Official), [ptlflow](https://github.com/hmorimitsu/ptlflow), and [LoFTR](https://github.com/zju3dv/LoFTR). We thank the original authors for their excellent work.
