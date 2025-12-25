import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'core'))

import argparse
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config.parser import parse_args
from raft import RAFT
from utils.flow_viz import flow_to_image
from utils.utils import load_ckpt, coords_grid, bilinear_sampler


def extract_frames(video_path, frames_dir, target_fps):
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    use_index_sampling = src_fps is not None and src_fps > 0
    if use_index_sampling:
        interval = src_fps / target_fps
        next_idx = 0.0
    else:
        interval_ms = 1000.0 / target_fps
        next_ms = 0.0

    frame_paths = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if use_index_sampling:
            if frame_idx + 1e-3 >= next_idx:
                out_path = os.path.join(frames_dir, f"frame_{len(frame_paths) + 1:06d}.jpg")
                cv2.imwrite(out_path, frame)
                frame_paths.append(out_path)
                next_idx += interval
        else:
            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if t_ms + 1e-3 >= next_ms:
                out_path = os.path.join(frames_dir, f"frame_{len(frame_paths) + 1:06d}.jpg")
                cv2.imwrite(out_path, frame)
                frame_paths.append(out_path)
                next_ms += interval_ms

        frame_idx += 1

    cap.release()
    return frame_paths


def save_sea_inputs(frame_paths, sea_dir):
    if len(frame_paths) < 2:
        raise ValueError("Need at least 2 extracted frames.")
    os.makedirs(sea_dir, exist_ok=True)
    img1 = cv2.imread(frame_paths[0])
    img2 = cv2.imread(frame_paths[1])
    if img1 is None or img2 is None:
        raise ValueError("Failed to read extracted frames for SEA inputs.")
    cv2.imwrite(os.path.join(sea_dir, "image1.jpg"), img1)
    cv2.imwrite(os.path.join(sea_dir, "image2.jpg"), img2)
    return os.path.join(sea_dir, "image1.jpg"), os.path.join(sea_dir, "image2.jpg")


def load_rgb_tensor(path, device):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img).float().permute(2, 0, 1)[None].to(device)
    return tensor


def calc_flow(args, model, image1, image2):
    img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    output = model(img1, img2, iters=args.iters, test_mode=True)
    flow = output['flow'][-1]
    info = output['info'][-1]
    flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
    info_down = F.interpolate(info, scale_factor=0.5 ** args.scale, mode='area')
    return flow_down, info_down


def predict_next_frame(image2, flow):
    _, _, h, w = image2.shape
    coords = coords_grid(1, h, w, device=image2.device)
    coords = coords + flow
    coords = coords.permute(0, 2, 3, 1)
    pred = bilinear_sampler(image2, coords)
    return pred


def save_rgb_tensor(tensor, out_path):
    img = tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='config json')
    parser.add_argument('--video', required=True, type=str, help='input video path')
    parser.add_argument('--path', type=str, default=None, help='checkpoint path')
    parser.add_argument('--url', type=str, default=None, help='huggingface model id')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--fps', type=float, default=10.0, help='extract fps')
    parser.add_argument('--frames_dir', type=str, default='mini_project/frames')
    parser.add_argument('--sea_dir', type=str, default='mini_project/sea_input')
    parser.add_argument('--out_dir', type=str, default='mini_project/output')
    args = parse_args(parser)

    if args.path is None and args.url is None:
        raise ValueError("Either --path or --url must be provided")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = torch.device('cpu')

    frame_paths = extract_frames(args.video, args.frames_dir, args.fps)
    img1_path, img2_path = save_sea_inputs(frame_paths, args.sea_dir)

    if args.path is not None:
        model = RAFT(args)
        load_ckpt(model, args.path)
    else:
        model = RAFT.from_pretrained(args.url, args=args)

    model = model.to(device)
    model.eval()

    image1 = load_rgb_tensor(img1_path, device)
    image2 = load_rgb_tensor(img2_path, device)

    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        # Use reverse flow (t+1 -> t) so warping image2 moves forward.
        flow, _ = calc_flow(args, model, image2, image1)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        pred3 = predict_next_frame(image2, flow)

    os.makedirs(args.out_dir, exist_ok=True)
    save_rgb_tensor(pred3, os.path.join(args.out_dir, "frame_000003.jpg"))
    flow_vis = flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
    cv2.imwrite(os.path.join(args.out_dir, "flow.jpg"), flow_vis)

    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    print(f"Inference time: {elapsed * 1000:.2f} ms | FPS: {fps:.2f}")


if __name__ == '__main__':
    main()
