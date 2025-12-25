import os
import sys
import argparse
import csv
import shutil
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'core'))

from config.parser import parse_args
from raft import RAFT
from utils.utils import load_ckpt, coords_grid, bilinear_sampler


def resize_frame(frame, target_height, target_width):
    if target_height <= 0 and target_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if target_width <= 0 and target_height > 0:
        scale = target_height / float(h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
    elif target_height <= 0 and target_width > 0:
        scale = target_width / float(w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
    else:
        new_w = int(target_width)
        new_h = int(target_height)
    new_w = max(2, (new_w // 2) * 2)
    new_h = max(2, (new_h // 2) * 2)
    if new_w == w and new_h == h:
        return frame
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def extract_first_two_frames(video_path, frames_dir, target_fps, resize_height, resize_width, start_time):
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    if start_time > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000.0)

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
    while len(frame_paths) < 2:
        ok, frame = cap.read()
        if not ok:
            break

        if use_index_sampling:
            if frame_idx + 1e-3 >= next_idx:
                frame = resize_frame(frame, resize_height, resize_width)
                out_path = os.path.join(frames_dir, f"frame_{len(frame_paths) + 1:06d}.jpg")
                cv2.imwrite(out_path, frame)
                frame_paths.append(out_path)
                next_idx += interval
        else:
            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if t_ms + 1e-3 >= next_ms:
                frame = resize_frame(frame, resize_height, resize_width)
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
    flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False)
    flow_down = flow_down * (0.5 ** args.scale)
    return flow_down


def predict_next_frame(image2, flow):
    _, _, h, w = image2.shape
    coords = coords_grid(1, h, w, device=image2.device)
    coords = coords + flow
    coords = coords.permute(0, 2, 3, 1)
    pred = bilinear_sampler(image2, coords)
    return pred


def tensor_to_bgr(tensor):
    img = tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def save_frame(tensor, out_path):
    cv2.imwrite(out_path, tensor_to_bgr(tensor))


def write_video(frames_bgr, out_path, fps):
    if not frames_bgr:
        raise ValueError("No frames to write.")
    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for frame in frames_bgr:
        writer.write(frame)
    writer.release()


def read_telemetry(csv_path):
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].strip().startswith('#'):
                continue
            # skip header-like rows
            try:
                values = [float(x) for x in row if x.strip() != '']
            except ValueError:
                continue
            if len(values) >= 2:
                # use last two columns as speed, steer
                rows.append((values[-2], values[-1]))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='config json')
    parser.add_argument('--video', required=True, type=str, help='input video path')
    parser.add_argument('--telemetry_csv', required=True, type=str, help='csv with speed,steer per frame')
    parser.add_argument('--path', type=str, default=None, help='checkpoint path')
    parser.add_argument('--url', type=str, default=None, help='huggingface model id')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--fps', type=float, default=30.0, help='extract/output fps')
    parser.add_argument('--num_frames', type=int, default=100, help='total frames to generate')
    parser.add_argument('--start_time', type=float, default=10.0, help='start time in seconds')
    parser.add_argument('--resize_height', type=int, default=720, help='resize height before inference')
    parser.add_argument('--resize_width', type=int, default=0, help='resize width before inference (0 keeps aspect)')
    parser.add_argument('--speed_eps', type=float, default=1e-3, help='epsilon to avoid divide-by-zero')
    parser.add_argument('--steer_scale', type=float, default=0.0,
                        help='optional scale for horizontal flow using steer (0 disables)')
    parser.add_argument('--frames_dir', type=str, default='mini_project_loop_30fps_meta/frames')
    parser.add_argument('--sea_dir', type=str, default='mini_project_loop_30fps_meta/sea_input')
    parser.add_argument('--out_dir', type=str, default='mini_project_loop_30fps_meta/output')
    parser.add_argument('--output_mode', type=str, default='clean', choices=['clean', 'timestamp'],
                        help='clean old outputs or append timestamp to output dirs')
    args = parse_args(parser)

    if args.path is None and args.url is None:
        raise ValueError("Either --path or --url must be provided")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.num_frames < 3:
        raise ValueError("--num_frames must be >= 3")

    telemetry = read_telemetry(args.telemetry_csv)
    if len(telemetry) < args.num_frames:
        raise ValueError(f"Telemetry rows {len(telemetry)} < num_frames {args.num_frames}")

    if args.output_mode == 'timestamp':
        stamp = time.strftime('%Y%m%d_%H%M%S')
        args.frames_dir = f"{args.frames_dir}_{stamp}"
        args.sea_dir = f"{args.sea_dir}_{stamp}"
        args.out_dir = f"{args.out_dir}_{stamp}"
    else:
        for path in [args.frames_dir, args.sea_dir, args.out_dir]:
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.exists(path):
                os.remove(path)

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = torch.device('cpu')

    frame_paths = extract_first_two_frames(
        args.video, args.frames_dir, args.fps, args.resize_height, args.resize_width, args.start_time
    )
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

    frames = [image1, image2]
    frames_bgr = [tensor_to_bgr(image1), tensor_to_bgr(image2)]

    meta = [(telemetry[0][0], telemetry[0][1]), (telemetry[1][0], telemetry[1][1])]
    scale_used = [1.0, 1.0]

    total_time = 0.0
    with torch.no_grad():
        for idx in range(2, args.num_frames):
            v_curr, _ = meta[-1]
            v_out, steer_out = telemetry[idx]
            speed_scale = v_out / max(args.speed_eps, v_curr)
            if args.steer_scale != 0.0:
                speed_scale = speed_scale * (1.0 + args.steer_scale * steer_out)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            flow = calc_flow(args, model, frames[-2], frames[-1])
            if device.type == 'cuda':
                torch.cuda.synchronize()
            total_time += time.perf_counter() - start

            # Approximate forward flow using speed scaling.
            flow_forward = -flow * speed_scale
            next_frame = predict_next_frame(frames[-1], flow_forward)
            frames.append(next_frame)
            frames_bgr.append(tensor_to_bgr(next_frame))
            meta.append((v_out, steer_out))
            scale_used.append(speed_scale)

    os.makedirs(args.out_dir, exist_ok=True)
    out_frames_dir = os.path.join(args.out_dir, "frames")
    os.makedirs(out_frames_dir, exist_ok=True)
    for idx, frame in enumerate(frames, start=1):
        out_path = os.path.join(out_frames_dir, f"frame_{idx:06d}.jpg")
        save_frame(frame, out_path)

    video_path = os.path.join(args.out_dir, "pred_video.mp4")
    write_video(frames_bgr, video_path, args.fps)

    meta_path = os.path.join(args.out_dir, "telemetry_used.csv")
    with open(meta_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'speed_mps', 'steer', 'scale_used'])
        for i, ((speed, steer), scale) in enumerate(zip(meta, scale_used), start=1):
            writer.writerow([i, speed, steer, scale])

    avg_ms = (total_time / (args.num_frames - 2)) * 1000.0
    avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    print(f"Generated frames: {args.num_frames} | Avg inference: {avg_ms:.2f} ms | FPS: {avg_fps:.2f}")
    print(f"Output dir: {args.out_dir}")
    print(f"Output video: {video_path}")
    print(f"Telemetry log: {meta_path}")


if __name__ == '__main__':
    main()
