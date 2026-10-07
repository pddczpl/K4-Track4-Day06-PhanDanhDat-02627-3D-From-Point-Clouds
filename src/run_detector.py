"""Run 3D Object Detector (PointPillars) inference on LiDAR frames.

Usage:
    python src/run_detector.py --data-root data/kitti_mini --frame 000011 --score-thr 0.3
    python src/run_detector.py --data-root data/kitti_mini --frame 000001 --score-thr 0.3
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from starter.datasets import load_frame
from src.pointpillars import (
    PointPillarsDetector,
    box3d_to_corners_lidar,
    project_box3d_lidar_to_camera,
    draw_box3d_on_image,
    visualize_bev,
    CLASS_COLORS,
)

CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/"
    "hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/"
    "hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth"
)
DEFAULT_CKPT_PATH = Path("checkpoints/pointpillars_kitti_3class.pth")


def ensure_checkpoint(ckpt_path: Path = DEFAULT_CKPT_PATH) -> Path:
    if not ckpt_path.exists():
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading pre-trained PointPillars weights from {CHECKPOINT_URL}...")
        urllib.request.urlretrieve(CHECKPOINT_URL, ckpt_path)
        print(f"Saved weights to {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.1f} MB)")
    return ckpt_path


def main():
    parser = argparse.ArgumentParser(description="Run PointPillars 3D object detection on LiDAR frames")
    parser.add_argument("--data-root", default="data/kitti_mini", help="Path to KITTI dataset split")
    parser.add_argument("--frame", default="000011", help="Frame ID to infer")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Detection score threshold")
    parser.add_argument("--nms-thr", type=float, default=0.1, help="NMS IoU threshold")
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT_PATH), help="Path to model checkpoint")
    parser.add_argument("--out-dir", default="results/figures", help="Output directory for visualizations")
    parser.add_argument("--device", default="cuda", help="Execution device: cuda or cpu")
    args = parser.parse_args()

    ckpt_path = ensure_checkpoint(Path(args.ckpt))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading frame {args.frame} from {args.data_root}...")
    fr = load_frame(args.data_root, args.frame)
    points = fr["points"]
    calib = fr["calib"]
    image = fr["image"]
    gt_labels = fr.get("labels", [])

    print(f"Initializing PointPillars detector on {args.device} (GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() and args.device == 'cuda' else 'CPU'})...")
    detector = PointPillarsDetector(checkpoint_path=ckpt_path, device=args.device)

    print(f"Running inference (score_thr={args.score_thr})...")
    t0 = cv2.getTickCount()
    detections = detector.predict(points, score_thr=args.score_thr, nms_iou_thr=args.nms_thr)
    infer_time_ms = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000

    print(f"Detected {len(detections)} objects in {infer_time_ms:.1f} ms:")
    print("-" * 75)
    print(f"{'Class':12s} {'Score':>7s} {'Dist(m)':>8s} {'LiDAR Box [x, y, z, dx, dy, dz, yaw]'}")
    print("-" * 75)
    for det in detections:
        b = [round(v, 2) for v in det["box3d"]]
        print(f"{det['name']:12s} {det['score']:7.2f} {det['range']:8.1f} {b}")
    print("-" * 75)

    # 1. Save BEV visualization
    bev_out = out_dir / f"demo_bev_{args.frame}.png"
    visualize_bev(points, detections, bev_out)
    print(f"Saved BEV visualization -> {bev_out}")

    # 2. Project 3D bounding boxes to camera image
    vis_img = image.copy()
    for det in detections:
        corners_lidar = box3d_to_corners_lidar(det["box3d"])
        corners_2d = project_box3d_lidar_to_camera(corners_lidar, calib, image.shape)
        if corners_2d is not None:
            color = CLASS_COLORS.get(det["name"], (0, 255, 0))
            label_text = f"{det['name']} {det['score']:.2f}"
            vis_img = draw_box3d_on_image(vis_img, corners_2d, color=color, label=label_text)

    img_out = out_dir / f"demo_pred3d_{args.frame}.png"
    cv2.imwrite(str(img_out), vis_img)
    print(f"Saved 3D box projection -> {img_out}")


if __name__ == "__main__":
    main()
