"""Generate Failure Case visual evidence for PointPillars 3D detector.
Analyzes Frame 000001: Missing distant Cyclist (46.1m) due to LiDAR point sparsity.

Usage:
    python src/visualize_failure.py
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from starter.datasets import load_frame
from starter.kitti_io import KittiObject
from src.pointpillars import (
    PointPillarsDetector,
    box3d_to_corners_lidar,
    project_box3d_lidar_to_camera,
    draw_box3d_on_image,
)
from src.run_detector import ensure_checkpoint


def main():
    ckpt_path = ensure_checkpoint()
    fid = "000001"
    fr = load_frame("data/kitti_mini", fid)
    points = fr["points"]
    calib = fr["calib"]
    img = fr["image"]
    labels = fr["labels"]

    detector = PointPillarsDetector(checkpoint_path=ckpt_path, device="cuda")
    dets = detector.predict(points, score_thr=0.3)

    # Find the cyclist in GT: loc = [4.59, 1.32, 45.84] (camera frame)
    # In LiDAR frame: x ~ 45.8, y ~ -4.6, z ~ -1.3
    gt_cyclist = [o for o in labels if o.type == "Cyclist"][0]

    # Create visualization
    vis_img = img.copy()

    # Draw detected boxes in green/cyan
    for det in dets:
        corners_lidar = box3d_to_corners_lidar(det["box3d"])
        corners_2d = project_box3d_lidar_to_camera(corners_lidar, calib, img.shape)
        if corners_2d is not None:
            vis_img = draw_box3d_on_image(
                vis_img, corners_2d, color=(0, 255, 0), label=f"Pred {det['name']} {det['score']:.2f}"
            )

    # Draw Ground Truth 2D box for Cyclist in RED with warning
    c_box = [int(v) for v in gt_cyclist.bbox]
    cv2.rectangle(vis_img, (c_box[0], c_box[1]), (c_box[2], c_box[3]), (0, 0, 255), 3)
    cv2.putText(
        vis_img, "MISSED GT CYCLIST (dist=46.1m)", (c_box[0] - 80, c_box[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
    )

    # Build side-by-side plot with BEV crop
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Top: Camera view
    axes[0].imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Failure Case: Frame {fid} — Distant Cyclist Missed by PointPillars Detector (Camera Overlay)", fontsize=13)
    axes[0].axis("off")

    # Bottom: BEV point cloud zoomed into the cyclist area
    x_c = 45.8
    y_c = -4.6
    crop_mask = (points[:, 0] >= x_c - 10) & (points[:, 0] <= x_c + 10) & \
                (points[:, 1] >= y_c - 10) & (points[:, 1] <= y_c + 10)
    crop_pts = points[crop_mask]

    axes[1].scatter(crop_pts[:, 1], crop_pts[:, 0], c="deepskyblue", s=12, alpha=0.8, label="LiDAR Points")
    axes[1].scatter([y_c], [x_c], c="red", marker="x", s=150, linewidth=3, label="GT Cyclist Location (dist=46.1m)")
    circle = plt.Circle((y_c, x_c), 1.5, color="red", fill=False, linestyle="--", linewidth=2, label="Cyclist BBox Area")
    axes[1].add_patch(circle)

    # Annotate number of points in cyclist area
    pts_in_cyl = crop_pts[(np.abs(crop_pts[:, 0] - x_c) < 1.0) & (np.abs(crop_pts[:, 1] - y_c) < 0.8)]
    axes[1].annotate(
        f"Only {len(pts_in_cyl)} LiDAR points inside object bbox!\n(Insufficient for PillarFeatureNet activation)",
        xy=(y_c, x_c), xytext=(y_c + 2.5, x_c - 3),
        arrowprops=dict(facecolor="yellow", edgecolor="black", shrink=0.05, width=2),
        fontsize=11, bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.9)
    )

    axes[1].set_xlim(y_c - 8, y_c + 8)
    axes[1].set_ylim(x_c - 6, x_c + 6)
    axes[1].set_xlabel("Lateral Y (meters)", fontsize=11)
    axes[1].set_ylabel("Forward X (meters)", fontsize=11)
    axes[1].set_title(f"BEV LiDAR Inspection around Cyclist (Sparse Beam Density at Range > 45m)", fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    out_path = Path("results/figures/fail_01_distant_cyclist_missed.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Failure case visualization saved to {out_path}")


if __name__ == "__main__":
    main()
