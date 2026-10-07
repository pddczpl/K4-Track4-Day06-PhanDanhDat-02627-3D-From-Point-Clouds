"""Benchmark 3D Object Detector (PointPillars) on NVIDIA GeForce RTX 3060.
Measures latency (p50/p95 over >=20 runs), score distributions, range distributions,
and compares 2 configurations (score_thr=0.3 vs score_thr=0.5).

Usage:
    python src/benchmark_detector.py --data-root data/kitti_mini
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from starter.datasets import load_frame
from src.pointpillars import PointPillarsDetector
from src.run_detector import ensure_checkpoint, DEFAULT_CKPT_PATH


BENCHMARK_FRAMES = ["000001", "000008", "000011", "000012", "000049"]


def measure_frame_latency(detector: PointPillarsDetector, points: np.ndarray,
                          score_thr: float, warmup: int = 5, reps: int = 25) -> tuple[float, float, float]:
    """Measure inference latency adhering to Rubric B3:
    drops initial warmups and runs at least 20 synchronized repetitions.
    Returns: (mean_ms, p50_ms, p95_ms)
    """
    for _ in range(warmup):
        detector.predict(points, score_thr=score_thr)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    timings = []
    for _ in range(reps):
        t0 = time.perf_counter()
        detector.predict(points, score_thr=score_thr)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000
        timings.append(dt_ms)

    return float(np.mean(timings)), float(np.percentile(timings, 50)), float(np.percentile(timings, 95))


def main():
    parser = argparse.ArgumentParser(description="Benchmark PointPillars on KITTI dataset with RTX 3060")
    parser.add_argument("--data-root", default="data/kitti_mini", help="Path to KITTI dataset")
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT_PATH), help="Path to model checkpoint")
    parser.add_argument("--out-csv", default="results/detector_benchmark.csv", help="Output benchmark CSV path")
    parser.add_argument("--out-dir", default="results/figures", help="Output directory for plots")
    parser.add_argument("--device", default="cuda", help="Execution device: cuda or cpu")
    args = parser.parse_args()

    ckpt_path = ensure_checkpoint(Path(args.ckpt))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() and args.device == "cuda" else "CPU"
    print(f"=== PointPillars 3D Detector Benchmark on {gpu_name} ===")

    detector = PointPillarsDetector(checkpoint_path=ckpt_path, device=args.device)

    rows = []
    all_detections_conf1 = []
    all_detections_conf2 = []

    configs = [
        {"name": "Conf_A (score_thr=0.3)", "score_thr": 0.3},
        {"name": "Conf_B (score_thr=0.5)", "score_thr": 0.5},
    ]

    for fid in BENCHMARK_FRAMES:
        print(f"\nProcessing Frame {fid}...")
        fr = load_frame(args.data_root, fid)
        pts = fr["points"]
        gt_count = len(fr.get("labels", []))

        for cfg in configs:
            s_thr = cfg["score_thr"]
            mean_ms, p50_ms, p95_ms = measure_frame_latency(detector, pts, score_thr=s_thr)
            dets = detector.predict(pts, score_thr=s_thr)

            if s_thr == 0.3:
                all_detections_conf1.extend(dets)
            else:
                all_detections_conf2.extend(dets)

            cars = sum(1 for d in dets if d["name"] == "Car")
            peds = sum(1 for d in dets if d["name"] == "Pedestrian")
            cycs = sum(1 for d in dets if d["name"] == "Cyclist")
            mean_score = float(np.mean([d["score"] for d in dets])) if dets else 0.0
            max_range = float(np.max([d["range"] for d in dets])) if dets else 0.0

            status = "PASS" if len(dets) > 0 else "FAIL"

            row = {
                "frame_id": fid,
                "config": cfg["name"],
                "score_threshold": s_thr,
                "points_count": len(pts),
                "gt_object_count": gt_count,
                "detected_objects": len(dets),
                "cars": cars,
                "pedestrians": peds,
                "cyclists": cycs,
                "mean_score": round(mean_score, 3),
                "max_range_m": round(max_range, 2),
                "latency_mean_ms": round(mean_ms, 2),
                "latency_p50_ms": round(p50_ms, 2),
                "latency_p95_ms": round(p95_ms, 2),
                "fps": round(1000.0 / p50_ms, 1),
                "hardware": gpu_name,
                "status": status,
            }
            rows.append(row)
            print(f"  [{cfg['name']}] Detections: {len(dets)} (Car:{cars}, Ped:{peds}, Cyc:{cycs}) | "
                  f"Latency p50: {p50_ms:.2f} ms, p95: {p95_ms:.2f} ms | Status: {status}")

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] Benchmark CSV saved to {out_csv}")

    # Plot 1: Scores and Detection Range Distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    scores_a = [d["score"] for d in all_detections_conf1]
    scores_b = [d["score"] for d in all_detections_conf2]
    ranges_a = [d["range"] for d in all_detections_conf1]
    ranges_b = [d["range"] for d in all_detections_conf2]

    # Histogram of scores
    axes[0].hist(scores_a, bins=15, alpha=0.6, color="royalblue", label=f"Conf A (thr=0.3, N={len(scores_a)})")
    axes[0].hist(scores_b, bins=15, alpha=0.7, color="crimson", label=f"Conf B (thr=0.5, N={len(scores_b)})")
    axes[0].axvline(0.3, color="blue", linestyle="--", linewidth=1.2, label="Threshold 0.3")
    axes[0].axvline(0.5, color="red", linestyle="--", linewidth=1.2, label="Threshold 0.5")
    axes[0].set_xlabel("Confidence Score")
    axes[0].set_ylabel("Detection Frequency")
    axes[0].set_title("PointPillars Detection Score Distribution")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Scatter: Distance vs Confidence Score
    classes_a = [d["name"] for d in all_detections_conf1]
    for cname, color in [("Car", "green"), ("Pedestrian", "orange"), ("Cyclist", "purple")]:
        c_ranges = [r for r, c in zip(ranges_a, classes_a) if c == cname]
        c_scores = [s for s, c in zip(scores_a, classes_a) if c == cname]
        axes[1].scatter(c_ranges, c_scores, label=cname, color=color, s=50, alpha=0.75, edgecolors="k")

    axes[1].set_xlabel("Distance from Ego LiDAR (meters)")
    axes[1].set_ylabel("Confidence Score")
    axes[1].set_title("Confidence Score vs Detection Distance")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plot1_path = out_dir / "detector_benchmark_scores.png"
    plt.savefig(plot1_path, dpi=150)
    plt.close()
    print(f"[OK] Score analysis plot saved to {plot1_path}")

    # Plot 2: Latency Analysis across frames
    fig, ax = plt.subplots(figsize=(10, 5))
    df_a = df[df["score_threshold"] == 0.3]
    x_pos = np.arange(len(df_a))
    width = 0.35

    ax.bar(x_pos - width / 2, df_a["latency_p50_ms"], width, label="Latency p50 (median)", color="#2b5c8f")
    ax.bar(x_pos + width / 2, df_a["latency_p95_ms"], width, label="Latency p95 (95th percentile)", color="#e07a5f")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Frame {fid}" for fid in df_a["frame_id"]])
    ax.set_ylabel("Latency (milliseconds)")
    ax.set_title(f"PointPillars Inference Latency Profile on {gpu_name}")
    ax.axhline(df_a["latency_p50_ms"].mean(), color="navy", linestyle=":", label=f"Mean p50 ({df_a['latency_p50_ms'].mean():.1f} ms)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    for i in x_pos:
        p50_val = df_a["latency_p50_ms"].iloc[i]
        p95_val = df_a["latency_p95_ms"].iloc[i]
        ax.text(i - width / 2, p50_val + 0.5, f"{p50_val:.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, p95_val + 0.5, f"{p95_val:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plot2_path = out_dir / "detector_latency_analysis.png"
    plt.savefig(plot2_path, dpi=150)
    plt.close()
    print(f"[OK] Latency profile plot saved to {plot2_path}")


if __name__ == "__main__":
    main()
