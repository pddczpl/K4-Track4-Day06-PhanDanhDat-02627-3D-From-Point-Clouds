"""PointPillars 3D Object Detection Model for LiDAR.
Standalone implementation in PyTorch supporting NVIDIA RTX 3060 GPU acceleration.
Compatible with official MMDetection3D hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class weights.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as tv_ops

from starter.kitti_io import KittiCalib, KittiObject


CLASS_NAMES = ["Pedestrian", "Cyclist", "Car"]
CLASS_COLORS = {
    "Pedestrian": (255, 100, 100),  # Blue-ish in BGR
    "Cyclist": (0, 200, 255),        # Yellow in BGR
    "Car": (0, 255, 0),              # Green in BGR
}
POINT_CLOUD_RANGE = [0.0, -39.68, -3.0, 69.12, 39.68, 1.0]
VOXEL_SIZE = [0.16, 0.16, 4.0]
GRID_SIZE_Y = 496
GRID_SIZE_X = 432
FEATURE_MAP_Y = 248
FEATURE_MAP_X = 216


def generate_aligned_anchors(device: torch.device) -> torch.Tensor:
    """Generate 3D anchors matching AlignedAnchor3DRangeGenerator.
    Returns: (FEATURE_MAP_Y, FEATURE_MAP_X, num_anchors=6, 7)
    where 7 = [x, y, z, w, l, h, r]
    Anchor mapping:
        0, 1: Pedestrian (rot 0, 1.57)
        2, 3: Cyclist    (rot 0, 1.57)
        4, 5: Car        (rot 0, 1.57)
    """
    ranges = [
        [0.0, -39.68, -0.6, 69.12, 39.68, -0.6],
        [0.0, -39.68, -0.6, 69.12, 39.68, -0.6],
        [0.0, -39.68, -1.78, 69.12, 39.68, -1.78]
    ]
    sizes = [
        [0.8, 0.6, 1.73],
        [1.76, 0.6, 1.73],
        [3.9, 1.6, 1.56]
    ]
    rotations = [0.0, 1.5707963]

    mr_anchors = []
    for r_cfg, s_cfg in zip(ranges, sizes):
        zc = torch.linspace(r_cfg[2], r_cfg[5], 2, device=device)
        yc = torch.linspace(r_cfg[1], r_cfg[4], FEATURE_MAP_Y + 1, device=device)
        xc = torch.linspace(r_cfg[0], r_cfg[3], FEATURE_MAP_X + 1, device=device)
        zc += (zc[1] - zc[0]) / 2
        yc += (yc[1] - yc[0]) / 2
        xc += (xc[1] - xc[0]) / 2
        rots = torch.tensor(rotations, device=device)

        xx, yy, zz, rr = torch.meshgrid(xc[:FEATURE_MAP_X], yc[:FEATURE_MAP_Y], zc[:1], rots, indexing='ij')
        ss = torch.tensor(s_cfg, device=device).view(1, 1, 1, 1, 3).repeat(FEATURE_MAP_X, FEATURE_MAP_Y, 1, 2, 1)

        anc = torch.cat([xx.unsqueeze(-1), yy.unsqueeze(-1), zz.unsqueeze(-1), ss, rr.unsqueeze(-1)], dim=-1)
        anc = anc.permute([2, 1, 0, 3, 4]).squeeze(0)  # (FEATURE_MAP_Y, FEATURE_MAP_X, 2, 7)
        mr_anchors.append(anc)

    return torch.cat(mr_anchors, dim=2)  # (248, 216, 6, 7)


class SECONDFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.deblocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(64, 128, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, bias=False),
                nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=4, bias=False),
                nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ),
        ])

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        ups = [deblock(feat) for deblock, feat in zip(self.deblocks, features)]
        return torch.cat(ups, dim=1)  # 128*3 = 384 channels


class SECOND(nn.Module):
    def __init__(self):
        super().__init__()
        b0 = []
        for i in range(4):
            stride = 2 if i == 0 else 1
            b0.extend([
                nn.Conv2d(64, 64, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ])
        b1 = []
        for i in range(6):
            c_in = 64 if i == 0 else 128
            stride = 2 if i == 0 else 1
            b1.extend([
                nn.Conv2d(c_in, 128, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ])
        b2 = []
        for i in range(6):
            c_in = 128 if i == 0 else 256
            stride = 2 if i == 0 else 1
            b2.extend([
                nn.Conv2d(c_in, 256, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ])
        self.blocks = nn.ModuleList([
            nn.Sequential(*b0),
            nn.Sequential(*b1),
            nn.Sequential(*b2),
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs = []
        for block in self.blocks:
            x = block(x)
            outs.append(x)
        return outs


class Anchor3DHead(nn.Module):
    def __init__(self, in_channels: int = 384, num_classes: int = 3, num_anchors: int = 6):
        super().__init__()
        self.conv_cls = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=1)
        self.conv_reg = nn.Conv2d(in_channels, num_anchors * 7, kernel_size=1)
        self.conv_dir_cls = nn.Conv2d(in_channels, num_anchors * 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        dir_cls_pred = self.conv_dir_cls(x)
        return cls_score, bbox_pred, dir_cls_pred


def voxelize_pillars(points: np.ndarray, max_points_per_pillar: int = 32, max_pillars: int = 30000):
    """Vectorized pillar voxelization.
    Points: (N, 4) [x, y, z, reflectance]
    Returns:
        voxels: (M, max_points_per_pillar, 4) float32
        num_points: (M,) int32
        coors: (M, 4) int32 [batch, z, y, x]
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    mask = (x >= POINT_CLOUD_RANGE[0]) & (x < POINT_CLOUD_RANGE[3]) & \
           (y >= POINT_CLOUD_RANGE[1]) & (y < POINT_CLOUD_RANGE[4]) & \
           (z >= POINT_CLOUD_RANGE[2]) & (z < POINT_CLOUD_RANGE[5]) & \
           np.isfinite(points).all(axis=1)
    pts = points[mask]
    if len(pts) == 0:
        return (np.zeros((0, max_points_per_pillar, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0, 4), dtype=np.int32))

    x_idx = np.floor((pts[:, 0] - POINT_CLOUD_RANGE[0]) / VOXEL_SIZE[0]).astype(np.int64)
    y_idx = np.floor((pts[:, 1] - POINT_CLOUD_RANGE[1]) / VOXEL_SIZE[1]).astype(np.int64)
    x_idx = np.clip(x_idx, 0, GRID_SIZE_X - 1)
    y_idx = np.clip(y_idx, 0, GRID_SIZE_Y - 1)

    grid_keys = y_idx * GRID_SIZE_X + x_idx
    unq_keys = np.unique(grid_keys)
    if len(unq_keys) > max_pillars:
        unq_keys = unq_keys[:max_pillars]

    M = len(unq_keys)
    voxels = np.zeros((M, max_points_per_pillar, 4), dtype=np.float32)
    num_points = np.zeros((M,), dtype=np.int32)
    coors = np.zeros((M, 4), dtype=np.int32)

    key_map = {k: i for i, k in enumerate(unq_keys)}
    for i, k in enumerate(grid_keys):
        pidx = key_map.get(k)
        if pidx is not None and num_points[pidx] < max_points_per_pillar:
            voxels[pidx, num_points[pidx]] = pts[i]
            num_points[pidx] += 1

    for i, k in enumerate(unq_keys):
        coors[i, 0] = 0
        coors[i, 1] = 0
        coors[i, 2] = k // GRID_SIZE_X
        coors[i, 3] = k % GRID_SIZE_X

    return voxels, num_points, coors


class PointPillarsDetector:
    def __init__(self, checkpoint_path: str | Path, device: str | torch.device = "cuda"):
        if isinstance(device, str):
            device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.device = device

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.sd = ckpt.get("state_dict", ckpt)

        # Build submodules
        self.backbone = SECOND().to(self.device)
        self.neck = SECONDFPN().to(self.device)
        self.head = Anchor3DHead(in_channels=384, num_classes=3, num_anchors=6).to(self.device)

        # Load weights
        self.backbone.load_state_dict({k[len("backbone."):]: v for k, v in self.sd.items() if k.startswith("backbone.")}, strict=True)
        self.neck.load_state_dict({k[len("neck."):]: v for k, v in self.sd.items() if k.startswith("neck.")}, strict=True)
        self.head.load_state_dict({k[len("bbox_head."):]: v for k, v in self.sd.items() if k.startswith("bbox_head.")}, strict=True)

        # PFE parameters
        self.lin_w = self.sd["voxel_encoder.pfn_layers.0.linear.weight"].to(self.device)
        self.norm_w = self.sd["voxel_encoder.pfn_layers.0.norm.weight"].to(self.device)
        self.norm_b = self.sd["voxel_encoder.pfn_layers.0.norm.bias"].to(self.device)
        self.norm_rm = self.sd["voxel_encoder.pfn_layers.0.norm.running_mean"].to(self.device)
        self.norm_rv = self.sd["voxel_encoder.pfn_layers.0.norm.running_var"].to(self.device)

        self.backbone.eval()
        self.neck.eval()
        self.head.eval()

        self.anchors = generate_aligned_anchors(self.device)
        self.anchor_cls = torch.tensor([0, 0, 1, 1, 2, 2], device=self.device)

    @torch.no_grad()
    def predict(
        self,
        points: np.ndarray,
        score_thr: float = 0.3,
        nms_iou_thr: float = 0.1,
        nms_thr: float | None = None,
    ) -> list[dict]:
        """Inference on point cloud array (N, 4).
        Returns list of detection dicts:
            name: 'Car' | 'Pedestrian' | 'Cyclist'
            class_id: 0 | 1 | 2
            score: float
            box3d: [x, y, z, dx, dy, dz, yaw] in LiDAR frame (bottom-center)
            range: distance from sensor in meters
        """
        if nms_thr is not None:
            nms_iou_thr = nms_thr
        voxels, num_points, coors = voxelize_pillars(points)
        if len(voxels) == 0:
            return []

        voxels_t = torch.from_numpy(voxels).to(self.device)
        num_pts_t = torch.from_numpy(num_points).to(self.device)
        coors_t = torch.from_numpy(coors).to(self.device)

        vx, vy, vz = VOXEL_SIZE
        x_offset = vx / 2 + POINT_CLOUD_RANGE[0]
        y_offset = vy / 2 + POINT_CLOUD_RANGE[1]
        z_offset = vz / 2 + POINT_CLOUD_RANGE[2]

        pts_mean = voxels_t[:, :, :3].sum(dim=1, keepdim=True) / num_pts_t.view(-1, 1, 1).clamp(min=1)
        f_cluster = voxels_t[:, :, :3] - pts_mean
        f_center = voxels_t[:, :, :3].clone()
        f_center[:, :, 0] -= (coors_t[:, 3].unsqueeze(1) * vx + x_offset)
        f_center[:, :, 1] -= (coors_t[:, 2].unsqueeze(1) * vy + y_offset)
        f_center[:, :, 2] -= (coors_t[:, 1].unsqueeze(1) * vz + z_offset)

        features_mod = voxels_t.clone()
        features_mod[:, :, :3] = f_center
        in_feat = torch.cat([features_mod, f_cluster, f_center], dim=-1)
        in_feat *= (torch.arange(32, device=self.device).unsqueeze(0) < num_pts_t.unsqueeze(1)).unsqueeze(-1)

        x = F.linear(in_feat, self.lin_w)
        x = (x - self.norm_rm.view(1, 1, 64)) / torch.sqrt(self.norm_rv.view(1, 1, 64) + 1e-3) * self.norm_w.view(1, 1, 64) + self.norm_b.view(1, 1, 64)
        pfe_out = torch.max(F.relu(x), dim=1)[0]

        canvas = torch.zeros((1, 64, GRID_SIZE_Y, GRID_SIZE_X), device=self.device)
        canvas[0, :, coors_t[:, 2].long(), coors_t[:, 3].long()] = pfe_out.t()

        cls_score, bbox_pred, dir_cls_pred = self.head(self.neck(self.backbone(canvas)))

        cls_pred = cls_score.squeeze(0).permute(1, 2, 0).view(FEATURE_MAP_Y, FEATURE_MAP_X, 6, 3)
        reg_pred = bbox_pred.squeeze(0).permute(1, 2, 0).view(FEATURE_MAP_Y, FEATURE_MAP_X, 6, 7)
        dir_pred = dir_cls_pred.squeeze(0).permute(1, 2, 0).view(FEATURE_MAP_Y, FEATURE_MAP_X, 6, 2)

        probs = torch.sigmoid(cls_pred)
        scores = torch.gather(probs, -1, self.anchor_cls.view(1, 1, 6, 1).expand(FEATURE_MAP_Y, FEATURE_MAP_X, -1, 1)).squeeze(-1)

        keep = scores > score_thr
        if not keep.any():
            return []

        sc = scores[keep]
        del_box = reg_pred[keep]
        anc_box = self.anchors[keep]
        dirs = dir_pred[keep]
        clses = self.anchor_cls.view(1, 1, 6).expand(FEATURE_MAP_Y, FEATURE_MAP_X, 6)[keep]

        xa, ya, za, wa, la, ha, ra = torch.split(anc_box, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt = torch.split(del_box, 1, dim=-1)
        za_c = za + ha / 2
        diag = torch.sqrt(la**2 + wa**2)
        xg = xt * diag + xa
        yg = yt * diag + ya
        zg_c = zt * ha + za_c
        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
        rg = rt + ra
        zg = zg_c - hg / 2

        dir_labels = torch.argmax(dirs, dim=-1, keepdim=True)
        rg = rg % (2 * math.pi)
        rg = torch.where(dir_labels == 1, rg + math.pi, rg)
        rg = (rg + math.pi) % (2 * math.pi) - math.pi

        boxes3d = torch.cat([xg, yg, zg, wg, lg, hg, rg], dim=-1)

        r_bev = torch.max(wg, lg) / 2
        boxes2d = torch.cat([xg - r_bev, yg - r_bev, xg + r_bev, yg + r_bev], dim=-1)
        nms_idx = tv_ops.batched_nms(boxes2d, sc, clses, nms_iou_thr)

        results = []
        for idx in nms_idx:
            cid = int(clses[idx].item())
            b = boxes3d[idx].cpu().numpy().tolist()
            dist = math.sqrt(b[0] ** 2 + b[1] ** 2)
            results.append({
                "name": CLASS_NAMES[cid],
                "class_id": cid,
                "score": float(sc[idx].item()),
                "box3d": b,  # [x, y, z, dx, dy, dz, yaw]
                "range": dist,
            })
        return results


def box3d_to_corners_lidar(box3d: list[float]) -> np.ndarray:
    """Convert LiDAR box [x, y, z, w, l, h, yaw] (bottom center)
    to 8 corners (8, 3) in LiDAR coordinates.
    """
    x, y, z, w, l, h, yaw = box3d
    # 8 local corners relative to bottom center
    x_corners = l / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = np.array([0, 0, 0, 0, h, h, h, h])

    rot_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    corners = rot_matrix @ np.vstack([x_corners, y_corners, z_corners])
    corners[0, :] += x
    corners[1, :] += y
    corners[2, :] += z
    return corners.T


def project_box3d_lidar_to_camera(corners_lidar: np.ndarray, calib: KittiCalib, image_shape: tuple[int, ...]):
    """Project 8 LiDAR corners to camera image coordinates."""
    pts_homo = np.hstack([corners_lidar, np.ones((len(corners_lidar), 1))])
    pts_cam = pts_homo @ calib.T_cam_velo.T
    if (pts_cam[:, 2] <= 0.1).any():
        return None  # Behind camera

    pts_img_homo = pts_cam @ calib.P2.T
    u = pts_img_homo[:, 0] / pts_img_homo[:, 2]
    v = pts_img_homo[:, 1] / pts_img_homo[:, 2]
    H, W = image_shape[:2]
    if (u < -100).any() or (u > W + 100).any() or (v < -100).any() or (v > H + 100).any():
        return None
    return np.column_stack([u, v])


def draw_box3d_on_image(image: np.ndarray, corners_2d: np.ndarray, color=(0, 255, 0), label: str = None) -> np.ndarray:
    """Draw 3D bounding box wireframe on image."""
    out = image.copy()
    pts = corners_2d.astype(int)

    # Bottom lines
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        cv2.line(out, tuple(pts[i]), tuple(pts[j]), color, 2)
    # Top lines
    for i, j in [(4, 5), (5, 6), (6, 7), (7, 4)]:
        cv2.line(out, tuple(pts[i]), tuple(pts[j]), color, 2)
    # Vertical lines
    for i, j in [(0, 4), (1, 5), (2, 6), (3, 7)]:
        cv2.line(out, tuple(pts[i]), tuple(pts[j]), color, 2)

    if label:
        min_u = int(pts[:, 0].min())
        min_v = int(pts[:, 1].min())
        cv2.putText(out, label, (max(0, min_u), max(15, min_v - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


def visualize_bev(points: np.ndarray, detections: list[dict], out_path: str | Path,
                  pc_range: list[float] = [0.0, -35.0, 65.0, 35.0]):
    """Render Bird's Eye View (BEV) visualization with LiDAR points and 3D bounding boxes."""
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="black")
    ax.set_facecolor("black")

    # Filter points for BEV
    mask = (points[:, 0] >= pc_range[0]) & (points[:, 0] <= pc_range[2]) & \
           (points[:, 1] >= pc_range[1]) & (points[:, 1] <= pc_range[3])
    pts = points[mask]

    # Draw point cloud
    dist = np.linalg.norm(pts[:, :2], axis=1)
    ax.scatter(pts[:, 1], pts[:, 0], s=0.3, c=dist, cmap="viridis", alpha=0.6)

    # Draw detection boxes
    for det in detections:
        b = det["box3d"]
        corners = box3d_to_corners_lidar(b)  # (8, 3)
        bottom = corners[:4, :2]  # (x, y)
        # In BEV plot: x is lateral (pts[:, 1]), y is longitudinal (pts[:, 0])
        poly = np.vstack([bottom[[0, 1, 2, 3, 0], 1], bottom[[0, 1, 2, 3, 0], 0]]).T

        color = "lime" if det["name"] == "Car" else "cyan" if det["name"] == "Pedestrian" else "yellow"
        ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=1.8)
        ax.text(b[1], b[0], f"{det['name']} {det['score']:.2f}", color="white", fontsize=8,
                bbox=dict(boxstyle="square,pad=0.1", fc=color, ec="none", alpha=0.6))

    ax.set_xlim(pc_range[1], pc_range[3])
    ax.set_ylim(pc_range[0], pc_range[2])
    ax.set_xlabel("Lateral Y (meters)", color="white")
    ax.set_ylabel("Forward X (meters)", color="white")
    ax.set_title(f"LiDAR 3D Object Detection BEV — Detections: {len(detections)}", color="white")
    ax.tick_params(colors="white")
    ax.grid(True, color="#333333", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, facecolor=fig.get_facecolor())
    plt.close()
