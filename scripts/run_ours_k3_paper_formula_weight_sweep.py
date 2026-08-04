from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pywt
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


BBoxXYWH = Tuple[float, float, float, float]
BBoxXYXY = Tuple[float, float, float, float]

TAU_IOU = 0.2
TAU_DEPTH = 0.10

TERNARY_DELTA = 0.15
TEMPORAL_WINDOW = 8
STABILITY_THRESHOLD = 0.75
IDENTITY_BIT_WEIGHT = 2.0
ENTROPY_TOP_K = 2

# Four-term association cost: spatial distance, HSV histogram, motion, watermark.
# Tuned all-active default; CLI --weights can override this.
DEFAULT_FEATURE_WEIGHTS = (0.10, 0.10, 0.40, 0.40)


def safe_roi(frame: np.ndarray, bbox: BBoxXYXY) -> np.ndarray | None:
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(frame.shape[1], int(bbox[2]))
    y2 = min(frame.shape[0], int(bbox[3]))
    if x1 >= x2 or y1 >= y2:
        return None
    roi = frame[y1:y2, x1:x2]
    return roi if roi.size > 0 else None


class MultiScaleWatermark:
    @staticmethod
    def extract(roi: np.ndarray) -> dict[str, np.ndarray] | None:
        try:
            resized = cv2.resize(roi, (64, 64))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float64)
            coeffs = pywt.wavedec2(gray, "db1", level=2)
            ll2 = coeffs[0]
            lh1 = cv2.resize(coeffs[2][0], (16, 16), interpolation=cv2.INTER_AREA)
            hl1 = cv2.resize(coeffs[2][1], (16, 16), interpolation=cv2.INTER_AREA)
            return {
                "LL2": MultiScaleWatermark._ternary_quantize(ll2).flatten(),
                "LH1": MultiScaleWatermark._ternary_quantize(lh1).flatten(),
                "HL1": MultiScaleWatermark._ternary_quantize(hl1).flatten(),
                "raw_LL2": ll2.flatten(),
                "raw_LH1": lh1.flatten(),
                "raw_HL1": hl1.flatten(),
            }
        except Exception:
            return None

    @staticmethod
    def _ternary_quantize(subband: np.ndarray) -> np.ndarray:
        median = np.median(subband)
        delta = TERNARY_DELTA * np.std(subband)
        result = np.zeros_like(subband, dtype=np.float32)
        result[subband > median + delta] = 1.0
        result[subband < median - delta] = -1.0
        return result

    @staticmethod
    def compute_similarity(desc_a: np.ndarray, desc_b: np.ndarray, stability_mask: np.ndarray | None = None) -> float:
        valid = (desc_a != 0) & (desc_b != 0)
        if valid.sum() == 0:
            return 0.5
        matches = desc_a[valid] == desc_b[valid]
        if stability_mask is not None and len(stability_mask) == len(desc_a):
            weights = np.ones(valid.sum(), dtype=np.float32)
            weights[stability_mask[valid]] = IDENTITY_BIT_WEIGHT
            return float((matches.astype(np.float32) * weights).sum() / max(weights.sum(), 1e-6))
        return float(matches.mean())


class TemporalWatermarkFingerprint:
    def __init__(self, window_size: int = TEMPORAL_WINDOW):
        self.history: deque[np.ndarray] = deque(maxlen=window_size)
        self.stability_mask: np.ndarray | None = None
        self.identity_signature: np.ndarray | None = None

    def update(self, descriptor: np.ndarray) -> None:
        self.history.append(descriptor.copy())
        if len(self.history) >= 3:
            self._update_stability()

    def _update_stability(self) -> None:
        stacked = np.array(list(self.history))
        pos_count = (stacked > 0).sum(axis=0)
        neg_count = (stacked < 0).sum(axis=0)
        nonzero_count = pos_count + neg_count
        dominant_positive = pos_count >= neg_count
        dominant_count = np.where(dominant_positive, pos_count, neg_count)
        stability = np.zeros(stacked.shape[1], dtype=np.float32)
        valid = nonzero_count > 0
        stability[valid] = dominant_count[valid].astype(float) / nonzero_count[valid]
        min_obs = max(2, len(self.history) // 2)
        self.stability_mask = (stability >= STABILITY_THRESHOLD) & (nonzero_count >= min_obs)
        self.identity_signature = np.zeros(stacked.shape[1], dtype=np.float32)
        self.identity_signature[self.stability_mask & dominant_positive] = 1.0
        self.identity_signature[self.stability_mask & ~dominant_positive] = -1.0


class TrackableObject:
    def __init__(self, object_id: int, centroid: np.ndarray, bbox: BBoxXYXY, frame: np.ndarray | None = None, watermark_mode: str = "simple"):
        self.objectID = object_id
        self.centroids = deque([centroid], maxlen=5_000_000)
        self.bbox = bbox
        self.watermark_mode = watermark_mode
        self.velocity = [0.0, 0.0]
        self.kalman = self._init_kalman()
        self.hist_features = None
        self.dwt_signature = None
        self.ms_watermark = None
        self.temporal_fp = {
            "LL2": TemporalWatermarkFingerprint(),
            "LH1": TemporalWatermarkFingerprint(),
            "HL1": TemporalWatermarkFingerprint(),
        }
        self.active_subbands = ["LL2", "LH1"]
        self.feature_history = deque(maxlen=10)
        self.overlap_state = "tracked"  # tracked / overlap_fg / overlap_bg
        self._update_kalman(centroid)
        if frame is not None:
            self.update_visual_features(frame, bbox)

    def _init_kalman(self) -> KalmanFilter:
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        kf.R *= 10
        kf.Q *= 0.1
        return kf

    def _update_kalman(self, centroid: np.ndarray) -> None:
        self.kalman.predict()
        self.kalman.update(np.array([centroid[0], centroid[1]]))
        state = self.kalman.x
        self.velocity = [float(state[2]), float(state[3])]

    def predicted_centroid(self) -> tuple[float, float]:
        state = self.kalman.x
        return (float(state[0] + state[2]), float(state[1] + state[3]))

    def predicted_bbox(self) -> BBoxXYXY:
        pred_x, pred_y = self.predicted_centroid()
        last_x, last_y = self.centroids[-1]
        dx = pred_x - float(last_x)
        dy = pred_y - float(last_y)
        x1, y1, x2, y2 = self.bbox
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

    def update(self, centroid: np.ndarray, bbox: BBoxXYXY, frame: np.ndarray | None = None) -> None:
        self.centroids.append(centroid)
        self.bbox = bbox
        self._update_kalman(centroid)
        if frame is not None:
            self.update_visual_features(frame, bbox)

    def update_visual_features(self, frame: np.ndarray, bbox: BBoxXYXY) -> None:
        roi = safe_roi(frame, bbox)
        if roi is None:
            return
        hist = self._compute_histogram(roi)
        dwt = self._compute_dwt_watermark(roi) if self.watermark_mode == "simple" else None
        if self.hist_features is not None:
            alpha = 0.7
            self.hist_features = alpha * self.hist_features + (1 - alpha) * hist
            if dwt is not None:
                self.dwt_signature = alpha * self.dwt_signature + (1 - alpha) * dwt
        else:
            self.hist_features = hist
            if dwt is not None:
                self.dwt_signature = dwt

        if self.watermark_mode == "temporal":
            ms_desc = MultiScaleWatermark.extract(roi)
            if ms_desc is not None:
                self.ms_watermark = ms_desc
                for band in ("LL2", "LH1", "HL1"):
                    self.temporal_fp[band].update(ms_desc[band])
                self._update_active_subbands(ms_desc)
        self.feature_history.append((hist, dwt if dwt is not None else self.ms_watermark))

    def _compute_histogram(self, roi: np.ndarray) -> np.ndarray:
        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            hist = hist / max(float(hist.sum()), 1e-6)
            return hist.flatten()
        except Exception:
            return np.zeros(256, dtype=np.float32)

    def _compute_dwt_watermark(self, roi: np.ndarray) -> np.ndarray:
        try:
            resized = cv2.resize(roi, (64, 64))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            coeffs = pywt.wavedec2(gray, "db1", level=2)
            ll = coeffs[0]
            return (ll > np.median(ll)).astype(np.float32).flatten()
        except Exception:
            return np.zeros(256, dtype=np.float32)

    def _update_active_subbands(self, ms_desc: dict[str, np.ndarray]) -> None:
        scores = {
            "LL2": float(np.var(ms_desc["raw_LL2"])),
            "LH1": float(np.var(ms_desc["raw_LH1"])),
            "HL1": float(np.var(ms_desc["raw_HL1"])),
        }
        self.active_subbands = sorted(scores.keys(), key=lambda band: scores[band], reverse=True)[:ENTROPY_TOP_K]

    def watermark_distance(self, roi: np.ndarray) -> float:
        if self.watermark_mode == "temporal":
            det_ms = MultiScaleWatermark.extract(roi)
            if self.ms_watermark is None or det_ms is None:
                return 0.5
            scores = []
            for band in self.active_subbands:
                sim = MultiScaleWatermark.compute_similarity(
                    self.ms_watermark[band],
                    det_ms[band],
                    self.temporal_fp[band].stability_mask,
                )
                scores.append(sim)
            return 1.0 - float(np.mean(scores)) if scores else 0.5

        if self.dwt_signature is None:
            return 0.5
        watermark = self._compute_dwt_watermark(roi)
        distance = float(np.linalg.norm(self.dwt_signature - watermark, ord=2))
        return 1.0 - 1.0 / (distance + 1.0)


class RobustMultiObjectTracker:
    def __init__(
        self,
        maxDisappeared: int = 300,
        maxDistance: float | None = None,
        maxObjects: int = 100,
        adaptive_ratio: float = 0.20,
        watermark_mode: str = "simple",
        feature_weights: Sequence[float] | None = None,
        output_visible_only: bool = True,
    ):
        self.nextObjectID = 0
        self.objects: Dict[int, TrackableObject] = {}
        self.disappeared: Dict[int, int] = {}
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance
        self.adaptive_ratio = float(adaptive_ratio)
        self.maxObjects = maxObjects
        self.watermark_mode = watermark_mode
        self.output_visible_only = output_visible_only
        weights = np.asarray(feature_weights if feature_weights is not None else DEFAULT_FEATURE_WEIGHTS, dtype=float)
        weights = weights / max(float(weights.sum()), 1e-6)
        if len(weights) == 5:
            self.feature_weights = {
                "distance": weights[0],
                "iou": weights[1],
                "histogram": weights[2],
                "motion": weights[3],
                "watermark": weights[4],
            }
        else:
            self.feature_weights = {
                "distance": weights[0],
                "iou": 0.0,
                "histogram": weights[1],
                "motion": weights[2],
                "watermark": weights[3],
            }

    def register(self, centroid: np.ndarray, bbox: BBoxXYXY, frame: np.ndarray | None = None) -> None:
        self.objects[self.nextObjectID] = TrackableObject(self.nextObjectID, centroid, bbox, frame, self.watermark_mode)
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, object_id: int) -> None:
        del self.objects[object_id]
        del self.disappeared[object_id]

    def _get_max_distance(self, frame: np.ndarray) -> float:
        if self.maxDistance is not None:
            return self.maxDistance
        h, w = frame.shape[:2]
        return self.adaptive_ratio * (w**2 + h**2) ** 0.5

    @staticmethod
    def _iou(box_a: BBoxXYXY, box_b: BBoxXYXY) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return float(inter / union)

    def _compute_depth_scores(self, bboxes: List[BBoxXYXY]) -> np.ndarray:
        """Area-based depth heuristic: larger bbox = closer = foreground."""
        areas = np.array([max(0, b[2]-b[0]) * max(0, b[3]-b[1]) for b in bboxes], dtype=float)
        total = areas.sum()
        if total < 1e-6:
            return np.ones(len(bboxes)) / max(len(bboxes), 1)
        return areas / total

    def _update_overlap_states(self, matched_ids: set) -> None:
        """Mark overlapping objects as foreground/background based on depth."""
        ids = [oid for oid in matched_ids if oid in self.objects]
        if len(ids) < 2:
            for oid in ids:
                self.objects[oid].overlap_state = "tracked"
            return

        bboxes = [self.objects[oid].bbox for oid in ids]
        depth_scores = self._compute_depth_scores(bboxes)
        overlap_bg_set = set()

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                iou = self._iou(bboxes[i], bboxes[j])
                depth_diff = abs(depth_scores[i] - depth_scores[j])
                if iou > TAU_IOU and depth_diff > TAU_DEPTH:
                    if depth_scores[i] > depth_scores[j]:
                        overlap_bg_set.add(ids[j])
                        self.objects[ids[i]].overlap_state = "overlap_fg"
                    else:
                        overlap_bg_set.add(ids[i])
                        self.objects[ids[j]].overlap_state = "overlap_fg"

        for oid in ids:
            if oid in overlap_bg_set:
                self.objects[oid].overlap_state = "overlap_bg"
            elif self.objects[oid].overlap_state != "overlap_fg":
                self.objects[oid].overlap_state = "tracked"

    def update(self, frame: np.ndarray, detections: Sequence[BBoxXYXY]) -> Dict[int, TrackableObject]:
        if len(detections) == 0:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.maxDisappeared:
                    self.deregister(oid)
            return {} if self.output_visible_only else self.objects

        max_dist = self._get_max_distance(frame)
        input_centroids = np.zeros((len(detections), 2), dtype=int)
        input_bboxes: List[BBoxXYXY] = []
        for i, (x1, y1, x2, y2) in enumerate(detections):
            input_centroids[i] = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            input_bboxes.append((x1, y1, x2, y2))

        if len(self.objects) == 0:
            visible_ids = []
            for i in range(min(len(detections), self.maxObjects)):
                self.register(input_centroids[i], input_bboxes[i], frame)
                visible_ids.append(self.nextObjectID - 1)
            return {oid: self.objects[oid] for oid in visible_ids} if self.output_visible_only else self.objects

        object_ids = list(self.objects.keys())
        object_centroids = np.array([self.objects[oid].centroids[-1] for oid in object_ids])
        predicted_centroids = np.array([self.objects[oid].predicted_centroid() for oid in object_ids], dtype=float)
        predicted_bboxes = [self.objects[oid].predicted_bbox() for oid in object_ids]
        d_spatial = cdist(object_centroids, input_centroids)
        d_min = float(d_spatial.min())
        d_max = float(d_spatial.max())
        d_spatial_norm = (d_spatial - d_min) / (d_max - d_min + 1e-6)
        d_motion = cdist(predicted_centroids, input_centroids)
        d_motion_norm = 1.0 - 1.0 / (1.0 + d_motion)
        d_iou = np.ones((len(object_ids), len(input_centroids)))
        for i, pred_bbox in enumerate(predicted_bboxes):
            for j, bbox in enumerate(input_bboxes):
                d_iou[i, j] = 1.0 - self._iou(pred_bbox, bbox)
        d_hist = np.ones((len(object_ids), len(input_centroids)))
        d_watermark = np.ones_like(d_hist)

        for i, oid in enumerate(object_ids):
            obj = self.objects[oid]
            if obj.hist_features is None:
                continue
            for j, bbox in enumerate(input_bboxes):
                roi = safe_roi(frame, bbox)
                if roi is None:
                    continue
                try:
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
                    hist = hist / max(float(hist.sum()), 1e-6)
                    hist = hist.flatten()
                    coefficient = float(np.sum(np.sqrt(np.clip(obj.hist_features * hist, 0.0, None))))
                    d_hist[i, j] = 1.0 - min(coefficient, 1.0)
                except Exception:
                    pass
                d_watermark[i, j] = obj.watermark_distance(roi)

        d_combined = (
            self.feature_weights["distance"] * d_spatial_norm
            + self.feature_weights["iou"] * d_iou
            + self.feature_weights["histogram"] * d_hist
            + self.feature_weights["motion"] * d_motion_norm
            + self.feature_weights["watermark"] * d_watermark
        )
        d_combined[d_spatial > max_dist] = float("inf")

        if np.all(np.isinf(d_combined)):
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.maxDisappeared:
                    self.deregister(oid)
            visible_ids = []
            for c in range(len(input_centroids)):
                if len(self.objects) < self.maxObjects:
                    self.register(input_centroids[c], input_bboxes[c], frame)
                    visible_ids.append(self.nextObjectID - 1)
            return {oid: self.objects[oid] for oid in visible_ids} if self.output_visible_only else self.objects

        rows, cols = linear_sum_assignment(d_combined)
        used_rows, used_cols = set(), set()
        visible_ids = set()
        for row, col in zip(rows, cols):
            if d_combined[row, col] == float("inf"):
                continue
            oid = object_ids[row]
            self.objects[oid].update(input_centroids[col], input_bboxes[col], frame)
            self.disappeared[oid] = 0
            visible_ids.add(oid)
            used_rows.add(row)
            used_cols.add(col)

        # Depth-based occlusion handling
        matched_ids = {object_ids[r] for r in used_rows}
        self._update_overlap_states(matched_ids)
        # Roll back features for occluded background objects
        for oid in matched_ids:
            obj = self.objects.get(oid)
            if obj and obj.overlap_state == "overlap_bg" and len(obj.feature_history) >= 2:
                prev_hist, prev_wm = obj.feature_history[-2]
                if prev_hist is not None:
                    obj.hist_features = prev_hist
                if prev_wm is not None:
                    obj.ms_watermark = prev_wm

        unused_rows = set(range(d_combined.shape[0])) - used_rows
        unused_cols = set(range(d_combined.shape[1])) - used_cols
        if d_combined.shape[0] >= d_combined.shape[1]:
            for row in unused_rows:
                oid = object_ids[row]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.maxDisappeared:
                    self.deregister(oid)
        else:
            for col in unused_cols:
                if len(self.objects) < self.maxObjects:
                    self.register(input_centroids[col], input_bboxes[col], frame)
                    visible_ids.add(self.nextObjectID - 1)

        return {oid: self.objects[oid] for oid in visible_ids} if self.output_visible_only else self.objects


def read_detections(path: Path, conf_thresh: float) -> Dict[int, List[Tuple[BBoxXYWH, float]]]:
    frames: Dict[int, List[Tuple[BBoxXYWH, float]]] = {}
    with path.open("r", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            frame = int(float(row[0]))
            conf = float(row[6]) if len(row) > 6 else 1.0
            if conf < conf_thresh:
                continue
            box = (float(row[2]), float(row[3]), float(row[4]), float(row[5]))
            frames.setdefault(frame, []).append((box, conf))
    return frames


def detections_to_xyxy(detections: Sequence[Tuple[BBoxXYWH, float]]) -> List[BBoxXYXY]:
    boxes = []
    for (x, y, w, h), _ in detections:
        boxes.append((x, y, x + w, y + h))
    return boxes


def run_sequence(
    seq_dir: Path,
    out_dir: Path,
    conf_thresh: float,
    max_disappeared: int,
    adaptive_ratio: float,
    max_distance: float | None,
    watermark_mode: str,
    method_name: str,
    feature_weights: Sequence[float],
    output_visible_only: bool,
) -> Path:
    detections = read_detections(seq_dir / "det.txt", conf_thresh)
    frame_count = max(max(detections.keys(), default=0), len(list((seq_dir / "img1").glob("*.jpg"))))
    tracker = RobustMultiObjectTracker(
        maxDisappeared=max_disappeared,
        maxDistance=max_distance,
        maxObjects=100,
        adaptive_ratio=adaptive_ratio,
        watermark_mode=watermark_mode,
        feature_weights=feature_weights,
        output_visible_only=output_visible_only,
    )

    method_dir = out_dir / method_name / "data"
    method_dir.mkdir(parents=True, exist_ok=True)
    out_file = method_dir / f"{seq_dir.name}.txt"
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        for frame_id in range(1, frame_count + 1):
            img_path = seq_dir / "img1" / f"frame_{frame_id:06d}.jpg"
            frame = cv2.imread(str(img_path)) if img_path.exists() else None
            if frame is None:
                frame = np.zeros((1, 1, 3), dtype=np.uint8)
            objects = tracker.update(frame, detections_to_xyxy(detections.get(frame_id, [])))
            for oid, obj in objects.items():
                x1, y1, x2, y2 = obj.bbox
                writer.writerow([frame_id, oid + 1, f"{x1:.2f}", f"{y1:.2f}", f"{x2 - x1:.2f}", f"{y2 - y1:.2f}", "1.000000", -1, -1, -1])
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k3-root", default="k3")
    parser.add_argument("--output", default="baseline_comparison_results/boxmot_k3_live_prev_params")
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--conf-thresh", type=float, default=0.0)
    parser.add_argument("--max-disappeared", type=int, default=300)
    parser.add_argument("--adaptive-ratio", type=float, default=0.20)
    parser.add_argument("--max-distance", type=float, default=None)
    parser.add_argument("--watermark-mode", choices=["simple", "temporal"], default="simple")
    parser.add_argument("--method-name", default="Ours")
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=list(DEFAULT_FEATURE_WEIGHTS),
        metavar="W",
        help=(
            "association weights. Use 4 values as distance histogram motion watermark, "
            "or 5 values as distance iou histogram motion watermark"
        ),
    )
    parser.add_argument("--output-all-tracks", action="store_true", help="write unmatched live tracks as well as currently matched detections")
    args = parser.parse_args()

    k3_root = Path(args.k3_root)
    out_dir = Path(args.output)
    if len(args.weights) not in (4, 5):
        parser.error("--weights expects either 4 values (distance histogram motion watermark) or 5 values (distance iou histogram motion watermark)")
    weights = np.asarray(args.weights, dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        parser.error("--weights must be finite and strictly positive so every enabled module remains active")
    normalized_weights = weights / float(weights.sum())
    print(
        "[CONFIG] association weights "
        + ", ".join(f"{name}={value:.6f}" for name, value in zip(
            ("distance", "histogram", "motion", "watermark") if len(weights) == 4
            else ("distance", "iou", "histogram", "motion", "watermark"),
            normalized_weights,
        ))
    )
    sequence_filter = set(args.sequences) if args.sequences else None
    for seq_dir in sorted(p for p in k3_root.iterdir() if p.is_dir()):
        if sequence_filter is not None and seq_dir.name not in sequence_filter:
            continue
        if not (seq_dir / "det.txt").exists():
            continue
        print(f"Running {args.method_name} on {seq_dir.name}...")
        out_file = run_sequence(
            seq_dir,
            out_dir,
            args.conf_thresh,
            args.max_disappeared,
            args.adaptive_ratio,
            args.max_distance,
            args.watermark_mode,
            args.method_name,
            normalized_weights,
            not args.output_all_tracks,
        )
        print(out_file)


if __name__ == "__main__":
    main()
