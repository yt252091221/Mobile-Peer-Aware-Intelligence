"""
Run DeepSort on k3 datasets using local deep_sort_pytorch implementation.

Usage:
  python run_deepsort_k3.py --k3-root data --output output/deepsort --sequences watercraft_k3
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Add local deep_sort to path
DS_ROOT = Path(__file__).resolve().parent.parent / "deps" / "deep_sort"
sys.path.insert(0, str(DS_ROOT.parent))

from deep_sort.deep_sort import DeepSort


BBoxXYWH = tuple[float, float, float, float]


def read_detections(path: Path, conf_thresh: float) -> dict[int, list[tuple[BBoxXYWH, float]]]:
    frames: dict[int, list[tuple[BBoxXYWH, float]]] = {}
    with path.open("r", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 7:
                continue
            frame = int(row[0])
            x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            conf = float(row[6])
            if conf < conf_thresh:
                continue
            frames.setdefault(frame, []).append(((x, y, w, h), conf))
    return frames


def run_sequence(
    seq_dir: Path,
    out_dir: Path,
    conf_thresh: float,
    reid_model: str,
    use_cuda: bool,
    method_name: str = "DeepSort",
) -> Path:
    detections = read_detections(seq_dir / "det.txt", conf_thresh)
    img_dir = seq_dir / "img1"
    imgs = sorted(img_dir.glob("*.jpg"))
    frame_count = max(max(detections.keys(), default=0), len(imgs))

    # Init DeepSort
    deepsort = DeepSort(
        model_path=reid_model,
        max_dist=0.2,
        min_confidence=conf_thresh,
        nms_max_overlap=1.0,
        max_iou_distance=0.7,
        max_age=70,
        n_init=1,
        nn_budget=100,
        use_cuda=use_cuda,
    )

    method_dir = out_dir / method_name / "data"
    method_dir.mkdir(parents=True, exist_ok=True)
    out_file = method_dir / f"{seq_dir.name}.txt"

    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        for frame_idx in range(1, frame_count + 1):
            # Read image (try both naming conventions)
            img_path = img_dir / f"{frame_idx:06d}.jpg"
            if not img_path.exists():
                img_path = img_dir / f"frame_{frame_idx:06d}.jpg"
            if img_path.exists():
                img = cv2.imread(str(img_path))
            else:
                img = np.zeros((720, 1280, 3), dtype=np.uint8)

            # Get detections for this frame, filter valid bboxes
            dets = detections.get(frame_idx, [])
            h_img, w_img = img.shape[:2]
            valid_dets = []
            for bbox, conf in dets:
                x, y, w, h = bbox
                # Clip to image bounds
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(w_img, x + w)
                y2 = min(h_img, y + h)
                if x2 > x1 + 1 and y2 > y1 + 1:
                    valid_dets.append(((x1, y1, x2 - x1, y2 - y1), conf))

            if valid_dets:
                # Convert from tlwh (x,y,w,h) to center-based xywh for DeepSort
                raw_boxes = np.array([d[0] for d in valid_dets], dtype=np.float32)
                xc = raw_boxes[:, 0] + raw_boxes[:, 2] / 2.0
                yc = raw_boxes[:, 1] + raw_boxes[:, 3] / 2.0
                bboxes_xywh = np.stack([xc, yc, raw_boxes[:, 2], raw_boxes[:, 3]], axis=1)
                confidences = np.array([d[1] for d in valid_dets], dtype=np.float32)
                classes = np.zeros(len(valid_dets), dtype=int)
            else:
                bboxes_xywh = np.empty((0, 4), dtype=np.float32)
                confidences = np.empty(0, dtype=np.float32)
                classes = np.empty(0, dtype=int)

            # Run DeepSort: returns (outputs, mask_outputs).
            # outputs: ndarray of [x1, y1, x2, y2, track_cls, track_id]
            outputs, _ = deepsort.update(bboxes_xywh, confidences, classes, img)

            if outputs is not None and len(outputs) > 0:
                for det in outputs:
                    x1, y1, x2, y2 = det[:4]
                    track_id = int(det[5])  # track_id is at index 5
                    w = x2 - x1
                    h = y2 - y1
                    writer.writerow([frame_idx, int(track_id), f"{x1:.3f}", f"{y1:.3f}",
                                     f"{w:.3f}", f"{h:.3f}", "1", "-1", "-1", "-1"])

    print(f"  {seq_dir.name}: {out_file}")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Run DeepSort on k3 datasets")
    parser.add_argument("--k3-root", required=True)
    parser.add_argument("--output", default="output/deepsort")
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--conf-thresh", type=float, default=0.0)
    parser.add_argument("--reid-model", default=str(DS_ROOT / "deep" / "checkpoint" / "ckpt.t7"))
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--method-name", default="DeepSORT")
    args = parser.parse_args()

    k3_root = Path(args.k3_root)
    out_dir = Path(args.output)
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()

    seq_filter = set(args.sequences) if args.sequences else None
    for seq_dir in sorted(p for p in k3_root.iterdir() if p.is_dir()):
        if seq_filter and seq_dir.name not in seq_filter:
            continue
        if not (seq_dir / "det.txt").exists():
            continue
        print(f"Running DeepSort on {seq_dir.name}...")
        run_sequence(seq_dir, out_dir, args.conf_thresh, args.reid_model, use_cuda, args.method_name)


if __name__ == "__main__":
    main()
