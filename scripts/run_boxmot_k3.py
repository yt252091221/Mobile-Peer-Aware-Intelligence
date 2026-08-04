"""
Run BoxMOT trackers directly on k3 datasets.

This script does not reuse existing TrackEval tracker outputs. It reads each
k3/<seq>/det.txt, feeds detections into BoxMOT tracker implementations frame by
frame, and writes MOT-format tracker outputs.

Supported live methods:
  - ByteTrack: BoxMOT ByteTrack
  - BotSort: BoxMOT BotSort with ReID disabled unless --reid-weights is set
  - StrongSORT: requires --reid-weights

TransTrack is not included in the supported BoxMOT API and needs a separate official
implementation/checkpoint.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


BBox = Tuple[float, float, float, float]


def read_mot(path: Path) -> Dict[int, List[Tuple[int, BBox, float]]]:
    frames: Dict[int, List[Tuple[int, BBox, float]]] = {}
    with path.open("r", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            frame = int(float(row[0]))
            tid = int(float(row[1]))
            box = tuple(float(row[i]) for i in range(2, 6))  # type: ignore[assignment]
            conf = float(row[6]) if len(row) > 6 else 1.0
            frames.setdefault(frame, []).append((tid, box, conf))
    return frames


def make_tracker(method: str, frame_rate: int, reid_weights: Optional[Path], device: torch.device):
    if method == "ByteTrack":
        from boxmot.trackers.bytetrack.bytetrack import ByteTrack

        return ByteTrack(min_conf=0.1, track_thresh=0.5, match_thresh=0.8, track_buffer=30, frame_rate=frame_rate)

    if method == "BotSort":
        from boxmot.trackers.botsort.botsort import BotSort

        return BotSort(
            reid_weights=reid_weights or Path(""),
            device=device,
            half=False,
            with_reid=False,
            track_high_thresh=0.5,
            track_low_thresh=0.1,
            new_track_thresh=0.6,
            track_buffer=30,
            match_thresh=0.8,
            proximity_thresh=0.5,
            appearance_thresh=0.25,
            cmc_method="ecc",
            frame_rate=frame_rate,
        )

    if method == "StrongSORT":
        if reid_weights is None:
            raise RuntimeError("StrongSORT requires --reid-weights for BoxMOT official implementation")
        from boxmot.trackers.strongsort.strongsort import StrongSort

        return StrongSort(reid_weights=reid_weights, device=device, half=False)

    raise ValueError(f"Unsupported method: {method}")


def detections_to_array(items: Sequence[Tuple[int, BBox, float]]) -> np.ndarray:
    rows = []
    for _, (x, y, w, h), conf in items:
        rows.append([x, y, x + w, y + h, conf, 0.0])
    return np.asarray(rows, dtype=np.float32) if rows else np.empty((0, 6), dtype=np.float32)


def output_to_items(output: np.ndarray) -> List[Tuple[int, BBox, float]]:
    if output is None or len(output) == 0:
        return []
    arr = np.asarray(output)
    items: List[Tuple[int, BBox, float]] = []
    for row in arr:
        # BoxMOT AABB output is x1,y1,x2,y2,id,conf,cls,det_ind.
        x1, y1, x2, y2 = map(float, row[:4])
        tid = int(row[4])
        conf = float(row[5]) if len(row) > 5 else 1.0
        items.append((tid, (x1, y1, x2 - x1, y2 - y1), conf))
    return items


def run_sequence(seq_dir: Path, method: str, out_dir: Path, reid_weights: Optional[Path], device: torch.device):
    det = read_mot(seq_dir / "det.txt")
    gt = read_mot(seq_dir / "gt.txt") if (seq_dir / "gt.txt").exists() else {}
    frame_count = max(max(det.keys(), default=0), max(gt.keys(), default=0), len(list((seq_dir / "img1").glob("*.jpg"))))
    tracker = make_tracker(method, frame_rate=30, reid_weights=reid_weights, device=device)
    pred: Dict[int, List[Tuple[int, BBox, float]]] = {}

    for frame_id in range(1, frame_count + 1):
        img_path = seq_dir / "img1" / f"frame_{frame_id:06d}.jpg"
        img = cv2.imread(str(img_path)) if img_path.exists() else None
        if img is None:
            img = np.zeros((1, 1, 3), dtype=np.uint8)
        det_arr = detections_to_array(det.get(frame_id, []))
        out = tracker.update(det_arr, img)
        pred[frame_id] = output_to_items(out)

    method_dir = out_dir / method / "data"
    method_dir.mkdir(parents=True, exist_ok=True)
    out_file = method_dir / f"{seq_dir.name}.txt"
    out_file.touch()
    with out_file.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        for frame_id in sorted(pred):
            for tid, (x, y, w, h), conf in pred[frame_id]:
                writer.writerow([frame_id, tid, f"{x:.2f}", f"{y:.2f}", f"{w:.2f}", f"{h:.2f}", f"{conf:.6f}", -1, -1, -1])

    return {"Method": method, "Sequence": seq_dir.name, "Output": str(out_file), "Frames": frame_count}


def print_table(rows: Sequence[dict]) -> None:
    header = f"{'Sequence':<16} {'Method':<10} {'Frames':>8}  Output"
    print(header)
    print("-" * len(header))
    for r in rows:
        if r.get("status") == "missing":
            print(f"{r['Sequence']:<16} {r['Method']:<10} {'MISSING':>8}  {r.get('reason','')}")
        else:
            print(f"{r['Sequence']:<16} {r['Method']:<10} {r['Frames']:>8}  {r['Output']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k3-root", default="k3")
    parser.add_argument("--methods", nargs="+", default=["ByteTrack", "BotSort", "StrongSORT"])
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--output", default="baseline_comparison_results/boxmot_k3_live")
    parser.add_argument("--reid-weights", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    k3_root = Path(args.k3_root)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    reid_weights = Path(args.reid_weights) if args.reid_weights else None
    device = torch.device(args.device)
    rows = []

    sequence_filter = set(args.sequences) if args.sequences else None
    for seq_dir in sorted(p for p in k3_root.iterdir() if p.is_dir()):
        if sequence_filter is not None and seq_dir.name not in sequence_filter:
            continue
        if not (seq_dir / "det.txt").exists() or not (seq_dir / "gt.txt").exists():
            continue
        for method in args.methods:
            if method == "TransTrack":
                rows.append({"Sequence": seq_dir.name, "Method": method, "status": "missing", "reason": "not provided by BoxMOT; needs separate official implementation/checkpoint"})
                continue
            try:
                print(f"Running {method} on {seq_dir.name}...")
                rows.append(run_sequence(seq_dir, method, out_dir, reid_weights, device))
            except Exception as exc:
                rows.append({"Sequence": seq_dir.name, "Method": method, "status": "missing", "reason": f"{type(exc).__name__}: {exc}"})

    print_table(rows)
    missing = [row for row in rows if row.get("status") == "missing"]
    with (out_dir / "boxmot_k3_live_results.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["Sequence", "Method", "Frames", "Output", "status", "reason"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})
    if missing:
        raise RuntimeError(
            "BoxMOT did not produce complete output; see "
            f"{out_dir / 'boxmot_k3_live_results.csv'}"
        )


if __name__ == "__main__":
    main()
