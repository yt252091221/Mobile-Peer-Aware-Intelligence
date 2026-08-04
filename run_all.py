"""One-click reproduction for the complete k3 comparison table.

The default run executes Ours, ByteTrack, BotSort, StrongSORT and DeepSORT on
all six k3 sequences, evaluates every result with the bundled TrackEval, and
writes one CSV/Markdown table including the TrackEval COMBINED row.  Ablation
experiments remain separate in ``run_ablation.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


PKG = Path(__file__).resolve().parent
K3_ROOT = PKG / "data"
TE_ROOT = PKG / "deps" / "TrackEval"
GT_ROOT = TE_ROOT / "data" / "gt" / "mot_challenge"
TRK_ROOT = TE_ROOT / "data" / "trackers" / "mot_challenge"
WEIGHTS = PKG / "weights"
OUTPUT = PKG / "output"
DEFAULT_OURS_WEIGHTS = (0.10, 0.10, 0.40, 0.40)
EXPECTED_BOXMOT_VERSION = "12.0.1"
COMBINED_BENCHMARK = "K3Combined"

SEQUENCES = [
    {"dir": "UAVSwarm-10-k3", "bench": "UAVSwarm10K3"},
    {"dir": "UAVSwarm-48-k3", "bench": "UAVSwarm48K3"},
    {"dir": "count3_k3", "bench": "Count3K3"},
    {"dir": "count2_k3", "bench": "Count2K3"},
    {"dir": "uavswarm61_k3", "bench": "UAVSwarm61K3"},
    {"dir": "watercraft_k3", "bench": "WatercraftK3"},
]


def installed_boxmot_version() -> str:
    try:
        import boxmot
        return str(getattr(boxmot, "__version__", "unknown"))
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}"


def setup_gt() -> None:
    """Create the bundled TrackEval GT structure for all six sequences."""
    for sequence in SEQUENCES:
        seq_dir = K3_ROOT / sequence["dir"]
        bench = sequence["bench"]
        seq = sequence["dir"]
        gt_dst = GT_ROOT / f"{bench}-train" / seq
        (gt_dst / "gt").mkdir(parents=True, exist_ok=True)
        (gt_dst / "det").mkdir(parents=True, exist_ok=True)
        (gt_dst / "img1").mkdir(parents=True, exist_ok=True)

        for name in ["gt.txt", "det.txt"]:
            src = seq_dir / name
            subdir = "gt" if name == "gt.txt" else "det"
            if not src.exists():
                raise FileNotFoundError(f"Missing k3 input: {src}")
            shutil.copy2(src, gt_dst / subdir / name)

        for image in (seq_dir / "img1").glob("*.jpg"):
            dst = gt_dst / "img1" / image.name
            if not dst.exists():
                try:
                    dst.symlink_to(image.resolve())
                except OSError:
                    shutil.copy2(image, dst)

        images = sorted((seq_dir / "img1").glob("*.jpg"))
        sample = cv2.imread(str(images[0])) if images else None
        height, width = sample.shape[:2] if sample is not None else (720, 1280)
        (gt_dst / "seqinfo.ini").write_text(
            f"[Sequence]\nname={seq}\nimDir=img1\nframeRate=30\n"
            f"seqLength={len(images)}\nimWidth={width}\nimHeight={height}\n"
            "imExt=.jpg\n",
            encoding="ascii",
        )

        seqmap = GT_ROOT / "seqmaps" / f"{bench}-train.txt"
        seqmap.parent.mkdir(parents=True, exist_ok=True)
        seqmap.write_text(f"name\n{seq}\n", encoding="ascii")

    print(f"[GT] {len(SEQUENCES)} sequences configured")


def setup_combined_gt() -> None:
    """Create the six-sequence benchmark used for TrackEval's COMBINED row."""
    combined_root = GT_ROOT / f"{COMBINED_BENCHMARK}-train"
    for sequence in SEQUENCES:
        seq = sequence["dir"]
        src = GT_ROOT / f"{sequence['bench']}-train" / seq
        dst = combined_root / seq
        (dst / "gt").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "gt" / "gt.txt", dst / "gt" / "gt.txt")
        shutil.copy2(src / "seqinfo.ini", dst / "seqinfo.ini")

    seqmap = GT_ROOT / "seqmaps" / f"{COMBINED_BENCHMARK}-train.txt"
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqmap.write_text(
        "name\n" + "\n".join(sequence["dir"] for sequence in SEQUENCES) + "\n",
        encoding="ascii",
    )
    print(f"[GT] {COMBINED_BENCHMARK}: {len(SEQUENCES)} sequences configured")


def run_ours(weights: list[float], run_dir: Path) -> None:
    print("\n[METHOD] Ours")
    command = [
        sys.executable,
        str(PKG / "scripts" / "run_ours_k3_paper_formula_weight_sweep.py"),
        "--k3-root",
        str(K3_ROOT),
        "--output",
        str(run_dir),
        "--weights",
        *(str(value) for value in weights),
    ]
    subprocess.run(command, cwd=str(PKG), check=True)


def run_boxmot(device: str, baseline_dir: Path) -> None:
    print("\n[METHOD] ByteTrack + BotSort + StrongSORT")
    command = [
        sys.executable,
        str(PKG / "scripts" / "run_boxmot_k3.py"),
        "--k3-root",
        str(K3_ROOT),
        "--methods",
        "ByteTrack",
        "BotSort",
        "StrongSORT",
        "--reid-weights",
        str(WEIGHTS / "osnet_x0_25_msmt17.pt"),
        "--device",
        device,
        "--output",
        str(baseline_dir / "boxmot"),
    ]
    subprocess.run(command, cwd=str(PKG), check=True)


def run_deepsort(device: str, baseline_dir: Path) -> None:
    print("\n[METHOD] DeepSORT")
    command = [
        sys.executable,
        str(PKG / "scripts" / "run_deepsort_k3.py"),
        "--k3-root",
        str(K3_ROOT),
        "--output",
        str(baseline_dir / "deepsort"),
        "--method-name",
        "DeepSORT",
    ]
    if device == "cpu":
        command.append("--no-cuda")
    subprocess.run(command, cwd=str(PKG), check=True)


def run_trackeval(bench: str, seq: str, tracker_name: str) -> dict[str, float | int]:
    command = [
        sys.executable,
        str(TE_ROOT / "scripts" / "run_mot_challenge.py"),
        "--BENCHMARK",
        bench,
        "--SPLIT_TO_EVAL",
        "train",
        "--DO_PREPROC",
        "False",
        "--GT_FOLDER",
        str(GT_ROOT),
        "--TRACKERS_FOLDER",
        str(TRK_ROOT),
        "--TRACKER_SUB_FOLDER",
        "data",
        "--METRICS",
        "HOTA",
        "CLEAR",
        "Identity",
        "--TRACKERS_TO_EVAL",
        tracker_name,
        "--USE_PARALLEL",
        "False",
        "--SEQ_INFO",
        seq,
    ]
    completed = subprocess.run(command, cwd=str(TE_ROOT), capture_output=True, text=True)
    if completed.returncode != 0:
        tail = (completed.stdout + "\n" + completed.stderr)[-3000:]
        raise RuntimeError(f"TrackEval failed for {tracker_name}/{seq}:\n{tail}")

    csv_path = TRK_ROOT / f"{bench}-train" / tracker_name / "pedestrian_detailed.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"TrackEval output not found: {csv_path}")
    with csv_path.open(encoding="utf-8", errors="ignore") as handle:
        rows = list(csv.DictReader(handle))
    row = next((item for item in rows if item.get("seq") == seq), None)
    if row is None:
        raise RuntimeError(f"TrackEval did not emit {seq} in {csv_path}")
    return {
        "HOTA": float(row["HOTA___AUC"]) * 100,
        "MOTA": float(row["MOTA"]) * 100,
        "IDF1": float(row["IDF1"]) * 100,
        "IDSW": int(float(row["IDSW"])),
    }


def method_sources(run_dir: Path) -> dict[str, Path]:
    baseline_dir = run_dir / "baselines"
    return {
        "Ours": run_dir / "Ours" / "data",
        "ByteTrack": baseline_dir / "boxmot" / "ByteTrack" / "data",
        "BotSort": baseline_dir / "boxmot" / "BotSort" / "data",
        "StrongSORT": baseline_dir / "boxmot" / "StrongSORT" / "data",
        "DeepSORT": baseline_dir / "deepsort" / "DeepSORT" / "data",
    }


def evaluate_all(run_dir: Path) -> list[dict]:
    """Evaluate every method on every sequence and fail on incomplete output."""
    print("\n[EVAL] Running TrackEval for all methods and sequences...")
    rows: list[dict] = []
    for method_name, source_dir in method_sources(run_dir).items():
        tracker_name = f"{method_name}_eval"
        print(f"  [{method_name}]")
        for sequence in SEQUENCES:
            seq = sequence["dir"]
            source = source_dir / f"{seq}.txt"
            if not source.exists():
                raise FileNotFoundError(f"Missing {method_name} output required for Table IV: {source}")

            destination = TRK_ROOT / f"{sequence['bench']}-train" / tracker_name / "data" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            metrics = run_trackeval(sequence["bench"], seq, tracker_name)
            print(
                f"    {seq}: HOTA={metrics['HOTA']:.3f}  MOTA={metrics['MOTA']:.3f}  "
                f"IDF1={metrics['IDF1']:.3f}  IDSW={metrics['IDSW']}"
            )
            rows.append({"Method": method_name, "Dataset": seq, **metrics})
    return rows


def evaluate_trackeval_combined(run_dir: Path) -> dict[str, dict[str, float | int]]:
    """Evaluate each method in TrackEval's six-sequence benchmark."""
    print("\n[EVAL] Running TrackEval COMBINED across all six sequences...")
    combined: dict[str, dict[str, float | int]] = {}
    for method_name, source_dir in method_sources(run_dir).items():
        tracker_name = f"{method_name}_combined_eval"
        destination_dir = TRK_ROOT / f"{COMBINED_BENCHMARK}-train" / tracker_name / "data"
        destination_dir.mkdir(parents=True, exist_ok=True)
        for sequence in SEQUENCES:
            source = source_dir / f"{sequence['dir']}.txt"
            if not source.exists():
                raise FileNotFoundError(f"Missing {method_name} output for COMBINED: {source}")
            shutil.copy2(source, destination_dir / source.name)

        command = [
            sys.executable,
            str(TE_ROOT / "scripts" / "run_mot_challenge.py"),
            "--BENCHMARK",
            COMBINED_BENCHMARK,
            "--SPLIT_TO_EVAL",
            "train",
            "--DO_PREPROC",
            "False",
            "--GT_FOLDER",
            str(GT_ROOT),
            "--TRACKERS_FOLDER",
            str(TRK_ROOT),
            "--TRACKER_SUB_FOLDER",
            "data",
            "--METRICS",
            "HOTA",
            "CLEAR",
            "Identity",
            "--TRACKERS_TO_EVAL",
            tracker_name,
            "--USE_PARALLEL",
            "False",
        ]
        completed = subprocess.run(command, cwd=str(TE_ROOT), capture_output=True, text=True)
        (run_dir / f"trackeval_{method_name}_combined_stdout.log").write_text(completed.stdout, encoding="utf-8")
        (run_dir / f"trackeval_{method_name}_combined_stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"TrackEval COMBINED failed for {method_name}")

        csv_path = TRK_ROOT / f"{COMBINED_BENCHMARK}-train" / tracker_name / "pedestrian_detailed.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"COMBINED output not found: {csv_path}")
        with csv_path.open(encoding="utf-8", errors="ignore") as handle:
            rows = list(csv.DictReader(handle))
        row = next((item for item in rows if item.get("seq") == "COMBINED"), None)
        if row is None:
            raise RuntimeError(f"TrackEval did not emit COMBINED for {method_name}")
        metrics = {
            "HOTA": float(row["HOTA___AUC"]) * 100,
            "MOTA": float(row["MOTA"]) * 100,
            "IDF1": float(row["IDF1"]) * 100,
            "IDSW": int(float(row["IDSW"])),
        }
        combined[method_name] = metrics
        print(
            f"  [{method_name}] HOTA={metrics['HOTA']:.3f}  MOTA={metrics['MOTA']:.3f}  "
            f"IDF1={metrics['IDF1']:.3f}  IDSW={metrics['IDSW']}"
        )
    return combined


def write_results(rows: list[dict], run_dir: Path, combined: dict[str, dict]) -> None:
    """Write one complete CSV and a readable Markdown version."""
    all_rows = rows + [
        {"Method": method, "Dataset": "COMBINED", **metrics}
        for method, metrics in combined.items()
    ]
    csv_out = run_dir / "results.csv"
    with csv_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Method", "Dataset", "HOTA", "MOTA", "IDF1", "IDSW"])
        writer.writeheader()
        writer.writerows(all_rows)

    combined_out = run_dir / "trackeval_combined.csv"
    with combined_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Method", "Dataset", "HOTA", "MOTA", "IDF1", "IDSW"])
        writer.writeheader()
        writer.writerows(all_rows[-len(combined):])

    labels = [sequence["dir"] for sequence in SEQUENCES]
    methods = list(method_sources(run_dir))
    markdown = ["# Complete k3 comparison", "", "Metrics are percentages; lower IDSW is better.", ""]
    for metric in ["HOTA", "MOTA", "IDF1", "IDSW"]:
        markdown.extend([
            f"## {metric}",
            "",
            "| Method | " + " | ".join(labels) + " |",
            "|---|" + "---:|" * len(labels),
        ])
        for method in methods:
            values = []
            for label in labels:
                row = next(item for item in rows if item["Method"] == method and item["Dataset"] == label)
                value = row[metric]
                values.append(str(value) if metric == "IDSW" else f"{value:.3f}")
            markdown.append(f"| {method} | " + " | ".join(values) + " |")
        markdown.append("")

    markdown.extend([
        "## TrackEval COMBINED (all six sequences)",
        "",
        "| Method | HOTA | MOTA | IDF1 | IDSW |",
        "|---|---:|---:|---:|---:|",
    ])
    for method in methods:
        metrics = combined[method]
        markdown.append(
            f"| {method} | {metrics['HOTA']:.3f} | {metrics['MOTA']:.3f} | "
            f"{metrics['IDF1']:.3f} | {metrics['IDSW']} |"
        )
    markdown.append("")
    md_out = run_dir / "results.md"
    md_out.write_text("\n".join(markdown), encoding="utf-8")
    print(f"\n[DONE] {csv_out}")
    print(f"[DONE] {md_out}")
    print(f"[DONE] {combined_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all k3 methods and generate the complete comparison table.")
    parser.add_argument(
        "--ours-weights",
        nargs=4,
        type=float,
        default=list(DEFAULT_OURS_WEIGHTS),
        metavar="W",
        help="distance, histogram, motion and watermark weights; all must be strictly positive",
    )
    parser.add_argument("--run-tag", default="reproduction_all", help="subdirectory under output/")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda:0"], default="auto")
    args = parser.parse_args()

    weights = np.asarray(args.ours_weights, dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        parser.error("--ours-weights must be finite and strictly positive")
    weights = (weights / float(weights.sum())).tolist()
    device = "cuda:0" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    run_dir = OUTPUT / args.run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    boxmot_version = installed_boxmot_version()
    if boxmot_version != EXPECTED_BOXMOT_VERSION:
        raise RuntimeError(
            f"This K3 package requires BoxMOT {EXPECTED_BOXMOT_VERSION}; "
            f"the active environment has {boxmot_version}."
        )
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "weights": weights,
                "terms": ["distance", "histogram", "motion", "watermark"],
                "methods": list(method_sources(run_dir)),
                "device": device,
                "boxmot_version": boxmot_version,
                "sequences": [sequence["dir"] for sequence in SEQUENCES],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    start = time.time()
    print(f"[PKG] {PKG}")
    print(f"[CONFIG] run_dir={run_dir}; device={device}; boxmot={boxmot_version}; weights={weights}")
    setup_gt()
    setup_combined_gt()
    run_ours(weights, run_dir)
    run_boxmot(device, run_dir / "baselines")
    run_deepsort(device, run_dir / "baselines")
    rows = evaluate_all(run_dir)
    combined = evaluate_trackeval_combined(run_dir)
    write_results(rows, run_dir, combined)
    print(f"[DONE] Total: {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
