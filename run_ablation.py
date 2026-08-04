"""
一键跑消融实验：6 个逐步加模块的变体 × 6 k3 序列 → TrackEval

用法：python run_ablation.py
"""
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
K3_ROOT = PKG / "data"
TE_ROOT = PKG / "deps" / "TrackEval"
GT_ROOT = TE_ROOT / "data" / "gt" / "mot_challenge"
TRK_ROOT = TE_ROOT / "data" / "trackers" / "mot_challenge"
OUTPUT = PKG / "output" / "ablation"

SEQUENCES = [
    {"dir": "UAVSwarm-10-k3", "bench": "UAVSwarm10K3"},
    {"dir": "UAVSwarm-48-k3", "bench": "UAVSwarm48K3"},
    {"dir": "count3_k3",      "bench": "Count3K3"},
    {"dir": "count2_k3",      "bench": "Count2K3"},
    {"dir": "uavswarm61_k3",  "bench": "UAVSwarm61K3"},
    {"dir": "watercraft_k3",  "bench": "WatercraftK3"},
]

ABLATION_MODES = ["base", "motion", "hist", "depth", "watermark", "complete"]


def setup_gt():
    """Create TrackEval GT structure (same as run_all.py)."""
    import cv2
    for s in SEQUENCES:
        seq_dir = K3_ROOT / s["dir"]
        bench = s["bench"]
        seq = s["dir"]
        gt_dst = GT_ROOT / f"{bench}-train" / seq
        (gt_dst / "gt").mkdir(parents=True, exist_ok=True)
        (gt_dst / "det").mkdir(parents=True, exist_ok=True)
        (gt_dst / "img1").mkdir(parents=True, exist_ok=True)
        for name in ["gt.txt", "det.txt"]:
            src = seq_dir / name
            sub = "gt" if name == "gt.txt" else "det"
            if src.exists():
                (gt_dst / sub / name).write_text(src.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        for img in (seq_dir / "img1").glob("*.jpg"):
            dst = gt_dst / "img1" / img.name
            if not dst.exists():
                try:
                    dst.symlink_to(img.resolve())
                except OSError:
                    import shutil
                    shutil.copy2(img, dst)
        imgs = sorted((seq_dir / "img1").glob("*.jpg"))
        n = len(imgs)
        sample = cv2.imread(str(imgs[0])) if imgs else None
        h, w = sample.shape[:2] if sample is not None else (720, 1280)
        (gt_dst / "seqinfo.ini").write_text(
            f"[Sequence]\nname={seq}\nimDir=img1\nframeRate=30\nseqLength={n}\n"
            f"imWidth={w}\nimHeight={h}\nimExt=.jpg\n", encoding="ascii")
        seqmap = GT_ROOT / "seqmaps" / f"{bench}-train.txt"
        seqmap.parent.mkdir(parents=True, exist_ok=True)
        seqmap.write_text(f"name\n{seq}\n", encoding="ascii")
    print(f"[GT] {len(SEQUENCES)} sequences configured")


def run_ablation_mode(mode: str):
    """Run one ablation mode on all sequences."""
    print(f"\n[ABLATION] mode={mode}")
    cmd = [
        sys.executable, str(PKG / "scripts" / "run_ours_k3.py"),
        "--k3-root", str(K3_ROOT),
        "--output", str(OUTPUT / mode),
        "--ablation-mode", mode,
    ]
    subprocess.run(cmd, cwd=str(PKG), check=True)


def evaluate_mode(mode: str):
    """Evaluate one ablation mode using TrackEval."""
    print(f"\n[EVAL] mode={mode}")
    src_dir = OUTPUT / mode / f"Ours_{mode}" / "data"
    trk_name = f"Ours_{mode}_eval"
    results = []
    for s in SEQUENCES:
        seq = s["dir"]
        bench = s["bench"]
        src = src_dir / f"{seq}.txt"
        if not src.exists():
            print(f"  {seq}: NO OUTPUT")
            results.append({"seq": seq, "HOTA": None, "MOTA": None, "IDF1": None, "IDSW": None})
            continue
        # Copy to TrackEval
        trk_dst = TRK_ROOT / f"{bench}-train" / trk_name / "data"
        trk_dst.mkdir(parents=True, exist_ok=True)
        (trk_dst / f"{seq}.txt").write_text(src.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        # Run TrackEval
        cmd = [
            sys.executable, str(TE_ROOT / "scripts" / "run_mot_challenge.py"),
            "--BENCHMARK", bench, "--SPLIT_TO_EVAL", "train", "--DO_PREPROC", "False",
            "--GT_FOLDER", str(GT_ROOT), "--TRACKERS_FOLDER", str(TRK_ROOT),
            "--TRACKER_SUB_FOLDER", "data", "--METRICS", "HOTA", "CLEAR", "Identity",
            "--TRACKERS_TO_EVAL", trk_name, "--USE_PARALLEL", "False", "--SEQ_INFO", seq,
        ]
        subprocess.run(cmd, cwd=str(TE_ROOT), capture_output=True, text=True)
        csv_path = TRK_ROOT / f"{bench}-train" / trk_name / "pedestrian_detailed.csv"
        if not csv_path.exists():
            print(f"  {seq}: NO CSV")
            results.append({"seq": seq, "HOTA": None, "MOTA": None, "IDF1": None, "IDSW": None})
            continue
        import csv as csv_mod
        with open(csv_path, encoding="utf-8", errors="ignore") as f:
            rows = list(csv_mod.DictReader(f))
        row = next((r for r in rows if r.get("seq") == seq), None)
        if row is None:
            print(f"  {seq}: NO ROW")
            results.append({"seq": seq, "HOTA": None, "MOTA": None, "IDF1": None, "IDSW": None})
            continue
        r = {
            "seq": seq,
            "HOTA": float(row["HOTA___AUC"]) * 100,
            "MOTA": float(row["MOTA"]) * 100,
            "IDF1": float(row["IDF1"]) * 100,
            "IDSW": int(float(row["IDSW"])),
        }
        print(f"  {seq}: HOTA={r['HOTA']:.3f} MOTA={r['MOTA']:.3f} IDF1={r['IDF1']:.3f} IDSW={r['IDSW']}")
        results.append(r)
    return results


def main():
    setup_gt()
    all_results = {}
    for mode in ABLATION_MODES:
        run_ablation_mode(mode)
        all_results[mode] = evaluate_mode(mode)

    # Print summary table
    print("\n" + "=" * 90)
    print("ABLATION STUDY RESULTS")
    print("=" * 90)
    seq_names = [s["dir"] for s in SEQUENCES]
    header = f"{'Mode':<12}" + "".join(f"{s:>14}" for s in seq_names)
    print(header)
    print("-" * 90)
    for mode in ABLATION_MODES:
        row = f"{mode:<12}"
        for r in all_results[mode]:
            if r["HOTA"] is not None:
                row += f"{r['HOTA']:>10.3f}/{r['IDSW']:<3}"
            else:
                row += f"{'N/A':>14}"
        print(row)
    print("=" * 90)

    # Save to CSV
    out_csv = OUTPUT / "ablation_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv as csv_mod
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv_mod.writer(f)
        writer.writerow(["Mode"] + [s["dir"] for s in SEQUENCES])
        for mode in ABLATION_MODES:
            row = [mode]
            for r in all_results[mode]:
                if r["HOTA"] is not None:
                    row.append(f"{r['HOTA']:.3f}")
                else:
                    row.append("N/A")
            writer.writerow(row)
    print(f"\nResults saved to {out_csv}")


if __name__ == "__main__":
    main()
