# K3 Reproduction Package

This repository contains the final reproducible package for the K3 multi-object
tracking comparison. It includes the six K3 datasets, the final online tracker
runners, the local DeepSORT implementation and checkpoint, the bundled TrackEval
source, and the Ours implementation.

The package is intentionally distributed without generated tables, tracker
outputs, logs, caches, or exploratory figure/weight-sweep scripts. Running the
commands below creates those files locally under `output/`.

All runtime roots are resolved from the location of the entrypoint scripts, so
the repository can be cloned or moved to another drive and directory without
editing machine-specific absolute paths.

## One-click full comparison

Use Python 3.10 or a compatible Python 3 environment. Install Git LFS before
cloning or downloading the repository because the images and model checkpoints
are large binary assets.

```powershell
git lfs install
python -m pip install -r requirements.txt

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "1"
python runall.py --device auto
```

`runall.py` is an alias for `run_all.py`. The default run evaluates the
following methods on all six datasets and then runs the bundled TrackEval
`COMBINED` evaluation over all six sequences:

- Ours
- ByteTrack
- BotSort
- StrongSORT
- DeepSORT

The main outputs are written to:

```text
output/reproduction_all/results.csv
output/reproduction_all/results.md
output/reproduction_all/trackeval_combined.csv
```

Use `--device cpu` for a CPU-only run or `--device cuda:0` for the tested CUDA
path. To reproduce the final Ours setting explicitly:

```powershell
python runall.py --device auto --ours-weights 0.10 0.10 0.40 0.40
```

The four weights are ordered as `distance`, `histogram`, `motion`, and
`watermark`. They must all be strictly positive; the runner normalizes them
before execution.

## Ablation (separate run)

The ablation study is not part of the main comparison command:

```powershell
python run_ablation.py
```

Its outputs are written to `output/ablation/`.

## Reproduction details

- `data/` contains the six K3 datasets used by the final scripts.
- `scripts/run_ours_k3_paper_formula_weight_sweep.py` is the Ours runner used
  by the main comparison entrypoint.
- `scripts/run_boxmot_k3.py` runs ByteTrack, BotSort, and StrongSORT through
  the installed BoxMOT package using the final K3 configuration.
- `scripts/run_deepsort_k3.py` uses the bundled local DeepSORT code and
  `deps/deep_sort/deep/checkpoint/ckpt.t7`.
- `deps/TrackEval/` contains the TrackEval source only. Its `data/gt/`,
  `data/trackers/`, and `data/seqmaps/` directories are populated at runtime.
- The reproduction environment uses BoxMOT `12.0.1`, which is pinned in
  `requirements.txt`. Install that requirement before running the package.
- `run_all.py` checks this version before starting the full comparison and
  stops early if another BoxMOT version is active.
- The runner also records the actually imported BoxMOT version in
  `output/<run-tag>/run_config.json`. This is an audit record and sanity check;
  it does not select or change the tracker version.
- BoxMOT is installed from the Python environment rather than copied into this
  repository. Do not substitute the separate `bytetrack` environment, which
  currently contains BoxMOT `16.0.10`, when reproducing the existing K3 table.

## Repository layout

```text
runall.py                         one-click alias
run_all.py                        full six-dataset comparison
run_ablation.py                   separate ablation entrypoint
scripts/                          only final runtime scripts
data/                             six K3 datasets
weights/osnet_x0_25_msmt17.pt     BoxMOT appearance model
deps/deep_sort/                   local DeepSORT source + ckpt.t7
deps/TrackEval/                   bundled TrackEval source
```

Generated results are ignored by `.gitignore`; model and image binaries are
marked for Git LFS in `.gitattributes`.
