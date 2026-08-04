# UAV Identity-Preserving Tracking Reproduction

This repository provides a self-contained reproduction package for UAV
multi-object tracking and identity-preservation experiments. It includes the
bundled input data, runnable scripts, model files, DeepSORT, and TrackEval.

The package is portable: runtime paths are resolved from the repository
location, so it can be cloned or moved to another directory without editing
machine-specific paths.

## Run the full comparison

From the repository root:

```powershell
python -m pip install -r requirements.txt
python runall.py --device auto
```

The command runs the complete comparison on all included datasets and writes
the tables and evaluation summaries to:

```text
output/reproduction_all/
```

For CPU-only execution:

```powershell
python runall.py --device cpu
```

The separate ablation entrypoint is:

```powershell
python run_ablation.py
```

Its outputs are written to `output/ablation/`.

## Included datasets

The package contains six UAV tracking datasets:

- UAVSwarm-39
- UAVSwarm-25
- UAVSwarm-61
- Watercraft
- Court-3
- Court-2

## Repository layout

```text
runall.py                         one-command entrypoint
run_all.py                        full comparison runner
run_ablation.py                   separate ablation runner
data/                             six bundled datasets
weights/                          required model files
deps/deep_sort/                   bundled DeepSORT code and checkpoint
deps/TrackEval/                   bundled evaluation code
```

Generated results are created locally under `output/` and are not part of the
repository.
