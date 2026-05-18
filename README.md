# PhiSat-2 RPC Benchmark Evaluator

Evaluates a contributor-provided RPC algorithm against PhiSat-2 scenes using GCP-chip NCC verification, with optional comparison to reference orthorectified images.

---

## Setup

Clone the repo:

```bash
git clone https://github.com/DavidVan1/bcs-thesis && cd bcs-thesis
```

Create the minimal environment used by the evaluator:

```bash
conda env create -f environment_eval.yml
conda activate phisat-eval
```

---

## Dataset

The PhiSat-2 scenes and reference orthorectified images are hosted on Hugging Face:

https://huggingface.co/datasets/davidvan1/phisat2-ortho-reference

Note: the dataset is around 187 GB, so download may take some time.

Download it using your preferred method (Hugging Face CLI, git-lfs, or the web UI), then pass the local reference-ortho directory to the evaluator.

The evaluator looks for reference orthos using these patterns:

- `reference_ortho.tif` in the reference directory
- `reference_dir/<scene_id>/ortho_reference_*.tif`
- `reference_dir/*<scene_id>*ortho*.tif`

If you want to create the reference dataset, see [scripts/README.md](scripts/README.md).

## Implement Your Algorithm

Fill in `compute_rpc` in `make_rpc_template.py`:

```python
def compute_rpc(scene_dir: Path) -> dict:
    """Compute RPC parameters for a scene.

    Args:
        scene_dir: Input scene directory containing the data needed to derive RPCs.

    Returns:
        A dictionary with RPC metadata and coefficient arrays.
    """
    ### TODO: Implement the RPC computation here
    ...
    return {
        "LINE_OFF": 2048.0, "SAMP_OFF": 2048.0, "LAT_OFF": 0.0, "LONG_OFF": 0.0, "HEIGHT_OFF": 0.0,
        "LINE_SCALE": 2048.0, "SAMP_SCALE": 2048.0, "LAT_SCALE": 1.0, "LONG_SCALE": 1.0, "HEIGHT_SCALE": 500.0,
        "LINE_NUM_COEFF": [0.0] * 20, "LINE_DEN_COEFF": [1.0] + [0.0] * 19,
        "SAMP_NUM_COEFF": [0.0] * 20, "SAMP_DEN_COEFF": [1.0] + [0.0] * 19,
    }
```

The evaluator calls `process_scene` which invokes `compute_rpc` and writes the `_RPC.txt` sidecar automatically.

---
## Run

**All scenes:**
```bash
python evaluate_external_rpc.py \
    --dataset  ./path/to/scenes \
    --script   ./make_rpc_template.py \
    --output   ./path/to/output
```

**With reference comparison:**
```bash
python evaluate_external_rpc.py \
    --dataset             ./path/to/scenes \
    --script              ./make_rpc_template.py \
    --output              ./path/to/output \
    --reference-ortho-dir ./path/to/reference_orthos
```

**Single scene:**
```bash
python evaluate_external_rpc.py \
    --dataset  ./path/to/scenes \
    --script   ./make_rpc_template.py \
    --output   ./path/to/output \
    --scene    PHISAT-2_L1_000001341_...
```

**Options:**

| Argument | Default | Description |
|---|---|---|
| `--dataset` | required | Path to scenes folder |
| `--script` | required | Path to `make_rpc_template.py` |
| `--output` | required | Output folder |
| `--reference-ortho-dir` | None | Path to reference orthos for comparison |
| `--scene` | None | Run single scene by name or absolute path |
| `--min-ncc` | 0.4 | Minimum NCC threshold for GCP matching |
| `--workers` | 1 | Number of parallel workers |

---

## Outputs

The evaluator writes:

- `output/rpc/<scene_id>_RPC.txt`
- `output/ortho/<scene_id>.tif`
- `output/metrics/<scene_id>_metrics.json`
- `output/metrics/<scene_id>_reference_metrics.json` (when reference orthos are used)
- `output/metrics_<timestamp>.csv`
