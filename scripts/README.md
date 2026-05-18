# Reference Dataset Generation

This folder contains the scripts used to build the reference dataset for the PhiSat-2 benchmark.

Download the dataset from Hugging Face:
https://huggingface.co/datasets/davidvan1/phisat2-ortho-reference
It contains the input files, reference files, and other data needed for the scripts.

If you want to see how the input data was downloaded, see [data_preparation/README.md](../data_preparation/README.md).

In `input/scenes`, you will find the input data for each scene: PhiSat-2 imagery, DEM, Sentinel imagery, and Sentinel GRI GCP data.

Available matchers are `efficientloftr` and `lightglue`.

## Requirements

Use the full pipeline environment:

```bash
conda install -n base conda-libmamba-solver # Install libmamba solver first (one-time)
conda config --set solver libmamba

conda env create -f environment.yml
conda activate phis
```

## 1) Generate reference RPC files

This step creates RPC text files under your reference RPC folder and writes metrics to your reference output folder.

```bash
python scripts/make_reference_rpc.py \
    --dataset /path/to/your/scenes \
    --output /path/to/your/reference-output \
    --matcher efficientloftr \
    --stage all \
    --workers 1
```

Useful options:

- `--scene PHISAT-2_L1_...` processes only one scene
- `--stage calibrate rpc_fit` is the default stage set for reference RPC generation
- `--workers N` sets the number of parallel processes

## 2) Generate reference ortho images

After the RPC files exist, run the ortho step with the same `matcher` value:

```bash
python scripts/make_reference_ortho.py \
    --dataset /path/to/your/scenes \
    --output /path/to/your/reference-output \
    --matcher efficientloftr \
    --rpc-dir /path/to/your/reference-rpc/efficientloftr \
    --workers 1 \
    --overwrite-ortho
```

Outputs for each scene:

- `/path/to/your/reference-output/<scene_id>/ortho_reference_<matcher>.tif` — the orthorectified image
- `/path/to/your/reference-output/<scene_id>/verification_ncc_<matcher>.json` — statistics for individual GCPs
- `/path/to/your/reference-output/verification_summary_<matcher>_<timestamp>.csv` — overall summary

Note: if you use a different `matcher`, make sure the same value is used in both steps so the script can find `/path/to/your/reference-rpc/<matcher>/`.
