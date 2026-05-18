# Insula Batch Download Script

This folder contains the Python script used to download PhiSat-2 data from Insula.

The script was adapted from the official CGI Italy notebooks repository:

https://github.com/cgi-italy/notebooks

The original example is the Collection Batch Download Script. This local copy was adjusted for the PhiSat-2 collection and the dataset structure used in this project.

## What it does

- reads scene IDs from `dataset_full.csv`
- looks up the matching PhiSat-2 L1 files on Insula
- downloads the files to the local `data/l1_files/` folder

## Requirements

The script uses the Python environment stored in this folder. Install the notebook environment requirements if needed:

```bash
pip install -r notebooks/requirements.txt
```

## Configuration

Before running the script, update these files in `notebooks/CollectionBatchDownloadScript/`:

- `auth.py` for your Insula username and password
- `config.py` for the correct authorization and token endpoints

The script is configured for the PhiSat-2 Insula instance and collection ID 7.

## Run

From this folder:

```bash
python download_l1_working.py
```

The downloaded files are saved into `data/l1_files/`.

## Notes

- The PhiSat-2 instance requires mail verification.
- Log in to the Insula UI at least once before using the API.
- If you want the full original notebook examples, use the source repository linked above.