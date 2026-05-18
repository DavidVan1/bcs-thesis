import os
import sys
import re
import time
from pathlib import Path

import requests
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = BASE_DIR / "notebooks" / "CollectionBatchDownloadScript"
DATASET_CSV = BASE_DIR / "dataset_full.csv"
DOWNLOAD_PATH = BASE_DIR / "data" / "l1_files"
COLLECTION_ID = 7
BASE_URL = "https://phisat2.insula.earth"


if not SCRIPT_DIR.exists():
    raise RuntimeError(f"Could not find the Insula config directory: {SCRIPT_DIR}")

package_candidates = sorted((BASE_DIR / "notebooks" / "lib").glob("python*/site-packages/InsulaWorkflowClient"))
if not package_candidates:
    raise RuntimeError("Could not find the InsulaWorkflowClient package in notebooks/lib/python*/site-packages")

PACKAGE_DIR = package_candidates[0].parent

for path in (SCRIPT_DIR, PACKAGE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

print(f"[IMPORT] Selected config directory: {SCRIPT_DIR}", flush=True)
print(f"[IMPORT] Selected package directory: {PACKAGE_DIR}", flush=True)

if not DATASET_CSV.exists():
    raise FileNotFoundError(f"Dataset CSV not found: {DATASET_CSV}")

print("[IMPORT] sys.path[0] =", sys.path[0], flush=True)

import config as CONFIG
import auth as AUTH
from InsulaWorkflowClient import InsulaOpenIDConnect


class L1Downloader:
    def __init__(self):
        self.session = requests.session()
        self.auth = InsulaOpenIDConnect(
            authorization_endpoint=CONFIG.Download.authorization_endpoint,
            token_endpoint=CONFIG.Download.token_endpoint,
            redirect_uri=CONFIG.Download.redirect_uri,
            client_id=CONFIG.Download.client_id
        )
        self.auth.set_user_credentials(
            username=AUTH.Insula.username,
            password=AUTH.Insula.password
        )

    def get_auth_header(self):
        return {"Authorization": self.auth.get_authorization_header()}

    def get_all_l1_files(self, collection_id=COLLECTION_ID):
        all_files = []
        url = (
            f"{BASE_URL}/secure/api/v2.0/platformFiles/search/parametricFind"
            f"?collection={BASE_URL}/secure/api/v2.0/collections/{collection_id}"
            f"&sort=filename"
        )

        page = 0
        while url:
            r = self.session.get(url, headers=self.get_auth_header(), verify=True)
            print(f"[INDEX] page={page} status={r.status_code}", flush=True)

            if r.status_code != 200:
                raise RuntimeError(f"Index request failed: {r.status_code} {r.text[:500]}")

            data = r.json()

            if "_embedded" not in data or "platformFiles" not in data["_embedded"]:
                raise RuntimeError(f"Unexpected response: {data}")

            batch = data["_embedded"]["platformFiles"]
            all_files.extend(batch)
            print(f"[INDEX] +{len(batch)} files total={len(all_files)}", flush=True)

            url = data.get("_links", {}).get("next", {}).get("href")
            page += 1
            time.sleep(0.2)

        return all_files

    @staticmethod
    def extract_l1_id(image_id):
        m = re.search(r"PHISAT-2_L1_(\d+)_", str(image_id))
        return m.group(1) if m else None

    @staticmethod
    def build_l1_lookup(l1_files):
        lookup = {}
        for pf in l1_files:
            fname = pf["filename"]
            m = re.search(r"PHISAT-2_L1_(\d+)_", fname)
            if m:
                lookup[m.group(1)] = pf
        return lookup

    def download_file(self, dl_link, filename):
        local_filename = filename.rsplit("/", 1)[-1]
        local_path = DOWNLOAD_PATH / local_filename

        if local_path.exists():
            print(f"[SKIP] exists {local_filename}", flush=True)
            return True

        r = self.session.get(
            dl_link,
            headers=self.get_auth_header(),
            verify=True,
            stream=True
        )

        if r.status_code != 200:
            print(f"[FAIL] {local_filename} status={r.status_code}", flush=True)
            return False

        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"[OK] downloaded {local_filename}", flush=True)
        return True

    def run(self):
        DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)

        print("[START] loading dataset", flush=True)
        df = pd.read_csv(DATASET_CSV)
        print(f"[DATASET] rows={len(df)}", flush=True)

        print("[START] indexing L1 collection", flush=True)
        l1_files = self.get_all_l1_files(COLLECTION_ID)
        print(f"[L1] indexed={len(l1_files)}", flush=True)

        l1_lookup = self.build_l1_lookup(l1_files)

        total = 0
        matched = 0
        downloaded_or_present = 0
        missing = 0

        for _, row in df.iterrows():
            total += 1
            image_id = row["image_id"]
            l1_id = self.extract_l1_id(image_id)

            if not l1_id:
                print(f"[MISS] invalid image_id format: {image_id}", flush=True)
                missing += 1
                continue

            pf = l1_lookup.get(l1_id)
            if not pf:
                print(f"[MISS] no L1 for {image_id}", flush=True)
                missing += 1
                continue

            matched += 1
            dl_link = pf["_links"]["download"]["href"]
            filename = pf["filename"]

            if self.download_file(dl_link, filename):
                downloaded_or_present += 1

            time.sleep(0.5)

        print(
            f"[DONE] total={total} matched={matched} downloaded_or_present={downloaded_or_present} missing={missing}",
            flush=True
        )


if __name__ == "__main__":
    L1Downloader().run()