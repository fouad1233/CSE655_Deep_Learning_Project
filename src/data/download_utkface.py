"""Download and extract UTKFace (aligned & cropped faces) into data/raw/UTKFace/.

Source: the official Google Drive link (https://susanqq.github.io/UTKFace/) is no longer
available, so we use a Hugging Face re-upload pinned to an exact revision and verify
its SHA-256, so every machine gets byte-identical data.
Licence: the ORIGINAL terms apply (non-commercial research only); cite
Zhang, Song & Qi, "Age Progression/Regression by Conditional Adversarial Autoencoder", CVPR 2017.

Usage:  python src/data/download_utkface.py
"""
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
ARCHIVE = RAW_DIR / "UTKFace.tar.gz"
IMAGE_DIR = RAW_DIR / "UTKFace"

# pinned revision of https://huggingface.co/datasets/py97/UTKFace-Cropped
REVISION = "e80f21fb21fa3631380c3cb634b79be3ac2f4cba"
URL = f"https://huggingface.co/datasets/py97/UTKFace-Cropped/resolve/{REVISION}/UTKFace.tar.gz"
SHA256 = "2c0655397b498e81b82d685ee74bee8c2e4364a5897caf3afec6ea48929024a7"
SIZE = 106_634_631


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download() -> None:
    if ARCHIVE.exists() and sha256(ARCHIVE) == SHA256:
        print(f"archive already present and verified: {ARCHIVE}")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading {SIZE / 1e6:.1f} MB from {URL}")
    tmp = ARCHIVE.with_suffix(".part")
    urllib.request.urlretrieve(URL, tmp)
    digest = sha256(tmp)
    if digest != SHA256:
        tmp.unlink()
        sys.exit(f"SHA-256 mismatch: got {digest}, expected {SHA256}")
    tmp.replace(ARCHIVE)
    print("download verified (SHA-256 OK)")


def extract() -> None:
    if IMAGE_DIR.exists() and any(IMAGE_DIR.iterdir()):
        print(f"already extracted: {IMAGE_DIR}")
        return
    with tarfile.open(ARCHIVE) as tar:
        # filter="data" blocks path traversal / links / special files from the archive
        tar.extractall(RAW_DIR, filter="data")
    n = sum(1 for _ in IMAGE_DIR.glob("*.jpg*"))
    print(f"extracted {n} files to {IMAGE_DIR}")


if __name__ == "__main__":
    download()
    extract()
