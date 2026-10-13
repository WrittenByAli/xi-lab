"""
Download all Cricsheet ball-by-ball data and the player register.

Usage:
    python ml/ingest/download.py                  # full download
    python ml/ingest/download.py --incremental    # skip already-present zips

Cricsheet data is published under ODC-BY. Cite as:
    Cricsheet (https://cricsheet.org) — ODC-BY licence
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ── Target directories ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "cricsheet"

FORMAT_DIRS = {
    "t20s": RAW / "t20s",
    "odis": RAW / "odis",
    "tests": RAW / "tests",
    "listA": RAW / "listA",
    "firstClass": RAW / "firstClass",
}

# ── Cricsheet download manifest ───────────────────────────────────────────────
# Each entry: (local_subdir, zip_url)
# Cricsheet serves one zip per competition type. We download the "all" bundles
# (every match ever) rather than per-season, because we need the full history.
DOWNLOADS: list[tuple[str, str]] = [
    # ── International ──────────────────────────────────────────────────────
    ("t20s", "https://cricsheet.org/downloads/t20s_json.zip"),
    ("odis", "https://cricsheet.org/downloads/odis_json.zip"),
    ("tests", "https://cricsheet.org/downloads/tests_json.zip"),
    # ── T20 leagues ────────────────────────────────────────────────────────
    ("t20s", "https://cricsheet.org/downloads/ipl_json.zip"),
    ("t20s", "https://cricsheet.org/downloads/psl_json.zip"),
    ("t20s", "https://cricsheet.org/downloads/bbl_json.zip"),
    ("t20s", "https://cricsheet.org/downloads/cpl_json.zip"),
    ("t20s", "https://cricsheet.org/downloads/ntb_json.zip"),
    ("t20s", "https://cricsheet.org/downloads/sat_json.zip"),
    ("t20s", "https://cricsheet.org/downloads/ilt_json.zip"),
    ("t20s", "https://cricsheet.org/downloads/bpl_json.zip"),
    ("t20s", "https://cricsheet.org/downloads/lpl_json.zip"),
    ("t20s", "https://cricsheet.org/downloads/mlc_json.zip"),
    ("t20s", "https://cricsheet.org/downloads/ssm_json.zip"),
    # ── List A ─────────────────────────────────────────────────────────────
    ("listA", "https://cricsheet.org/downloads/odis_json.zip"),  # reuse ODI
    # ── Player register ────────────────────────────────────────────────────
    # people.csv and names.csv are separate downloads
]

REGISTER_URLS = [
    "https://cricsheet.org/register/people.csv",
    "https://cricsheet.org/register/names.csv",
]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path, retries: int = 3) -> Path:
    """Stream-download *url* to *dest*, with progress bar and retries."""
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".tmp")
                with (
                    tmp.open("wb") as f,
                    tqdm(
                        total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
                    ) as bar,
                ):
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
                        bar.update(len(chunk))
                tmp.rename(dest)
                return dest
        except Exception as exc:
            logger.warning("Attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts")


def _extract_zip(zip_path: Path, out_dir: Path) -> int:
    """Extract JSON match files from *zip_path* into *out_dir*. Returns count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith(".json") and not name.startswith("__"):
                dest = out_dir / Path(name).name
                if not dest.exists():
                    zf.extract(name, out_dir)
                    # Flatten: move out of any subdirectory
                    extracted = out_dir / name
                    if extracted != dest and extracted.exists():
                        extracted.rename(dest)
                    count += 1
    return count


# ── Main ──────────────────────────────────────────────────────────────────────


def download_all(incremental: bool = False) -> None:
    seen_urls: set[str] = set()

    for subdir_key, url in DOWNLOADS:
        out_dir = FORMAT_DIRS[subdir_key]
        zip_name = url.split("/")[-1]
        zip_path = out_dir / zip_name

        if url in seen_urls:
            continue
        seen_urls.add(url)

        if incremental and zip_path.exists():
            logger.info("Skip (exists): %s", zip_name)
            continue

        logger.info("Downloading %s → %s", zip_name, out_dir)
        try:
            _download(url, zip_path)
            n = _extract_zip(zip_path, out_dir)
            logger.info("  Extracted %d new JSON files from %s", n, zip_name)
        except Exception as exc:
            logger.error("  Failed: %s", exc)

    # Download register files
    register_dir = ROOT / "data" / "reference"
    register_dir.mkdir(parents=True, exist_ok=True)
    for url in REGISTER_URLS:
        name = url.split("/")[-1]
        dest = register_dir / name
        if incremental and dest.exists():
            logger.info("Skip (exists): %s", name)
            continue
        logger.info("Downloading register: %s", name)
        try:
            _download(url, dest)
        except Exception as exc:
            logger.error("  Failed: %s", exc)

    # Summary
    for key, d in FORMAT_DIRS.items():
        n = len(list(d.glob("*.json")))
        logger.info("  %-12s  %d match files", key, n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Cricsheet data")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Skip files that already exist locally",
    )
    args = parser.parse_args()
    download_all(incremental=args.incremental)
