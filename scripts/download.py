"""Download the raw Expedia Hotel Recommendations data into data/raw/hospitality/expedia/.

Usage:
    python scripts/download.py

Needs ~/.kaggle/kaggle.json (username + key). The Kaggle API key alone cannot accept the
competition rules: click "I Understand and Accept" on
https://www.kaggle.com/c/expedia-hotel-recommendations/rules while logged in as that Kaggle
account, BEFORE running this script.
"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEST = REPO_ROOT / "data" / "raw" / "hospitality" / "expedia"
COMPETITION = "expedia-hotel-recommendations"


def _extract_zip(zip_path: Path, dest_dir: Path, delete_after: bool = True) -> None:
    print(f"  extracting {zip_path.name}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
    if delete_after:
        zip_path.unlink()


def download_expedia() -> None:
    if (DEST / "train.csv").exists():
        print("expedia already present, skip")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(DEST)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        print(
            "  FAILED — most likely you haven't accepted the competition rules yet.\n"
            "  Log in as your Kaggle account and click 'I Understand and Accept' at:\n"
            f"  https://www.kaggle.com/c/{COMPETITION}/rules"
        )
        return
    for zip_path in DEST.glob("*.zip"):
        _extract_zip(zip_path, DEST)
    # some competition zips nest more zips inside
    for nested_zip in DEST.glob("*.zip"):
        _extract_zip(nested_zip, DEST)


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    download_expedia()


if __name__ == "__main__":
    main()
