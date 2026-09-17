"""Download all raw datasets into data/raw/<domain>/<name>/.

Usage:
    python scripts/download.py                # download everything (except yelp)
    python scripts/download.py --only akeed expedia
    python scripts/download.py --list          # list dataset names

Kaggle datasets/competitions require ~/.kaggle/kaggle.json (username + key).
Competitions (expedia, airbnb) additionally require clicking "I Understand
and Accept" on the competition's rules page while logged in as that Kaggle
account, BEFORE running this script — the API key alone cannot accept rules:
    - https://www.kaggle.com/c/expedia-hotel-recommendations/rules
    - https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/rules

Yelp Open Dataset has no API / stable direct-download URL. It requires
manually filling a form (name/email/purpose) and agreeing to Yelp's terms at
https://www.yelp.com/dataset/download, then saving the resulting archive to
data/raw/food/yelp/. This script cannot do that step; download_yelp() only
prints the instructions.
"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW = REPO_ROOT / "data" / "raw"


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)


def _extract_zip(zip_path: Path, dest_dir: Path, delete_after: bool = True) -> None:
    print(f"  extracting {zip_path.name}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
    if delete_after:
        zip_path.unlink()


def download_movielens() -> None:
    dest_dir = RAW / "prototype" / "movielens"
    if (dest_dir / "ml-latest-small").exists():
        print("movielens already present, skip")
        return
    zip_path = dest_dir / "ml-latest-small.zip"
    _download_file("https://files.grouplens.org/datasets/movielens/ml-latest-small.zip", zip_path)
    _extract_zip(zip_path, dest_dir)


def download_nyc_tlc(months: list[str] | None = None) -> None:
    months = months or ["2024-01"]
    for month in months:
        dest_dir = RAW / "ride" / "nyc_tlc" / month
        dest_file = dest_dir / f"yellow_tripdata_{month}.parquet"
        if dest_file.exists():
            print(f"nyc_tlc {month} already present, skip")
            continue
        url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"
        _download_file(url, dest_file)


def download_porto_taxi() -> None:
    dest_dir = RAW / "ride" / "porto_taxi"
    if (dest_dir / "train.csv").exists():
        print("porto_taxi already present, skip")
        return
    zip_path = dest_dir / "porto_taxi.zip"
    url = "https://archive.ics.uci.edu/static/public/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015.zip"
    _download_file(url, zip_path)
    _extract_zip(zip_path, dest_dir)
    nested_zip = dest_dir / "train.csv.zip"
    if nested_zip.exists():
        _extract_zip(nested_zip, dest_dir)


def _kaggle_dataset_download(slug: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download", "-d", slug, "-p", str(dest_dir)],
        check=True,
    )
    for zip_path in dest_dir.glob("*.zip"):
        _extract_zip(zip_path, dest_dir)


def _kaggle_competition_download(slug: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "competitions", "download", "-c", slug, "-p", str(dest_dir)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        print(
            f"  FAILED — most likely you haven't accepted the competition rules yet.\n"
            f"  Log in as your Kaggle account and click 'I Understand and Accept' at:\n"
            f"  https://www.kaggle.com/c/{slug}/rules"
        )
        return
    for zip_path in dest_dir.glob("*.zip"):
        _extract_zip(zip_path, dest_dir)
    # some competition zips nest more zips inside (e.g. Airbnb's per-file zips)
    for nested_zip in dest_dir.glob("*.zip"):
        _extract_zip(nested_zip, dest_dir)


def download_akeed() -> None:
    dest_dir = RAW / "food" / "akeed"
    if (dest_dir / "orders.csv").exists():
        print("akeed already present, skip")
        return
    _kaggle_dataset_download("darisdzakwanhoesien2/akeed-restaurant-recommendation-challenge", dest_dir)


def download_trivago() -> None:
    dest_dir = RAW / "hospitality" / "trivago_2019"
    if (dest_dir / "train.csv").exists():
        print("trivago_2019 already present, skip")
        return
    _kaggle_dataset_download("phhasian0710/trivago-recsys", dest_dir)
    # the zip contains a redundant nested folder with the same files
    nested = dest_dir / "trivagorecsyschallengedata2019_v2"
    if nested.exists():
        import shutil
        shutil.rmtree(nested)


def download_expedia() -> None:
    dest_dir = RAW / "hospitality" / "expedia"
    if (dest_dir / "train.csv").exists():
        print("expedia already present, skip")
        return
    print("expedia: requires rule-accept — see module docstring")
    _kaggle_competition_download("expedia-hotel-recommendations", dest_dir)


def download_airbnb() -> None:
    dest_dir = RAW / "hospitality" / "airbnb_new_user"
    if (dest_dir / "train_users_2.csv").exists():
        print("airbnb_new_user already present, skip")
        return
    print("airbnb_new_user: requires rule-accept — see module docstring")
    _kaggle_competition_download("airbnb-recruiting-new-user-bookings", dest_dir)


def download_yelp() -> None:
    dest_dir = RAW / "food" / "yelp"
    print(
        "yelp: NOT automatable — no API, no stable direct-download URL.\n"
        "  1. Open https://www.yelp.com/dataset/download\n"
        "  2. Fill the form (name, email, intended use) and agree to Yelp's Dataset License\n"
        "  3. Download the resulting archive and extract it into:\n"
        f"     {dest_dir}\n"
        "  (Kaggle also mirrors an older snapshot, no form needed, if that's acceptable instead:\n"
        "   https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset — run:\n"
        f"   python -m kaggle datasets download -d yelp-dataset/yelp-dataset -p {dest_dir})"
    )


DATASETS = {
    "movielens": download_movielens,
    "nyc_tlc": download_nyc_tlc,
    "porto_taxi": download_porto_taxi,
    "akeed": download_akeed,
    "trivago": download_trivago,
    "expedia": download_expedia,
    "airbnb": download_airbnb,
    "yelp": download_yelp,  # prints instructions only, never auto-downloads
}

DEFAULT_SET = [name for name in DATASETS if name != "yelp"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=list(DATASETS), help="download only these datasets")
    parser.add_argument("--list", action="store_true", help="list dataset names and exit")
    args = parser.parse_args()

    if args.list:
        for name in DATASETS:
            print(name)
        return

    names = args.only or DEFAULT_SET
    for name in names:
        print(f"=== {name} ===")
        DATASETS[name]()

    if not args.only:
        print("\n=== yelp (skipped by default, manual step required) ===")
        download_yelp()


if __name__ == "__main__":
    main()
