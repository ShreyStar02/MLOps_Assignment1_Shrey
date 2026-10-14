"""Download the UCI Heart Disease (Cleveland) dataset and write raw + cleaned CSVs."""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import clean, load_raw  # noqa: E402

URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "heart-disease/processed.cleveland.data")
RAW = ROOT / "data" / "raw" / "processed.cleveland.data"
CLEAN = ROOT / "data" / "processed" / "heart_clean.csv"


def main() -> None:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, RAW)  # noqa: S310 - fixed https URL
    df = clean(load_raw(RAW))
    df.to_csv(CLEAN, index=False)
    print(f"Wrote {CLEAN} shape={df.shape} positives={int(df['target'].sum())}")


if __name__ == "__main__":
    main()
