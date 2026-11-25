
from pathlib import Path
import pandas as pd

data_dir = Path("data")

files = [
    "samples.csv.gz",
    "actions.csv.gz",
    "rewards.csv.gz",
    "main.csv.gz",
]

for filename in files:
    path = data_dir / filename

    if not path.exists():
        print(f"Missing file: {path}")
        continue

    df = pd.read_csv(path, compression="gzip")

    print(f"\n--- {filename} ---")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("First 20 column names:")
    print(df.columns.tolist()[:20])

    print("\nFirst 3 rows (first 8 columns):")
    print(df.iloc[:3, :8])
