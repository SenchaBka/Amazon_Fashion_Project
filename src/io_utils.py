"""Utilities for loading JSONL datasets."""

import json
import pandas as pd


def load_jsonl(path: str) -> pd.DataFrame:
    """Load a JSONL file (one JSON object per line) into a DataFrame.

    Attempts pandas read_json with lines=True first; falls back to manual
    line-by-line parsing if pandas raises an error (e.g. mixed types).
    """
    print(f"[io_utils] Loading dataset from: {path}")
    try:
        df = pd.read_json(path, lines=True)
        print(f"[io_utils] Loaded {len(df):,} rows via pandas.read_json.")
        return df
    except Exception as e:
        print(f"[io_utils] pandas.read_json failed ({e}). Falling back to manual parsing.")
        rows = []
        skipped = 0
        with open(path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    skipped += 1
        if skipped:
            print(f"[io_utils] Skipped {skipped:,} malformed lines.")
        df = pd.DataFrame(rows)
        print(f"[io_utils] Loaded {len(df):,} rows via manual parsing.")
        return df
