import json
import pandas as pd

from src.config import CACHE_DIR


def print_example_row(df: pd.DataFrame):
    sample = df.iloc[0]
    for col in df.columns:
        print(f"{col}: {sample[col]}")
    print()


def load_from_cache(*filenames: str) -> list:
    """
    Load one or more cached files by filename (without extension).
    JSONL files are returned as DataFrames; single-object JSON as dicts.
    """
    results = []
    for filename in filenames:
        path = f"{CACHE_DIR}/{filename}.json"
        with open(path) as f:
            content = f.read().strip()
        if "\n" in content:
            results.append(pd.read_json(path, lines=True))
        else:
            results.append(json.loads(content))
    return results


def cache_data(**kwargs):
    """
    Cache data to disk. Accepts keyword args where key = filename, value = data.
    Supports DataFrames (saved as JSONL) and dicts (saved as JSON).
    Example: cache_data(dataframe_cache=df, pmid_mapping=mapping)
    """
    for filename, data in kwargs.items():
        path = f"{CACHE_DIR}/{filename}.json"
        if isinstance(data, pd.DataFrame):
            data.to_json(path, orient='records', lines=True)
        elif isinstance(data, dict):
            json.dump(data, open(path, "w"))
