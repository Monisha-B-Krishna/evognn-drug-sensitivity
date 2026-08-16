"""
build_ga_sample.py
Creates a smaller, representative sample of master_dataset.parquet for the
GA to repeatedly train surrogate classifiers against, without loading the
full 235,800 x 15,834 dataset into memory.

Reads only as many Parquet row-groups as needed to reach the target sample
size, rather than loading the whole file first.

Input : data/processed/master_dataset.parquet
Output: data/processed/ga_sample.parquet
"""
import os
import sys
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config_loader import load_config

TARGET_SAMPLE_SIZE = 30_000  # rows for the GA sample


def main():
    cfg = load_config()
    master_path = cfg["paths"]["processed_master_dataset"].replace(".csv", ".parquet")

    parquet_file = pq.ParquetFile(master_path)
    n_row_groups = parquet_file.num_row_groups
    print(f"master_dataset.parquet has {n_row_groups} row groups")

    collected = []
    rows_so_far = 0
    for i in range(n_row_groups):
        table = parquet_file.read_row_group(i)
        df_batch = table.to_pandas()
        collected.append(df_batch)
        rows_so_far += len(df_batch)
        print(f"  Read row group {i + 1}/{n_row_groups}, rows so far: {rows_so_far:,}")
        if rows_so_far >= TARGET_SAMPLE_SIZE:
            break

    combined = pd.concat(collected, ignore_index=True)
    del collected

    if len(combined) > TARGET_SAMPLE_SIZE:
        combined = combined.sample(n=TARGET_SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    class_balance = combined["label"].value_counts(normalize=True)
    print(f"\nGA sample: {combined.shape}")
    print(f"Class balance in sample -> Sensitive: {class_balance.get(1, 0):.1%}, "
          f"Resistant: {class_balance.get(0, 0):.1%}")

    output_path = "data/processed/ga_sample.parquet"
    combined.to_parquet(output_path, index=False, compression="snappy")
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    main()