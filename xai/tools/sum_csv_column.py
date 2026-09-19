#!/usr/bin/env python3
"""Sum (or average) a numeric column of a metrics CSV, e.g. the per-image AUC_topK
column produced by xai/metrics/q_metric.py or xai/metrics/gen_metric.py.

Usage:
    python xai/tools/sum_csv_column.py data/qmetric_out/results_per_image.csv AUC_topK
"""
import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", help="Path to the metrics CSV")
    parser.add_argument("column_name", help="Name of the column to sum")
    parser.add_argument("--mean", action="store_true", help="Print the mean instead of the sum")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    value = df[args.column_name].mean() if args.mean else df[args.column_name].sum()
    label = "Mean" if args.mean else "Sum"
    print(f"{label} of column '{args.column_name}': {value:.4f}")


if __name__ == "__main__":
    main()
