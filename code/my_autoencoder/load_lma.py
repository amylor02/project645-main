from pathlib import Path
import argparse
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt


def load_annotations(data_dir: Path):
    annotations_dir = data_dir / "annotations"
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    results = {}
    for csv in sorted(annotations_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv)
            results[csv.stem] = df
        except Exception as e:
            print(f"Failed to read {csv}: {e}", file=sys.stderr)
    return results


def print_annotation_shapes(annotations: dict):
    print("=" * 60)
    print(f"{'Motion File':<40} {'Rows':>6} {'Cols':>6}")
    print("=" * 60)
    for name, df in annotations.items():
        r, c = df.shape
        print(f"{name:<40} {r:6d} {c:6d}")
    print("=" * 60)
    print(f"Total files: {len(annotations)}")


def plot_per_frame_features(df: pd.DataFrame, title: str = None):
    """
    Plot numeric per-frame features from the annotation dataframe.
    Each numeric column becomes a subplot.
    """
    if df is None or df.empty:
        print("No data to plot")
        return

    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        print("No numeric columns to plot")
        return

    cols = list(numeric.columns)
    n = len(cols)
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 3 * nrows), squeeze=False)
    for i, col in enumerate(cols):
        r = i // ncols
        c = i % ncols
        ax = axes[r][c]
        ax.plot(numeric[col].values, marker='o', linestyle='-')
        ax.set_title(col)
        ax.set_xlabel('frame')
        ax.set_ylabel(col)
        ax.grid(True)

    # hide unused axes
    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r][c].axis('off')

    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main(argv=None):
    p = argparse.ArgumentParser(description="Load LMA annotations and print shapes")
    p.add_argument("--data-dir", type=Path, required=True, help="Path to data directory containing annotations/")
    args = p.parse_args(argv)

    data_dir = args.data_dir
    if not data_dir.exists():
        print(f"Data dir does not exist: {data_dir}", file=sys.stderr)
        return 2

    annotations = load_annotations(data_dir)
    if not annotations:
        print("No annotation CSVs found.")
        return 0

    print_annotation_shapes(annotations)

    # preview first file
    first_name = next(iter(annotations))
    print(f"\nPreview ({first_name}):")
    print(annotations[first_name].head())
    # plot per-frame numeric features for the first file
    # try:
    #     plot_per_frame_features(annotations[first_name], title=first_name)
    # except Exception as e:
    #     print(f"Failed to plot features for {first_name}: {e}", file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
import os
import pandas as pd
from pathlib import Path

def load_annotations(data_dir: str) -> dict[str, pd.DataFrame]:
    """
    Load LMA annotations from the annotations subfolder.
    
    Args:
        data_dir: Path to the data directory containing train, eval, and annotations folders
        
    Returns:
        Dictionary mapping motion file names to their annotation DataFrames
    """
    annotations_dir = Path(data_dir) / "annotations"
    
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
    
    annotations = {}
    csv_files = list(annotations_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {annotations_dir}")
        return annotations
    
    print(f"Found {len(csv_files)} annotation file(s) in {annotations_dir}\n")
    
    for csv_path in sorted(csv_files):
        motion_name = csv_path.stem  # filename without extension
        try:
            df = pd.read_csv(csv_path)
            annotations[motion_name] = df
        except Exception as e:
            print(f"  [ERROR] Failed to load {csv_path.name}: {e}")
    
    return annotations


def print_annotation_shapes(annotations: dict[str, pd.DataFrame]) -> None:
    """
    Print the shape of each motion file's annotation DataFrame.
    
    Args:
        annotations: Dictionary mapping motion file names to DataFrames
    """
    print("=" * 60)
    print(f"{'Motion File':<35} {'Rows':>6}  {'Columns':>7}")
    print("=" * 60)
    
    for motion_name, df in annotations.items():
        rows, cols = df.shape
        print(f"{motion_name:<35} {rows:>6}  {cols:>7}")
    
    print("=" * 60)
    print(f"Total motion files loaded: {len(annotations)}")
    
    # Also print column names (same for all files, so just use the first)
    if annotations:
        first_df = next(iter(annotations.values()))
        print(f"\nColumns ({len(first_df.columns)}):")
        for col in first_df.columns:
            print(f"  - {col}")


def main():
    # Resolve data directory relative to this script
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
    print(f"Data directory : {data_dir.resolve()}")
    print(f"Annotations dir: {(data_dir / 'annotations').resolve()}\n")
    
    # Load annotations
    annotations = load_annotations(data_dir)
    
    if not annotations:
        print("No annotations were loaded.")
        return
    
    # Print shapes
    print_annotation_shapes(annotations)
    
    # Optional: show a preview of the first file
    print("\n--- Preview of first annotation file ---")
    first_name, first_df = next(iter(annotations.items()))
    print(f"File: {first_name}")
    print(first_df.head())


if __name__ == "__main__":
    main()