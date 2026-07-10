import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1. CONFIG
# =========================
DATA_ROOT = "dataa"  # <-- change this to your root folder

# The exact feature names in your CSVs
FEATURE_COLUMNS = [
    "voltage mean", "voltage std", "voltage kurtosis", "voltage skewness",
    "CC Q", "CC charge time", "voltage slope", "voltage entropy",
    "current mean", "current std", "current kurtosis", "current skewness",
    "CV Q", "CV charge time", "current slope", "current entropy",
    "capacity",
]

# Features to highlight in pairplots
KEY_FEATURES = [
    "voltage mean",
    "voltage std",
    "current mean",
    "current std",
    "CC Q",
    "CV Q",
    "capacity",
]

# =========================
# 2. LOAD & MERGE CSV FILES
# =========================
def load_all_data(data_root: str) -> pd.DataFrame:
    """
    Recursively load all CSV files under data_root,
    and add 'dataset' and 'battery' columns based on folder structure:
        data_root/dataset/battery/*.csv
    """
    all_rows = []

    pattern = os.path.join(data_root, "**", "*.csv")
    csv_files = glob.glob(pattern, recursive=True)

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {data_root}")

    for csv_path in csv_files:
        # Extract dataset and battery names from path
        rel = os.path.relpath(csv_path, data_root)
        parts = rel.split(os.sep)

        if len(parts) >= 3:
            dataset_name = parts[0]
            battery_name = parts[1]
        elif len(parts) == 2:
            dataset_name = parts[0]
            battery_name = "unknown"
        else:
            dataset_name = "unknown"
            battery_name = "unknown"

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Skipping {csv_path} due to read error: {e}")
            continue

        # Keep only known columns that exist in this file
        cols_in_file = [c for c in FEATURE_COLUMNS if c in df.columns]
        df = df[cols_in_file].copy()

        df["dataset"] = dataset_name
        df["battery"] = battery_name

        all_rows.append(df)

    if not all_rows:
        raise RuntimeError("No valid CSVs loaded. Check column names and paths.")

    full_df = pd.concat(all_rows, ignore_index=True)
    return full_df


# =========================
# 3. PLOTTING UTILITIES
# =========================
def set_presentation_style():
    """Use a clean, colourful style for figures."""
    sns.set_theme(
        context="talk",      # larger fonts
        style="whitegrid",   # light grid
        font_scale=1.1
    )
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["legend.fontsize"] = 12


def plot_feature_histograms(df: pd.DataFrame, save_path=None):
    """Histograms for all numeric features."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    n_cols = 3
    n_rows = int((len(numeric_cols) + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for ax, col in zip(axes, numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, edgecolor="black")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    # Hide any unused subplots
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_feature_boxplots(df: pd.DataFrame, save_path=None):
    """Boxplots to show spread and outliers for each feature."""
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != "capacity"]

    plt.figure(figsize=(1.4 * len(numeric_cols), 6))
    sns.boxplot(data=df[numeric_cols], orient="v")
    plt.xticks(rotation=45, ha="right")
    plt.title("Feature Distributions (Boxplots)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, save_path=None):
    """Correlation heatmap between features (including capacity)."""
    corr = df[FEATURE_COLUMNS].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=False,
        cmap="coolwarm",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_pairplot_key_features(df: pd.DataFrame, save_path=None):
    """Pairplot of a few key features vs capacity, coloured by dataset."""
    cols = [c for c in KEY_FEATURES if c in df.columns]

    # To avoid massive pairplot, downsample if very large
    if len(df) > 5000:
        df_plot = df.sample(5000, random_state=42)
    else:
        df_plot = df

    g = sns.pairplot(
        df_plot,
        vars=cols,
        hue="dataset",
        diag_kind="kde",
        corner=True,
        plot_kws={"alpha": 0.6, "s": 20}
    )
    g.fig.suptitle("Key Feature Relationships vs Capacity", y=1.02)

    if save_path:
        g.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_capacity_by_dataset(df: pd.DataFrame, save_path=None):
    """Capacity distribution per dataset, for sanity check."""
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=df,
        x="dataset",
        y="capacity",
        inner="box",
        cut=0
    )
    plt.title("Capacity Distribution by Dataset")
    plt.xlabel("Dataset")
    plt.ylabel("Capacity")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


# =========================
# 4. MAIN
# =========================
if __name__ == "__main__":
    set_presentation_style()

    print(f"Loading data from: {DATA_ROOT}")
    df_all = load_all_data(DATA_ROOT)
    print(f"Loaded {len(df_all)} rows from {df_all['dataset'].nunique()} datasets.")

    # Basic info
    print("\nColumns:", df_all.columns.tolist())
    print("\nDataset counts:\n", df_all["dataset"].value_counts())
    print("\nBattery counts (top 10):\n", df_all["battery"].value_counts().head(10))

    # Plotting
    plot_feature_histograms(df_all, save_path="fig_feature_histograms.png")
    plot_feature_boxplots(df_all, save_path="fig_feature_boxplots.png")
    plot_correlation_heatmap(df_all, save_path="fig_correlation_heatmap.png")
    plot_pairplot_key_features(df_all, save_path="fig_pairplot_key_features.png")
    plot_capacity_by_dataset(df_all, save_path="fig_capacity_by_dataset.png")
