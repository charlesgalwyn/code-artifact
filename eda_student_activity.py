import argparse
import csv
import sys
from pathlib import Path
from typing import Optional, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def detect_delimiter(sample_path: Path, default: str = ",") -> str:
    try:
        with sample_path.open("r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(2048)
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return default


def safe_parse_datetime(series: pd.Series) -> pd.Series:
    if series is None:
        return None
    try:
        return pd.to_datetime(series, errors="coerce", infer_datetime_format=True, utc=False)
    except Exception:
        return pd.to_datetime(series, errors="coerce")


def save_text(text: str, path: Path):
    path.write_text(text, encoding="utf-8")


def ensure_outdir(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)


def plot_missing_heatmap(df: pd.DataFrame, outpath: Path, title: str = "Missing Values Heatmap"):
    # Create a boolean mask (True = missing), imshow will show it as an image.
    mask = df.isnull().values
    plt.figure(figsize=(10, 6))
    plt.imshow(mask, aspect='auto', interpolation='nearest')
    plt.title(title)
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.xticks(ticks=np.arange(df.shape[1]), labels=df.columns, rotation=90, fontsize=7)
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def hist_numeric(df: pd.DataFrame, outdir: Path):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        try:
            plt.figure()
            df[col].dropna().hist(bins=30, edgecolor='black')
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(outdir / f"hist_{col}.png", dpi=150)
            plt.close()
        except Exception as e:
            print(f"[WARN] Skipping histogram for {col}: {e}", file=sys.stderr)


def bar_top_categories(df: pd.DataFrame, outdir: Path, topn: int = 15):
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        try:
            vc = df[col].astype("string").fillna("NA").value_counts().head(topn)
            plt.figure(figsize=(10, 4))
            plt.bar(vc.index.tolist(), vc.values.tolist())
            plt.title(f"Top {topn} categories in {col}")
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(outdir / f"topcats_{col}.png", dpi=150)
            plt.close()

            # Save table too
            vc.to_frame(name="count").to_csv(outdir / f"topcats_{col}.csv")
        except Exception as e:
            print(f"[WARN] Skipping top-categories for {col}: {e}", file=sys.stderr)


def timeseries_activity(df: pd.DataFrame, outdir: Path, dt_col: str = "DateCreated"):
    if dt_col not in df.columns:
        return
    ser = safe_parse_datetime(df[dt_col])
    if ser.isnull().all():
        return

    # Global activity counts per day/hour
    ts = pd.Series(1, index=ser).sort_index()
    # Per day
    daily = ts.resample("D").sum()
    plt.figure()
    daily.plot()
    plt.title("Activity count per day")
    plt.xlabel("Date")
    plt.ylabel("Events")
    plt.tight_layout()
    plt.savefig(outdir / "activity_per_day.png", dpi=150)
    plt.close()
    daily.to_frame(name="events").to_csv(outdir / "activity_per_day.csv")

    # Per hour
    hourly = ts.resample("H").sum()
    plt.figure()
    hourly.plot()
    plt.title("Activity count per hour")
    plt.xlabel("DateTime")
    plt.ylabel("Events")
    plt.tight_layout()
    plt.savefig(outdir / "activity_per_hour.png", dpi=150)
    plt.close()
    hourly.to_frame(name="events").to_csv(outdir / "activity_per_hour.csv")


def session_summary(df: pd.DataFrame, outdir: Path, session_col: str = "sessionId"):
    if session_col not in df.columns:
        return
    # Build a compact summary per session
    agg = {
        "UserId": (lambda s: s.astype("string").nunique() if "UserId" in df.columns else np.nan),
        "ActionShort": (lambda s: s.astype("string").nunique() if "ActionShort" in df.columns else np.nan),
        "Action": (lambda s: s.astype("string").nunique() if "Action" in df.columns else np.nan),
        "Duration": "sum" if "Duration" in df.columns else (lambda s: np.nan)
    }
    # Filter agg to existing columns
    real_agg = {}
    for k, v in agg.items():
        if k in df.columns:
            real_agg[k] = v

    summary = df.groupby(session_col).agg(real_agg).rename(columns={
        "UserId": "n_users",
        "ActionShort": "n_actionshort",
        "Action": "n_actions",
        "Duration": "duration_sum"
    })
    summary.to_csv(outdir / "session_summary.csv")


def user_summary(df: pd.DataFrame, outdir: Path, user_col: str = "UserId"):
    if user_col not in df.columns:
        return
    agg = {
        "sessionId": (lambda s: s.nunique() if "sessionId" in df.columns else np.nan),
        "ActionShort": (lambda s: s.astype("string").nunique() if "ActionShort" in df.columns else np.nan),
        "Action": (lambda s: s.astype("string").nunique() if "Action" in df.columns else np.nan),
        "Duration": "sum" if "Duration" in df.columns else (lambda s: np.nan)
    }
    real_agg = {}
    for k, v in agg.items():
        if k in df.columns:
            real_agg[k] = v

    summary = df.groupby(user_col).agg(real_agg).rename(columns={
        "sessionId": "n_sessions",
        "ActionShort": "n_actionshort",
        "Action": "n_actions",
        "Duration": "duration_sum"
    })
    summary.to_csv(outdir / "user_summary.csv")

    # Plot top 10 users by events
    counts = df[user_col].astype("string").value_counts().head(10)
    plt.figure(figsize=(10, 4))
    plt.bar(counts.index.tolist(), counts.values.tolist())
    plt.title("Top 10 users by event count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "top10_users_by_events.png", dpi=150)
    plt.close()


def crosstabs(df: pd.DataFrame, outdir: Path, row: str = "ActionShort", col: str = "Phase"):
    if row not in df.columns or col not in df.columns:
        return
    tab = pd.crosstab(df[row].astype("string"), df[col].astype("string"))
    tab.to_csv(outdir / "crosstab_ActionShort_by_Phase.csv")

    # Plot top rows if small
    if tab.shape[0] <= 30 and tab.shape[1] <= 20:
        plt.figure(figsize=(max(6, tab.shape[1] * 0.7), max(4, tab.shape[0] * 0.3)))
        # Render as stacked bars along columns per row?
        # For compactness, plot column sums
        colsum = tab.sum(axis=0)
        plt.bar(colsum.index.tolist(), colsum.values.tolist())
        plt.title("Total events per Phase")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(outdir / "events_per_phase.png", dpi=150)
        plt.close()


def write_schema_report(df: pd.DataFrame, outdir: Path):
    report = []
    report.append(f"Shape: {df.shape}")
    report.append("\nDtypes:\n" + df.dtypes.to_string())
    report.append("\nMissing values per column:\n" + df.isnull().sum().to_string())
    # Numeric stats
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        desc = df[num_cols].describe().to_string()
        report.append("\n\nNumeric summary:\n" + desc)
    # Categorical stats (top-10 frequency)
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        report.append("\n\nCategorical top-10 frequencies:")
        for c in cat_cols:
            vc = df[c].astype("string").value_counts().head(10)
            report.append(f"\n- {c}:\n{vc.to_string()}")
    save_text("\n".join(report), outdir / "schema_report.txt")


def main(args=None):
    parser = argparse.ArgumentParser(description="EDA for Student Activity Dataset")
    parser.add_argument("--input", required=True, help="Path to CSV input file")
    parser.add_argument("--outdir", default="eda_output", help="Directory to save outputs")
    parser.add_argument("--dtcol", default="DateCreated", help="Datetime column name (default: DateCreated)")
    parser.add_argument("--dtend", default="ActionEnd", help="Optional end datetime col (parsed if present)")
    parsed = parser.parse_args(args=args)

    in_path = Path(parsed.input)
    outdir = Path(parsed.outdir)
    ensure_outdir(outdir)

    # Detect delimiter & read
    delim = detect_delimiter(in_path)
    print(f"[INFO] Detected delimiter: {delim!r}")
    df = pd.read_csv(in_path, sep=delim, engine="python")

    # Try to coerce some common columns
    for col in ["UserId", "Action", "ActionShort", "Phase", "REG_TYPE", "SELF0_PEER1"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    for col in ["Duration", "sessionId", "LA"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse datetimes if present
    if parsed.dtcol in df.columns:
        df[parsed.dtcol] = safe_parse_datetime(df[parsed.dtcol])
    if parsed.dtend in df.columns:
        df[parsed.dtend] = safe_parse_datetime(df[parsed.dtend])

    # Save a cleaned copy with parsed dtypes
    df.to_csv(outdir / "dataset_cleaned.csv", index=False)

    # Reports and figures
    write_schema_report(df, outdir)
    plot_missing_heatmap(df, outdir / "figures" / "missing_heatmap.png")
    hist_numeric(df, outdir / "figures")
    bar_top_categories(df, outdir / "figures")
    timeseries_activity(df, outdir / "figures", dt_col=parsed.dtcol)
    session_summary(df, outdir / "tables")
    user_summary(df, outdir / "figures")
    crosstabs(df, outdir / "tables", row="ActionShort", col="Phase")

    print(f"[DONE] EDA complete. Outputs saved in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
