from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DEFAULT_YEARS = ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21"]


def _resolve_input_path(path: str | None) -> Path:
    if path:
        return Path(path).expanduser().resolve()

    repo_root = Path(__file__).resolve().parent
    candidates = [repo_root / "results.csv", repo_root / "data" / "data.csv", repo_root / "data.csv"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No CSV input file found. Pass --input with a data file or place a CSV in the repository root."
    )


def prepare_features(df: pd.DataFrame, years: list[str]) -> pd.DataFrame:
    frame = df.copy()
    year_columns = [year for year in years if year in frame.columns]
    for year in year_columns:
        frame[year] = pd.to_numeric(frame[year], errors="coerce")

    if "mean_gen" not in frame.columns:
        if year_columns:
            frame["mean_gen"] = frame[year_columns].mean(axis=1)
        else:
            raise ValueError("No year columns were found in the input CSV. Expected e.g. 2016-17, 2017-18, ...")

    if "growth_pct" not in frame.columns:
        def growth(row: pd.Series) -> float:
            values = row[year_columns].dropna()
            if len(values) < 2:
                return np.nan
            return (values.iloc[-1] - values.iloc[0]) / values.iloc[0] * 100

        frame["growth_pct"] = frame.apply(growth, axis=1)

    feature_frame = frame[["mean_gen", "growth_pct"]].copy()
    feature_frame = feature_frame.fillna(feature_frame.median())
    return feature_frame


def select_best_k(X: np.ndarray, min_k: int, max_k: int):
    best = None
    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        if best is None or score > best[0]:
            best = (score, k, labels, km)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster state/UT waste data using K-means and hierarchical clustering.")
    parser.add_argument("--input", type=str, default=None, help="Path to the input CSV file.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for results and summaries.")
    parser.add_argument("--years", nargs="+", default=DEFAULT_YEARS, help="Year columns to aggregate across.")
    parser.add_argument("--min-k", type=int, default=2, help="Smallest cluster count to evaluate.")
    parser.add_argument("--max-k", type=int, default=6, help="Largest cluster count to evaluate.")
    args = parser.parse_args()

    input_path = _resolve_input_path(args.input)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    feature_frame = prepare_features(df, args.years)
    X = StandardScaler().fit_transform(feature_frame)

    best = select_best_k(X, args.min_k, args.max_k)
    if best is None:
        raise ValueError("Could not identify a valid clustering configuration.")

    silhouette_score_value, k, labels, km = best
    df["mean_gen"] = feature_frame["mean_gen"].values
    df["growth_pct"] = feature_frame["growth_pct"].values
    df["kmeans_cluster"] = labels
    df["hier_cluster"] = AgglomerativeClustering(n_clusters=k).fit_predict(X)

    csv_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.json"
    df.to_csv(csv_path, index=False)

    summary = {
        "n_states": len(df),
        "best_k": int(k),
        "silhouette": float(silhouette_score_value),
        "cluster_means": df.groupby("kmeans_cluster")[["mean_gen", "growth_pct"]].mean().round(2).to_dict("index"),
    }
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
