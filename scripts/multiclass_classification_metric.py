"""Multiclass classification metrics for Expedia Hotel Cluster recommendation.

Calculates multiclass metrics (Precision, Recall, F1-score, Support) for Top-1 predictions
against actual booked hotel clusters.

Compares:
  1. SMLP4Rec Final (Query Embedding + sameDest Hybrid)
  2. SMLP4Rec (+ Query Embedding only)
  3. SMLP4Rec Gốc (Plain SMLP4Rec - History only, trained on clean data)
  4. Logistic Regression Baseline
  5. ItemKNN Baseline

Data sources:
  - `data/processed/splits_csv/test.csv` (test events and baselines)
  - `data/interim/final_topk_test.npz` (model top-k predictions for SMLP4Rec variants)

Usage:
    python scripts/multiclass_classification_metric.py
    python scripts/multiclass_classification_metric.py --all-events
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = ROOT / "data" / "processed" / "splits_csv" / "test.csv"
FINAL_NPZ = ROOT / "data" / "interim" / "final_topk_test.npz"


def load_data(all_events: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Loads test data and model predictions."""
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Test CSV not found at: {TEST_CSV}")

    print(f"Loading test events from: {TEST_CSV.relative_to(ROOT)}...")
    cols = ["user_id", "item_id", "timestamp", "n_prior", "top10_clusters_lr", "top10_clusters_itemknn"]
    df = pd.read_csv(TEST_CSV, usecols=cols)
    print(f"Total test events loaded: {len(df):,}")

    # Load SMLP4Rec variants from final_topk_test.npz if available
    model_cols = []
    if FINAL_NPZ.exists():
        print(f"Loading SMLP4Rec model variants from: {FINAL_NPZ.relative_to(ROOT)}...")
        z = np.load(FINAL_NPZ)
        df["top_smlp4rec_final"] = [arr[:10].tolist() for arr in z["top_main"]]
        df["top_smlp4rec_query"] = [arr[:10].tolist() for arr in z["top_query_embedding"]]
        df["top_smlp4rec_plain_clean"] = [arr[:10].tolist() for arr in z["top_plain_all"]]
        model_cols.extend([
            ("SMLP4Rec Final (Query + sameDest)", "top_smlp4rec_final"),
            ("SMLP4Rec (+ Query Embedding)", "top_smlp4rec_query"),
            ("SMLP4Rec Goc (Plain Clean - History only)", "top_smlp4rec_plain_clean"),
        ])
    else:
        # Fallback to test.csv top10_clusters (main 3-epoch model)
        df_csv_main = pd.read_csv(TEST_CSV, usecols=["top10_clusters"])
        df["top_smlp4rec_main"] = df_csv_main["top10_clusters"].apply(lambda x: [int(i) for i in x.split()])
        model_cols.append(("SMLP4Rec (Main Model)", "top_smlp4rec_main"))

    # Parse baselines from test.csv
    df["top_lr"] = df["top10_clusters_lr"].apply(lambda x: [int(i) for i in x.split()])
    df["top_itemknn"] = df["top10_clusters_itemknn"].apply(lambda x: [int(i) for i in x.split()])
    model_cols.extend([
        ("Logistic Regression", "top_lr"),
        ("ItemKNN", "top_itemknn"),
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)

    if not all_events:
        df_eval = df.drop_duplicates("user_id", keep="first").copy()
        print(f"Evaluation mode: Deduplicated (first booking per user) -> {len(df_eval):,} users")
    else:
        df_eval = df.copy()
        print(f"Evaluation mode: All test events -> {len(df_eval):,} events")

    return df_eval, model_cols


def evaluate_top10_classes(
    df_eval: pd.DataFrame, pred_col: str, top10_classes: list[int]
) -> tuple[pd.DataFrame, float]:
    """Evaluates top-1 prediction on top-10 frequent classes with sentinel -1."""
    df_filtered = df_eval[df_eval["item_id"].isin(top10_classes)].copy()
    y_pred_top1 = df_filtered[pred_col].apply(lambda x: x[0])
    y_pred_final = y_pred_top1.apply(lambda p: p if p in top10_classes else -1)

    pct_outside = float((y_pred_final == -1).mean())
    report = classification_report(
        df_filtered["item_id"],
        y_pred_final,
        labels=top10_classes,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T
    return report_df, pct_outside


def evaluate_all_100_classes(
    df_eval: pd.DataFrame, pred_col: str
) -> dict[str, float]:
    """Evaluates top-1 prediction across all 100 hotel clusters (0-99)."""
    y_true = df_eval["item_id"]
    y_pred = df_eval[pred_col].apply(lambda x: x[0])

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Breakdown by Warm (history >= 1) and Cold (history = 0)
    warm_mask = df_eval["n_prior"] > 0
    cold_mask = df_eval["n_prior"] == 0
    acc_warm = accuracy_score(y_true[warm_mask], y_pred[warm_mask]) if warm_mask.any() else 0.0
    acc_cold = accuracy_score(y_true[cold_mask], y_pred[cold_mask]) if cold_mask.any() else 0.0

    return {
        "Top-1 Accuracy": acc,
        "Warm Acc (L>=1)": acc_warm,
        "Cold Acc (L=0)": acc_cold,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multiclass classification metric evaluation.")
    parser.add_argument(
        "--all-events",
        action="store_true",
        help="Evaluate on all test events instead of deduplicating to first event per user.",
    )
    args = parser.parse_args()

    df_eval, model_cols = load_data(all_events=args.all_events)

    # Step 1: Identify top 10 actual classes by frequency
    top10_classes = df_eval["item_id"].value_counts().head(10).index.tolist()
    print(f"\nTop 10 most frequent actual classes: {top10_classes}")

    summary_top10_rows = []
    summary_100_rows = []

    print("\n" + "=" * 76)
    print("DETAILED PER-CLASS REPORT ON TOP-10 CLASSES")
    print("=" * 76)

    for model_name, pred_col in model_cols:
        print(f"\n>>> Model: {model_name}")
        rep_df, pct_out = evaluate_top10_classes(df_eval, pred_col, top10_classes)
        print(f"% predictions falling outside top-10 classes (sentinel -1): {pct_out:.2%}")
        print(rep_df.round(4).to_string())

        macro = rep_df.loc["macro avg"]
        weighted = rep_df.loc["weighted avg"]
        summary_top10_rows.append({
            "Model": model_name,
            "% Outside Top-10": f"{pct_out:.2%}",
            "Macro Precision": macro["precision"],
            "Macro Recall": macro["recall"],
            "Macro F1": macro["f1-score"],
            "Weighted F1": weighted["f1-score"],
        })

        metrics_100 = evaluate_all_100_classes(df_eval, pred_col)
        summary_100_rows.append({
            "Model": model_name,
            **metrics_100,
        })

    # Summary comparisons
    print("\n" + "=" * 76)
    print("SUMMARY COMPARISON: TOP-10 FREQUENT CLASSES")
    print("=" * 76)
    sum_10_df = pd.DataFrame(summary_top10_rows)
    print(sum_10_df.to_string(index=False))

    print("\n" + "=" * 76)
    print("SUMMARY COMPARISON: FULL 100 CLASSES (TOP-1 CLASSIFICATION METRICS)")
    print("=" * 76)
    sum_100_df = pd.DataFrame(summary_100_rows)
    print(sum_100_df.to_string(index=False))


if __name__ == "__main__":
    main()