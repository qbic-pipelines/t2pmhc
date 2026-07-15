#!/usr/bin/env python3
"""
Analyse peptide-grouped CV experiment results.
Compares 5-fold peptide-grouped CV AUROC between models with and without PAE features.
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(EXP_DIR, "models")

CONDITIONS = {
    "GAT + PAE": "gat_with_pae",
    "GAT - PAE": "gat_no_pae",
    "GCN + PAE": "gcn_with_pae",
    "GCN - PAE": "gcn_no_pae",
}


def load_results():
    """Load per-fold AUROC results for all conditions."""
    results = {}
    for label, run_name in CONDITIONS.items():
        auroc_path = os.path.join(MODELS_DIR, run_name, f"{run_name}_cv_auroc.csv")
        if not os.path.exists(auroc_path):
            print(f"WARNING: Missing results for {label} at {auroc_path}")
            continue
        df = pd.read_csv(auroc_path)
        results[label] = df
    return results


def load_peptide_folds():
    """Load peptide fold assignments from the first available condition."""
    for run_name in CONDITIONS.values():
        folds_path = os.path.join(MODELS_DIR, run_name, f"{run_name}_peptide_folds.tsv")
        if os.path.exists(folds_path):
            return pd.read_csv(folds_path, sep="\t")
    return None


def print_summary(results, peptide_folds_df):
    """Print summary table of per-fold AUROCs with validation peptides."""
    print("\n" + "=" * 80)
    print("Peptide-Grouped CV — 5-fold AUROC Summary")
    print("=" * 80)

    # Build fold -> validation peptides mapping
    fold_peptides = {}
    if peptide_folds_df is not None:
        for fold in sorted(peptide_folds_df["fold"].unique()):
            fold_df = peptide_folds_df[peptide_folds_df["fold"] == fold]
            peptides = fold_df.sort_values("n_samples", ascending=False)["peptide"].values
            total = fold_df["n_samples"].sum()
            fold_peptides[fold] = {"peptides": peptides, "total": total}

    rows = []
    for label in CONDITIONS:
        if label not in results:
            continue
        df = results[label]
        aurocs = df["auroc"].values
        row = {"Condition": label}
        for i, a in enumerate(aurocs, 1):
            row[f"Fold {i}"] = f"{a:.4f}"
        row["Mean"] = f"{aurocs.mean():.4f}"
        row["Std"] = f"{aurocs.std():.4f}"
        rows.append(row)

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    # Print validation peptides per fold
    if fold_peptides:
        print("\nValidation peptides per fold:")
        for fold, info in fold_peptides.items():
            peptide_list = ", ".join(info["peptides"])
            print(f"  Fold {fold} ({info['total']} samples): {peptide_list}")

    return summary


def paired_comparison(results):
    """Paired comparison between with-PAE and no-PAE for each model type."""
    print("\n" + "-" * 80)
    print("Paired Comparisons (Wilcoxon signed-rank test)")
    print("-" * 80)

    comparisons = [
        ("GAT + PAE", "GAT - PAE", "GAT"),
        ("GCN + PAE", "GCN - PAE", "GCN"),
    ]

    for with_label, without_label, model_name in comparisons:
        if with_label not in results or without_label not in results:
            print(f"\n{model_name}: SKIPPED (missing results)")
            continue

        with_auroc = results[with_label]["auroc"].values
        without_auroc = results[without_label]["auroc"].values
        diff = with_auroc - without_auroc

        print(f"\n{model_name}:")
        print(f"  With PAE:    {with_auroc.mean():.4f} +/- {with_auroc.std():.4f}")
        print(f"  Without PAE: {without_auroc.mean():.4f} +/- {without_auroc.std():.4f}")
        print(f"  Difference:  {diff.mean():.4f} +/- {diff.std():.4f}")

        if len(diff) >= 5:
            stat, p_value = stats.wilcoxon(with_auroc, without_auroc)
            print(f"  Wilcoxon p-value: {p_value:.4f}")
        else:
            print("  (Too few folds for Wilcoxon test)")


def verify_splits(results):
    """Verify that all conditions used the same peptide fold assignments."""
    print("\n" + "-" * 80)
    print("Split Verification")
    print("-" * 80)

    splits = {}
    for label, run_name in CONDITIONS.items():
        splits_path = os.path.join(MODELS_DIR, run_name, f"{run_name}_cv_splits.tsv")
        if os.path.exists(splits_path):
            splits[label] = pd.read_csv(splits_path, sep="\t")

    if len(splits) < 2:
        print("Not enough split files to compare.")
        return

    ref_label = list(splits.keys())[0]
    ref_df = splits[ref_label].sort_values(["val_fold", "identifier"]).reset_index(drop=True)

    for label, df in splits.items():
        if label == ref_label:
            continue
        other_df = df.sort_values(["val_fold", "identifier"]).reset_index(drop=True)
        if ref_df.equals(other_df):
            print(f"  {ref_label} vs {label}: IDENTICAL splits")
        else:
            print(f"  {ref_label} vs {label}: DIFFERENT splits (WARNING!)")


def main():
    results = load_results()

    if not results:
        print("No results found. Have you run all 4 conditions?")
        sys.exit(1)

    peptide_folds_df = load_peptide_folds()
    summary = print_summary(results, peptide_folds_df)
    paired_comparison(results)
    verify_splits(results)

    # Save summary
    out_path = os.path.join(EXP_DIR, "results_summary.csv")
    summary.to_csv(out_path, index=False)
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
