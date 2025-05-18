#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

def get_hfe_columns(df):
    return [col for col in df.columns if 'hfe' in str(col).lower()]

def classify_risk(row):
    c282y = row.get('HFE G845A (C282Y)[HFE]', None)
    h63d = row.get('HFE C187G (H63D)[HFE]', None)
    s65c = row.get('HFE A193T (S65C)[HFE]', None)
    
    if not all([c282y, h63d, s65c]): return "Unknown"
    if c282y == "mutant": return "High Risk"
    if c282y == h63d == "heterozygot": return "Moderate Risk"
    if any(x == "heterozygot" for x in [c282y, h63d, s65c]): return "Carrier"
    if h63d == "mutant" or s65c == "mutant": return "Lower Risk"
    return "Minimal Risk"

def simplify_diag(diag):
    if pd.isna(diag): return "Unknown"
    if str(diag).startswith("K76.0"): return "K76.0 (Fatty liver)"
    if str(diag).startswith("K75.9"): return "K75.9 (Inflammatory)"
    if str(diag).startswith("K"): return "Other Digestive"
    if str(diag).startswith("E"): return "Endocrine"
    if str(diag).startswith("B"): return "Infectious"
    return "Other"

def run_analysis(df, hfe_columns):
    print("\nRunning analysis...")
    df['Risk'] = df.apply(classify_risk, axis=1)
    df['Diagnosis'] = df['diagnoza MKCH-10'].apply(simplify_diag)
    df['Mutation_Present'] = df[hfe_columns].apply(lambda r: any(g != 'normal' for g in r), axis=1)
    df['Mutation_Present'] = df['Mutation_Present'].map({True: 'Yes', False: 'No'})

    print("\nRisk category counts:")
    print(df['Risk'].value_counts())

    print("\nDiagnosis vs Mutation Status:")
    table = pd.crosstab(df['Diagnosis'], df['Mutation_Present'])
    print(table)

    chi2, p, _, _ = chi2_contingency(table)
    print(f"\nChi2: {chi2:.2f}, p-value: {p:.4f}")

    return df

def plot_heatmap(df, output_dir):
    table = pd.crosstab(df['Diagnosis'], df['Risk'])
    sns.heatmap(table, annot=True, fmt="d", cmap="Blues")
    plt.title("Diagnosis vs Risk Category")
    plt.tight_layout()
    if output_dir:
        path = Path(output_dir) / "diagnosis_vs_risk.png"
        plt.savefig(path)
        print(f"Plot saved to {path}")
    else:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", default="../res/SSBU25_dataset_adam.xlsx")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    df = pd.read_excel(args.path)
    hfe_cols = get_hfe_columns(df)
    if not hfe_cols:
        print("No HFE columns found.")
        return

    print("HFE columns:", hfe_cols)
    df = run_analysis(df, hfe_cols)

    if args.output:
        Path(args.output).mkdir(exist_ok=True, parents=True)
        df.to_excel(Path(args.output) / "hfe_analysis_results.xlsx", index=False)

    plot_heatmap(df, args.output)

if __name__ == "__main__":
    main()