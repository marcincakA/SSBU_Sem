#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path):
    df = pd.read_excel(path)
    hfe_cols = [col for col in df.columns if 'hfe' in col.lower()]
    if not hfe_cols:
        raise ValueError("No HFE mutation columns found.")
    return df, hfe_cols

def analyze_genotypes(df, hfe_cols):
    print("\n=== GENOTYPE DISTRIBUTION ===")
    for col in hfe_cols:
        counts = df[col].value_counts()
        total = len(df)
        print(f"\n{col}:")
        for gt, c in counts.items():
            print(f"  {gt}: {c} ({c/total*100:.2f}%)")

def classify_risk(row):
    c282y = next((row[c] for c in row.index if 'C282Y' in c), None)
    h63d  = next((row[c] for c in row.index if 'H63D' in c), None)
    s65c  = next((row[c] for c in row.index if 'S65C' in c), None)

    if not all([c282y, h63d, s65c]): return "Unknown"
    if c282y == 'mutant': return "High Risk"
    if c282y == h63d == 'heterozygot': return "Moderate Risk"
    if c282y == s65c == 'heterozygot': return "Moderate Risk"
    if h63d == s65c == 'heterozygot': return "Lower Risk"
    if h63d == 'mutant' or s65c == 'mutant': return "Lower Risk"
    if c282y == 'heterozygot': return "Carrier C282Y"
    if h63d == 'heterozygot': return "Carrier H63D"
    if s65c == 'heterozygot': return "Carrier S65C"
    return "Minimal Risk"

def analyze_risk(df):
    print("\n=== HH RISK DISTRIBUTION ===")
    df['HH_Risk'] = df.apply(classify_risk, axis=1)
    counts = df['HH_Risk'].value_counts()
    total = len(df)
    for k, v in counts.items():
        print(f"{k}: {v} ({v/total*100:.2f}%)")

def plot_distribution(df, cols, risk=False, out=None):
    if risk:
        data = df['HH_Risk'].value_counts()
        title = "HH Risk Distribution"
        fname = "hh_risk.png"
    else:
        for col in cols:
            data = df[col].value_counts()
            plt.figure(figsize=(8,6))
            sns.barplot(x=data.index, y=data.values)
            plt.title(f"{col} Genotype Distribution")
            plt.ylabel("Count")
            plt.xticks(rotation=30)
            if out:
                plt.savefig(Path(out)/f"{col.replace(' ', '_')}.png")
            else:
                plt.show()
            plt.close()
        return
    plt.figure(figsize=(10,6))
    sns.barplot(x=data.index, y=data.values)
    plt.title(title)
    plt.xticks(rotation=45)
    if out:
        plt.savefig(Path(out)/fname)
    else:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", default="../res/SSBU25_dataset_samuel.xlsx")
    parser.add_argument("--plots", "-g", action="store_true")
    parser.add_argument("--output", "-o")
    args = parser.parse_args()

    df, hfe_cols = load_data(args.path)
    print(f"Loaded: {args.path}\nHFE columns: {', '.join(hfe_cols)}")

    analyze_genotypes(df, hfe_cols)
    analyze_risk(df)

    if args.plots:
        plot_distribution(df, hfe_cols, out=args.output)
        plot_distribution(df, hfe_cols, risk=True, out=args.output)

if __name__ == "__main__":
    main()
