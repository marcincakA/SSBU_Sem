#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

def classify_hh_risk(row):
    c282y = h63d = s65c = None
    for col in row.index:
        if 'C282Y' in col: c282y = row[col]
        elif 'H63D' in col: h63d = row[col]
        elif 'S65C' in col: s65c = row[col]

    if not all([c282y, h63d, s65c]): return "Unknown"
    if c282y == "mutant": return "High Risk"
    if c282y == h63d == "heterozygot": return "Moderate Risk"
    if c282y == s65c == "heterozygot": return "Moderate Risk"
    if h63d == s65c == "heterozygot": return "Lower Risk"
    if h63d == "mutant": return "Lower Risk"
    if s65c == "mutant": return "Lower Risk"
    if c282y == "heterozygot": return "Carrier"
    if h63d == "heterozygot": return "Carrier"
    if s65c == "heterozygot": return "Carrier"
    return "Minimal Risk"

def preprocess_data(df, hfe_columns):
    df['HH_Risk'] = df.apply(classify_hh_risk, axis=1)
    df['Risk_Category'] = df['HH_Risk'].replace({
        'High Risk': 'High/Moderate Risk',
        'Moderate Risk': 'High/Moderate Risk',
        'Lower Risk': 'Lower Risk',
        'Carrier': 'Carrier',
    }).fillna('Minimal Risk')

    def diag_cat(d):
        if pd.isna(d): return "Unknown"
        s = str(d)
        if any(s.startswith(code) for code in ['K76.0', 'K75.9', 'K70', 'K71', 'K72', 'K73', 'K74', 'K76', 'K77']):
            return "Liver Disease"
        if s.startswith('K'): return "Other Digestive"
        if s.startswith('E'): return "Metabolic"
        if s.startswith('B'): return "Infectious"
        return "Other"

    df['Diagnosis_Category'] = df['diagnoza MKCH-10'].apply(diag_cat)

    def specific_liver(d):
        if pd.isna(d): return "Other"
        s = str(d)
        if s.startswith('K76.0'): return "K76.0 (Fatty liver)"
        if s.startswith('K75.9'): return "K75.9 (Inflammatory liver)"
        return "Other"

    df['Specific_Liver_Disease'] = df['diagnoza MKCH-10'].apply(specific_liver)

    def any_mut(row):
        return "Mutation Present" if any(row[c] != 'normal' for c in hfe_columns) else "No Mutation"

    df['Any_HFE_Mutation'] = df.apply(any_mut, axis=1)
    df['Has_K760'] = df['diagnoza MKCH-10'].astype(str).str.startswith('K76.0').astype(int)
    df['Has_K759'] = df['diagnoza MKCH-10'].astype(str).str.startswith('K75.9').astype(int)
    return df

def plot_diagnosis_associations(df, output_dir=None):
    Path(output_dir).mkdir(parents=True, exist_ok=True) if output_dir else None

    def save_or_show(name):
        if output_dir:
            plt.savefig(Path(output_dir) / name, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {name}")
        else:
            plt.tight_layout()
            plt.show()
        plt.close()

    pd.crosstab(df['Diagnosis_Category'], df['Any_HFE_Mutation'], normalize='index').mul(100).plot(kind='bar', stacked=True, figsize=(10,6))
    plt.title('Diagnosis by HFE Mutation Status')
    save_or_show("diagnosis_by_mutation.png")

    pd.crosstab(df['Specific_Liver_Disease'], df['Any_HFE_Mutation'], normalize='index').mul(100).plot(kind='bar', stacked=True, figsize=(10,6))
    plt.title('Specific Liver Disease by Mutation')
    save_or_show("liver_disease_by_mutation.png")

    liver_prev = df.groupby('Risk_Category')[['Has_K760', 'Has_K759']].mean().mul(100)
    liver_prev.rename(columns={
        'Has_K760': 'K76.0 (Fatty liver)',
        'Has_K759': 'K75.9 (Inflammatory liver)'
    }).plot(kind='bar', figsize=(10,6))
    plt.title('Liver Disease Prevalence by HH Risk')
    save_or_show("liver_disease_by_risk.png")

    high_mod = df[df['Risk_Category'] == 'High/Moderate Risk']
    if len(high_mod) > 10:
        diag_counts = high_mod['Diagnosis_Category'].value_counts()
        diag_pct = diag_counts.div(len(high_mod)).mul(100).round(1)
        ax = diag_counts.plot(kind='bar', figsize=(10,6))
        for i, p in enumerate(ax.patches):
            ax.annotate(f'{diag_counts.iloc[i]} ({diag_pct.iloc[i]}%)', (p.get_x() + p.get_width()/2., p.get_height()), ha='center', va='bottom')
        plt.title('Diagnoses in High/Moderate Risk Patients')
        save_or_show("high_risk_diagnoses.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", default="../res/SSBU25_dataset_samuel.xlsx")
    parser.add_argument("--plots", "-g", action="store_true")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    try:
        df = pd.read_excel(args.path)
        hfe_columns = [col for col in df.columns if 'hfe' in col.lower()]
        if not hfe_columns:
            print("❌ No HFE mutation columns found.")
            return

        print(f"🔍 Found HFE columns: {hfe_columns}")
        df = preprocess_data(df, hfe_columns)

        if args.plots:
            plot_diagnosis_associations(df, args.output)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()