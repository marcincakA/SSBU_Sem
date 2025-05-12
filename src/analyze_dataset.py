#!/usr/bin/env python3
# Jednoduchá analýza SSBU25_dataset.xlsx

import pandas as pd
from pathlib import Path

# Cesta k súboru
file_path = Path("../res/SSBU25_dataset_cleaned.xlsx")
print(f"🔍 Načítavam dataset z: {file_path}")

try:
    # Načítanie dát
    df = pd.read_excel(file_path)

    print("\n📊 ZÁKLADNÉ INFO")
    print("=" * 40)
    print(f"Riadky: {df.shape[0]}, Stĺpce: {df.shape[1]}")

    print("\n🧾 STĹPCE A TYPY")
    print("=" * 40)
    print(df.dtypes)

    print("\n🔎 CHÝBAJÚCE HODNOTY")
    print("=" * 40)
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(missing.sort_values(ascending=False))
    else:
        print("Žiadne chýbajúce hodnoty.")

    print("\n📐 ZÁKLADNÁ ŠTATISTIKA (číselné stĺpce)")
    print("=" * 40)
    print(df.describe())

    print("\n📋 HODNOTY V TEXTOVÝCH STĹPICOCH")
    print("=" * 40)
    for col in df.select_dtypes(include="object"):
        print(f"\n{col} (unikátne: {df[col].nunique()})")
        print(df[col].value_counts().head(5))

    print("\n🧯 DUPLIKÁTY")
    print("=" * 40)
    print(f"Počet duplikátov: {df.duplicated().sum()}")

    print("\n✅ Analýza dokončená.")

except Exception as e:
    print(f"❌ Chyba počas analýzy: {e}")

