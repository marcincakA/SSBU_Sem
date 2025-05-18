#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

file_path = Path("../res/SSBU25_dataset_samuel.xlsx")

try:
    df = pd.read_excel(file_path)

    print("\n📊 Info")
    print("=" * 40)
    print(f"Riadky: {df.shape[0]}, Stĺpce: {df.shape[1]}")

    print("\n🧾 Stlpce a typy")
    print("=" * 40)
    print(df.dtypes)

    print("\n🔎 Chybajuce hodnoty")
    print("=" * 40)
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(missing.sort_values(ascending=False))
    else:
        print("Žiadne chýbajúce hodnoty.")

    print("\n📐 Zakladna statistika")
    print("=" * 40)
    print(df.describe())

    print("\n📋 Hodnoty v textovych stlpcoch")
    print("=" * 40)
    for col in df.select_dtypes(include="object"):
        print(f"\n{col} (unikátne: {df[col].nunique()})")
        print(df[col].value_counts().head(5))

    print("\n🧯 Duplikaty")
    print("=" * 40)
    print(f"Počet duplikátov: {df.duplicated().sum()}")

    print("\n✅ Analýza dokončená.")

except Exception as e:
    print(f"❌ Chyba počas analýzy: {e}")