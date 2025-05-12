#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re

INPUT_FILE = Path("../res/SSBU25_dataset.xlsx")
OUTPUT_FILE = Path("../res/SSBU25_dataset_cleaned.xlsx")

def load_dataset(path):
    return pd.read_excel(path)

def drop_unwanted_columns(df):
    return df.drop(columns=[df.columns[1]])

def rename_columns(df):
    col_map = {}
    if len(df.columns) > 4:
        col_map[df.columns[2]] = "cas_validacie"
        col_map[df.columns[4]] = "cas_prijmu"
    df = df.rename(columns=col_map)
    return df

def fix_id_format(df, id_col, old_format=False):
    df[id_col] = df[id_col].astype(str).replace(['nan', '<NA>'], None)
    df.loc[df[id_col].notna(), id_col] = df.loc[df[id_col].notna(), id_col].str.zfill(10 if not old_format else 9)
    return df

def find_reception_date_column(df):
    for col in df.columns:
        if "prijem" in col.lower() and "cas" not in col.lower():
            return col
    raise ValueError("❌ Nepodarilo sa nájsť stĺpec s dátumom prijatia.")

def extract_date_prefix(date_val):
    if pd.isna(date_val):
        return None
    try:
        dt = pd.to_datetime(date_val)
        return f"{str(dt.year)[-2:]}{dt.month:02d}{dt.day:02d}"
    except:
        return None

def reconstruct_missing_ids(df, id_col, date_col, time_col):
    mask = df[id_col].isna() | (df[id_col] == '')
    for idx in df[mask].index:
        date = df.at[idx, date_col]
        if pd.isna(date):
            continue
        prefix = extract_date_prefix(date)
        same_date = df[df[date_col] == date].copy()
        same_date["time_rank"] = same_date[time_col].astype(str).apply(
            lambda t: int(re.sub(r'\D', '', t)) if pd.notna(t) else 999999)
        same_date = same_date.sort_values(by="time_rank").reset_index()
        rank = same_date[same_date["index"] == idx].index[0] + 1
        df.at[idx, id_col] = f"{prefix}{rank:04d}"
    return df

def clean_rows(df, required_cols):
    initial = len(df)
    for col in required_cols:
        df = df[df[col].notna() & (df[col] != '')]
    return df

def main():
    df = load_dataset(INPUT_FILE)
    df.columns = df.columns.str.strip().str.lower()
    id_col = df.columns[0]

    df = drop_unwanted_columns(df)
    df = rename_columns(df)

    # Detect old ID format
    id_lengths = df[id_col].dropna().astype(str).map(len)
    old_format = id_lengths.max() <= 9

    df = fix_id_format(df, id_col, old_format)
    date_col = find_reception_date_column(df)
    df = reconstruct_missing_ids(df, id_col, date_col, "cas_prijmu")

    required_cols = ["validovany vysledok", "diagnoza mkch-10"] + [col for col in df.columns if "hfe" in col.lower()]
    df = clean_rows(df, required_cols)

    df.to_excel(OUTPUT_FILE, index=False)
    print(f"💾 Cleaned dataset saved to {OUTPUT_FILE.absolute()}")

if __name__ == "__main__":
    main()
