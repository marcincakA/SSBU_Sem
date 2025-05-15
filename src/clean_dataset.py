
#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re
from openpyxl.utils import get_column_letter
from openpyxl.styles import numbers

INPUT_FILE = Path("../res/SSBU25_dataset.xlsx")
OUTPUT_FILE = Path("../res/SSBU25_dataset_samuel.xlsx")

def load_dataset(path):
    return pd.read_excel(path)

def drop_unwanted_columns(df):
    return df.drop(columns=[df.columns[1]])

def rename_columns(df):
    col_map = {}
    if len(df.columns) > 4:
        col_map[df.columns[2]] = "cas_validacie"
        col_map[df.columns[4]] = "cas_prijmu"
    return df.rename(columns=col_map)

def restore_year_prefix(df, id_col, date_col):
    def fix(row):
        raw_id = str(row[id_col]).strip()
        raw_id = re.sub(r'\.0$', '', raw_id)
        raw_id = raw_id.replace('.0', '')

        if re.fullmatch(r'\d{10}', raw_id):
            return raw_id
        if re.fullmatch(r'\d{8}', raw_id) and pd.notna(row[date_col]):
            try:
                date = pd.to_datetime(row[date_col], dayfirst=True)
                prefix = str(date.year)[-2:]
                return f"{prefix}{raw_id}"
            except:
                return raw_id
        return raw_id
    df[id_col] = df.apply(fix, axis=1)
    return df

def fix_id_format(df, id_col, old_format=False):
    def clean(val):
        if pd.isna(val):
            return None
        val = str(val).strip()
        val = re.sub(r'\.0$', '', val)
        if re.fullmatch(r'\d{10}', val):
            return val  # already valid
        return val.zfill(10 if not old_format else 9)

    df[id_col] = df[id_col].apply(clean)
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
        dt = pd.to_datetime(date_val, dayfirst=True)
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
    for col in required_cols:
        df = df[df[col].notna() & (df[col] != '')]
    return df

def save_with_text_column(df, output_path, text_column_index=0):
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
        worksheet = writer.sheets['Sheet1']
        col_letter = get_column_letter(text_column_index + 1)
        for cell in worksheet[col_letter]:
            cell.number_format = numbers.FORMAT_TEXT

def main():
    df = load_dataset(INPUT_FILE)
    df.columns = df.columns.str.strip().str.lower()
    id_col = df.columns[0]

    df = drop_unwanted_columns(df)
    df = rename_columns(df)
    date_col = find_reception_date_column(df)

    df = restore_year_prefix(df, id_col, date_col)

    id_lengths = df[id_col].dropna().astype(str).map(len)
    old_format = id_lengths.max() <= 9

    df = fix_id_format(df, id_col, old_format)
    df = reconstruct_missing_ids(df, id_col, date_col, "cas_prijmu")

    required_cols = ["validovany vysledok", "diagnoza mkch-10"] + [col for col in df.columns if "hfe" in col.lower()]
    df = clean_rows(df, required_cols)

    save_with_text_column(df, OUTPUT_FILE)
    print(f"💾 Cleaned dataset saved to {OUTPUT_FILE.absolute()}")

if __name__ == "__main__":
    main()
