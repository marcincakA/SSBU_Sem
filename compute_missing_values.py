#!/usr/bin/env python3
# Script to impute missing validovany_vysledok values in SSBU25_dataset_modified.xlsx
# and save as DD.MM.YYYY string format

import pandas as pd
from pathlib import Path

# Define input and output file paths
input_file = Path("SSBU25_dataset_modified.xlsx")
output_file = Path("SSBU25_dataset_imputed.xlsx")

print(f"Loading dataset from {input_file.absolute()}")

try:
    # Read the Excel file
    df = pd.read_excel(input_file)
    
    # Convert date and time columns to datetime for imputation
    df['cas_prijmu_full'] = pd.to_datetime(
        df['prijem_vzorky'].astype(str) + ' ' + df['cas_prijmu'].astype(str),
        format='%d.%m.%Y %H:%M', errors='coerce'
    )
    df['validovany_vysledok'] = pd.to_datetime(
        df['validovany_vysledok'].astype(str),
        format='%d.%m.%Y', errors='coerce'
    )
    
    # Calculate median time difference for non-missing rows
    non_missing = df.dropna(subset=['validovany_vysledok', 'cas_prijmu_full'])
    time_diffs = (non_missing['validovany_vysledok'] - non_missing['cas_prijmu_full']).dt.total_seconds() / 3600  # in hours
    median_time_diff_hours = time_diffs.median()
    print(f"Median time difference (hours): {median_time_diff_hours:.2f}")
    
    # Impute missing validovany_vysledok values
    missing_indices = df[df['validovany_vysledok'].isnull()].index
    if len(missing_indices) > 0:
        print(f"Found {len(missing_indices)} missing validovany_vysledok values at rows: {missing_indices.tolist()}")
        for idx in missing_indices:
            if pd.notnull(df.loc[idx, 'cas_prijmu_full']):
                imputed_value = df.loc[idx, 'cas_prijmu_full'] + pd.Timedelta(hours=median_time_diff_hours)
                df.loc[idx, 'validovany_vysledok'] = imputed_value
                print(f"Row {idx}: Imputed validovany_vysledok = {imputed_value.strftime('%d.%m.%Y')}")
            else:
                print(f"Row {idx}: Cannot impute (missing cas_prijmu_full)")
    else:
        print("No missing validovany_vysledok values found.")
    
    # Convert validovany_vysledok to DD.MM.YYYY string format
    df['validovany_vysledok'] = df['validovany_vysledok'].dt.strftime('%d.%m.%Y')
    
    # Drop temporary column
    df = df.drop(columns=['cas_prijmu_full'])
    
    # Save the updated dataset
    df.to_excel(output_file, index=False)
    print(f"\nImputation complete. Saved to {output_file.absolute()}")
    
except Exception as e:
    print(f"Error processing the dataset: {e}")