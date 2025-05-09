#!/usr/bin/env python3
# Script to remove the second column from SSBU25_dataset.xls

import pandas as pd
import numpy as np
from pathlib import Path

# Input and output file paths
input_file = Path("Info/SSBU25_dataset.xls")
output_file = Path("SSBU25_dataset_modified.xlsx")

print(f"Loading dataset from {input_file.absolute()}")

try:
    # Read the Excel file with ID column as string to preserve leading zeros
    # ID format: YMMDDNNNN where:
    # Y = last digit of the year from prijem vzorky
    # MM = month of prijem vzorky
    # DD = day of prijem vzorky
    # NNNN = order of patient based on reception time
    df = pd.read_excel(input_file, dtype={'id': str})
    
    # If the ID column wasn't properly detected, try to find it by column name or position
    if 'id' not in df.columns and any(col.lower() == 'id' for col in df.columns):
        id_col = next(col for col in df.columns if col.lower() == 'id')
    elif 'id' not in df.columns:
        # Assuming the first column is the ID
        id_col = df.columns[0]
    else:
        id_col = 'id'
    
    # Display original column information
    print("\nOriginal columns:")
    for i, col in enumerate(df.columns):
        print(f"Column {i}: '{col}'")
        if i == 0:  # Assuming first column is ID
            print(f"  Sample ID values: {df[col].head(3).tolist()}")
    
    # Remove the second column (index 1)
    second_col = df.columns[1]
    df = df.drop(columns=[second_col])
    print(f"\nRemoved column at index 1: '{second_col}'")
    
    # Display updated column information
    print("\nRemaining columns:")
    for i, col in enumerate(df.columns):
        print(f"Column {i}: '{col}'")

    # Rename columns at indices 2 and 4
    # Get the current column names at those indices
    col_index_2 = df.columns[2]
    col_index_4 = df.columns[4]
    
    # Create a mapping for renaming
    rename_mapping = {
        col_index_2: 'cas_validacie',  # Rename column at index 2 to "cas_validacie"
        col_index_4: 'cas_prijmu'      # Rename column at index 4 to "cas_prijmu"
    }
    
    # Apply the renaming
    df = df.rename(columns=rename_mapping)
    print(f"\nRenamed columns:")
    print(f"- Column at index 2: '{col_index_2}' → 'cas_validacie'")
    print(f"- Column at index 4: '{col_index_4}' → 'cas_prijmu'")
    
    # Properly handle the ID column, preserving leading zeros but keeping missing values empty
    # Convert ID column to string but leave NaN values as is
    df[id_col] = df[id_col].astype(object)
    
    # Only add leading zeros to valid ID values (not NaN)
    mask = df[id_col].notna()
    df.loc[mask, id_col] = df.loc[mask, id_col].astype(str).str.zfill(9)
    
    # Verify ID column format
    print(f"\nID column '{id_col}' sample values after formatting:")
    print(df[id_col].head(10).tolist())
    
    # Check for any missing IDs
    missing_ids = df[id_col].isna().sum()
    if missing_ids > 0:
        print(f"Number of missing IDs: {missing_ids}")
    
    # Display final column information
    print("\nFinal columns:")
    for i, col in enumerate(df.columns):
        print(f"Column {i}: '{col}'")
    
    # Save to new file with string preservation for ID column
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
        # Apply string format to ID column in Excel to preserve leading zeros
        worksheet = writer.sheets['Sheet1']
        worksheet.column_dimensions['A'].number_format = '@'  # Text format for Excel
    
    print(f"\nModified dataset saved to {output_file.absolute()}")
    print("ID column format preserved with leading zeros, missing values left empty")

except Exception as e:
    print(f"Error processing the dataset: {e}") 