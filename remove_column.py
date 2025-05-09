#!/usr/bin/env python3
# Script to remove the second column from SSBU25_dataset.xls

import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime

# Input and output file paths
input_file = Path("Info/SSBU25_dataset.xls")
output_file = Path("SSBU25_dataset_modified.xlsx")

print(f"Loading dataset from {input_file.absolute()}")

try:
    # Read the Excel file with ID column as string to preserve leading zeros
    # ID format: YYMMDDNNNN where:
    # YY = last two digits of the year from prijem vzorky
    # MM = month of prijem vzorky
    # DD = day of prijem vzorky
    # NNNN = order of patient based on reception time
    df = pd.read_excel(input_file)
    
    # Print raw data types for debugging
    print("\nRaw data types at load:")
    print(df.dtypes.head())
    
    # Find ID column (likely first column)
    if 'id' in df.columns:
        id_col = 'id'
    elif any(col.lower() == 'id' for col in df.columns):
        id_col = next(col for col in df.columns if col.lower() == 'id')
    else:
        # Assuming the first column is the ID
        id_col = df.columns[0]
    
    print(f"Using '{id_col}' as the ID column")
    
    # Check for truly empty, NaN, and other problematic ID values
    total_rows = len(df)
    null_ids = df[id_col].isnull().sum()
    empty_str_ids = (df[id_col] == '').sum() if df[id_col].dtype == object else 0
    zero_ids = (df[id_col] == 0).sum() if df[id_col].dtype in [np.int64, np.float64] else 0
    
    print(f"\nID column diagnostics:")
    print(f"Total rows: {total_rows}")
    print(f"Null/NaN IDs: {null_ids}")
    print(f"Empty string IDs: {empty_str_ids}")
    print(f"Zero IDs: {zero_ids}")
    print(f"ID column dtype: {df[id_col].dtype}")
    
    # Display first few rows with problematic IDs for inspection
    print("\nSample problematic IDs:")
    problem_ids = df[(df[id_col].isnull()) | 
                    ((df[id_col] == '') & (df[id_col].dtype == object)) | 
                    ((df[id_col] == 0) & (df[id_col].dtype != object))]
    
    if not problem_ids.empty:
        print(problem_ids.head().to_string())
    else:
        print("No problematic IDs found.")
    
    # Display original column information
    print("\nOriginal columns:")
    for i, col in enumerate(df.columns):
        print(f"Column {i}: '{col}'")
        if i == 0:  # Assuming first column is ID
            non_null_values = df[col].dropna().head(3).tolist()
            print(f"  Sample ID values: {non_null_values}")
    
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
    if len(df.columns) > 4:  # Ensure columns exist
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
    
    # Determine if we need to update the format of existing IDs
    # Check if existing IDs are in the old format (YMMDDNNNN - 9 digits) or new format (YYMMDDNNNN - 10 digits)
    sample_ids = df[id_col].dropna().head(10).tolist()
    id_lengths = [len(str(id_val)) for id_val in sample_ids if str(id_val) != 'nan' and str(id_val) != '']
    
    old_format = False
    if id_lengths and max(id_lengths) <= 9:
        old_format = True
        print("\nDetected old ID format (YMMDDNNNN). Will convert to new format (YYMMDDNNNN).")
    else:
        print("\nAssuming IDs are already in the new format (YYMMDDNNNN) or no consistent format detected.")
    
    # Improved ID handling - Convert all to string and fix leading zeros
    # First convert any numeric IDs to avoid NaN issues during conversion
    if df[id_col].dtype in [np.int64, np.float64]:
        # For numeric columns, only convert non-NaN values to string
        df[id_col] = df[id_col].astype('Int64')  # nullable integer type
        df[id_col] = df[id_col].astype(str)
        # Replace 'nan' or '<NA>' strings with None
        df[id_col] = df[id_col].replace(['nan', '<NA>'], None)
    
    # Now apply zfill to non-null values only
    mask = df[id_col].notna() & (df[id_col] != '')
    if any(mask):
        # Use zfill with the appropriate length based on format
        zfill_length = 10 if not old_format else 9
        df.loc[mask, id_col] = df.loc[mask, id_col].str.zfill(zfill_length)
        
    # Re-check missing ID count after processing
    missing_ids = df[id_col].isna().sum() + (df[id_col] == '').sum()
    print(f"\nAfter processing, missing/empty IDs: {missing_ids}")
    
    # Sample of processed IDs
    print("\nSample of processed IDs:")
    sample_ids = []
    for i, id_val in enumerate(df[id_col].head(10)):
        sample_ids.append(f"Row {i}: {id_val}")
    print("\n".join(sample_ids))
    
    # Find datum_prijmu column - could be named differently
    datum_prijmu_col = None
    for col in df.columns:
        if any(term in str(col).lower() for term in ['datum_prijmu', 'prijem', 'date', 'datum']):
            if 'cas' not in str(col).lower():  # Exclude time columns
                datum_prijmu_col = col
                break
    
    if datum_prijmu_col is None:
        print("Could not identify the datum_prijmu column. Please specify the column name.")
    else:
        print(f"Using '{datum_prijmu_col}' as the reception date column")
    
    # Get rows with missing IDs - include empty strings and NaN values
    missing_id_rows = df[(df[id_col].isna()) | (df[id_col] == '')]
    
    # Function to extract date parts from a date value and construct ID prefix
    def extract_date_parts(date_val):
        if pd.isna(date_val):
            return None
        
        # Try to handle different date formats by converting to datetime
        try:
            # If already a datetime
            if isinstance(date_val, (pd.Timestamp, datetime)):
                dt = date_val
            else:
                # Try parsing string with different formats
                for fmt in ['%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d']:
                    try:
                        dt = pd.to_datetime(date_val, format=fmt)
                        break
                    except:
                        continue
                else:
                    # If none of the specific formats worked, let pandas guess
                    dt = pd.to_datetime(date_val)
            
            # Extract parts - now get last TWO digits of year instead of just one
            year_last_two_digits = str(dt.year)[-2:]  # Last two digits of year
            month = f"{dt.month:02d}"  # Month with leading zero
            day = f"{dt.day:02d}"  # Day with leading zero
            
            # Construct ID prefix: YYMMDD (now 6 characters instead of 5)
            id_prefix = f"{year_last_two_digits}{month}{day}"
            return id_prefix
            
        except Exception as e:
            print(f"Error extracting date parts from {date_val}: {e}")
            return None
    
    if len(missing_id_rows) > 0 and datum_prijmu_col is not None:
        print("\n" + "="*50)
        print("RECONSTRUCTING MISSING IDs")
        print("="*50)
        
        # Function to clean and convert time to an integer (for sorting)
        def time_to_int(time_str):
            if pd.isna(time_str):
                return 999999  # High value for missing times (will be ordered last)
            # Try to clean and standardize the time string
            cleaned = str(time_str).replace(':', '').replace(' ', '').strip()
            # Extract just the digits
            digits = re.sub(r'\D', '', cleaned)
            if digits:
                return int(digits)
            return 999999
        
        # Track how many IDs were successfully reconstructed
        reconstructed_count = 0
        solo_reconstructed_count = 0
        
        # Process each missing ID row
        for idx, row in missing_id_rows.iterrows():
            reception_date = row[datum_prijmu_col]
            
            if pd.isna(reception_date):
                print(f"Row {idx}: Cannot reconstruct ID - missing reception date")
                continue
            
            # Find all rows with the same reception date
            same_date_rows = df[df[datum_prijmu_col] == reception_date].copy()
            
            # METHOD 1: If multiple records with same date, use ranking by time
            if len(same_date_rows) >= 2:
                # Extract valid IDs from same date (to use as template)
                valid_ids = same_date_rows[(same_date_rows[id_col].notna()) & (same_date_rows[id_col] != '')][id_col]
                
                if not valid_ids.empty:
                    # Extract a template ID - check if we need to use old or new format logic
                    template_id = valid_ids.iloc[0]
                    if old_format and len(template_id) == 9:
                        # For old format, construct new format prefix
                        id_prefix = extract_date_parts(reception_date)
                        if not id_prefix:
                            print(f"Row {idx}: Failed to extract date parts from {reception_date}")
                            continue
                    else:
                        # For new format, use first 6 characters (YYMMDD)
                        id_prefix = template_id[:6]  # Extract YYMMDD part
                    
                    # Add time processing for ordering
                    same_date_rows['time_int'] = same_date_rows['cas_prijmu'].apply(time_to_int)
                    
                    # Add original row position as tie-breaker for identical timestamps
                    same_date_rows['orig_position'] = range(len(same_date_rows))
                    
                    # Sort by time_int first, then by original position
                    same_date_rows = same_date_rows.sort_values(['time_int', 'orig_position'])
                    
                    # Add a rank column (1-based)
                    same_date_rows['rank'] = range(1, len(same_date_rows) + 1)
                    
                    # Print time ordering info for debugging
                    print(f"\nTime ordering for date {reception_date}:")
                    for _, r in same_date_rows.iterrows():
                        rank = r['rank']
                        time_val = r['cas_prijmu']
                        time_int = r['time_int']
                        orig_pos = r['orig_position']
                        r_id = r[id_col]
                        print(f"  Rank {rank}: Time {time_val} → {time_int} → Position {orig_pos} → ID: {r_id}")
                    
                    # Find the rank of the current missing ID row
                    missing_row_in_sorted = same_date_rows[same_date_rows.index == idx]
                    
                    if not missing_row_in_sorted.empty:
                        rank = missing_row_in_sorted['rank'].iloc[0]
                        
                        # Create a new ID by combining the prefix with the rank (as 4 digits with leading zeros)
                        new_id = f"{id_prefix}{rank:04d}"
                        
                        print(f"Row {idx}: Reconstructed ID {new_id} from template/date with rank {rank}")
                        
                        # Update the ID in the dataframe
                        df.at[idx, id_col] = new_id
                        reconstructed_count += 1
                        continue
                    else:
                        print(f"Row {idx}: Error finding row in sorted data")
                else:
                    print(f"Row {idx}: No valid IDs found for date {reception_date} to use as template")
            
            # METHOD 2: If this is the only record for this date or METHOD 1 failed, 
            # directly extract date parts and create an ID
            # Extract date parts to create ID prefix
            id_prefix = extract_date_parts(reception_date)
            
            if id_prefix:
                # Since this is the only record or we couldn't use ranking, use 0001 as default
                new_id = f"{id_prefix}0001"
                print(f"Row {idx}: Created ID {new_id} directly from date {reception_date}")
                
                # Update the ID in the dataframe
                df.at[idx, id_col] = new_id
                solo_reconstructed_count += 1
                continue
            else:
                print(f"Row {idx}: Failed to extract date parts from {reception_date}")
        
        print(f"\nReconstruction complete.")
        print(f"- IDs reconstructed using time ranking: {reconstructed_count}")
        print(f"- IDs reconstructed directly from date: {solo_reconstructed_count}")
        print(f"- Total reconstructed: {reconstructed_count + solo_reconstructed_count} of {len(missing_id_rows)}")
    
    # If we need to convert existing IDs from old to new format
    if old_format:
        print("\n" + "="*50)
        print("CONVERTING EXISTING IDs FROM OLD TO NEW FORMAT")
        print("="*50)
        
        # Get rows with valid IDs in old format
        valid_id_mask = df[id_col].notna() & (df[id_col] != '')
        old_format_ids = valid_id_mask & df[id_col].str.len().isin([8, 9])  # Allow for 8 or 9 digits
        
        if old_format_ids.sum() > 0:
            convert_count = 0
            
            for idx, row in df[old_format_ids].iterrows():
                old_id = row[id_col]
                reception_date = row[datum_prijmu_col]
                
                if pd.isna(reception_date):
                    print(f"Row {idx}: Cannot convert ID {old_id} - missing reception date")
                    continue
                
                # Extract new date prefix (YYMMDD)
                new_prefix = extract_date_parts(reception_date)
                
                if not new_prefix:
                    print(f"Row {idx}: Failed to extract date parts from {reception_date} for ID {old_id}")
                    continue
                
                # Extract sequence number from old ID (last 4 digits)
                seq_num = old_id[-4:]
                
                # Create new ID
                new_id = f"{new_prefix}{seq_num}"
                
                # Update ID
                df.at[idx, id_col] = new_id
                convert_count += 1
                
                # Print sample conversions (first 5)
                if convert_count <= 5:
                    print(f"Row {idx}: Converted {old_id} → {new_id}")
            
            print(f"\nConverted {convert_count} IDs from old format to new format.")
    
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
    print("ID column format preserved with leading zeros, missing values reconstructed where possible")
    
    # For each missing ID that couldn't be reconstructed, print all values with the same datum_prijmu
    still_missing_ids = df[(df[id_col].isna()) | (df[id_col] == '')]
    if len(still_missing_ids) > 0:
        print("\n" + "="*50)
        print(f"REMAINING MISSING IDs: {len(still_missing_ids)}")
        print("="*50)
        
        for idx, row in still_missing_ids.iterrows():
            reception_date = row[datum_prijmu_col]
            
            if pd.isna(reception_date):
                print(f"Row {idx}: Missing ID and missing reception date")
                continue
            
            print(f"\nRow {idx}: Could not reconstruct ID for reception date {reception_date}")
            # Add additional diagnostic info here if needed

except Exception as e:
    import traceback
    print(f"Error processing the dataset: {e}")
    print("\nDetailed error information:")
    print(traceback.format_exc()) 