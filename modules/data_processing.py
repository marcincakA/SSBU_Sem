#!/usr/bin/env python3
# Data processing module for HFE mutation analysis

import pandas as pd
import numpy as np
import re
from datetime import datetime

def clean_dataset(df, remove_blank_validovany=True, remove_blank_diagnoza=True, 
                 remove_blank_hfe=True, remove_second_column=True, 
                 min_age=0, max_age=250, filter_by_age=False):
    """
    Clean the dataset by fixing ID format and removing rows with blank values in specified columns,
    following same approach as format_dataset.py script
    """
    results = []
    results.append("Starting dataset cleaning...")
    results.append(f"Initial rows: {len(df)}")
    
    # Find ID column (likely first column)
    if 'id' in df.columns:
        id_col = 'id'
    elif any(col.lower() == 'id' for col in df.columns):
        id_col = next(col for col in df.columns if col.lower() == 'id')
    else:
        # Assuming the first column is the ID
        id_col = df.columns[0]
    
    results.append(f"Using '{id_col}' as the ID column")
    
    # Display original column information
    results.append("Original columns:")
    for i, col in enumerate(df.columns):
        results.append(f"Column {i}: '{col}'")
    
    # Remove the second column (index 1) if it exists and removal is requested
    if remove_second_column and len(df.columns) > 1:
        second_col = df.columns[1]
        df = df.drop(columns=[second_col])
        results.append(f"Removed column at index 1: '{second_col}'")
    
    # Rename columns at indices 2 and 4 if they exist
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
        results.append(f"Renamed columns:")
        results.append(f"- Column at index 2: '{col_index_2}' → 'cas_validacie'")
        results.append(f"- Column at index 4: '{col_index_4}' → 'cas_prijmu'")
    
    # Display updated column information
    results.append("Columns after modifications:")
    for i, col in enumerate(df.columns):
        results.append(f"Column {i}: '{col}'")
    
    # Check for problematic ID values
    total_rows = len(df)
    null_ids = df[id_col].isnull().sum()
    empty_str_ids = (df[id_col] == '').sum() if df[id_col].dtype == object else 0
    zero_ids = (df[id_col] == 0).sum() if df[id_col].dtype in [np.int64, np.float64] else 0
    
    results.append(f"ID column diagnostics:")
    results.append(f"Total rows: {total_rows}")
    results.append(f"Null/NaN IDs: {null_ids}")
    results.append(f"Empty string IDs: {empty_str_ids}")
    results.append(f"Zero IDs: {zero_ids}")
    results.append(f"ID column dtype: {df[id_col].dtype}")
    
    # Initial ID conversion for numeric columns
    if df[id_col].dtype in [np.int64, np.float64]:
        # For numeric columns, only convert non-NaN values to string
        df[id_col] = df[id_col].astype('Int64')  # nullable integer type
        df[id_col] = df[id_col].astype(str)
        # Replace 'nan' or '<NA>' strings with None
        df[id_col] = df[id_col].replace(['nan', '<NA>'], None)
        results.append("Converted ID column to string format and fixed NaN values")
    
    # Determine if IDs are in old format (YMMDDNNNN - 9 digits) or new format (YYMMDDNNNN - 10 digits)
    sample_ids = df[id_col].dropna().head(10).tolist()
    id_lengths = [len(str(id_val)) for id_val in sample_ids if str(id_val) != 'nan' and str(id_val) != '']
    
    old_format = False
    if id_lengths and max(id_lengths) <= 9:
        old_format = True
        results.append("Detected old ID format (YMMDDNNNN). Will convert to new format (YYMMDDNNNN).")
    else:
        results.append("Assuming IDs are already in the new format (YYMMDDNNNN) or no consistent format detected.")
    
    # Apply zfill to non-null values to preserve leading zeros
    mask = df[id_col].notna() & (df[id_col] != '')
    if any(mask):
        # Use zfill with the appropriate length based on format
        zfill_length = 9 if old_format else 10
        df.loc[mask, id_col] = df.loc[mask, id_col].str.zfill(zfill_length)
        results.append(f"Applied {zfill_length}-digit zero padding to IDs")
    
    # Find reception date column
    datum_prijmu_col = None
    for col in df.columns:
        if any(term in str(col).lower() for term in ['datum_prijmu', 'prijem', 'date', 'datum']):
            if 'cas' not in str(col).lower():  # Exclude time columns
                datum_prijmu_col = col
                break
    
    if datum_prijmu_col:
        results.append(f"Using '{datum_prijmu_col}' as the reception date column")
    else:
        results.append("Could not identify the reception date column")
    
    # Define function to extract date parts and construct ID prefix
    def extract_date_parts(date_val):
        if pd.isna(date_val):
            return None
        
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
            
            # Extract parts - get last TWO digits of year
            year_last_two_digits = str(dt.year)[-2:]
            month = f"{dt.month:02d}"  # Month with leading zero
            day = f"{dt.day:02d}"  # Day with leading zero
            
            # Construct ID prefix: YYMMDD (6 characters)
            id_prefix = f"{year_last_two_digits}{month}{day}"
            return id_prefix
        except Exception as e:
            results.append(f"Error extracting date parts: {e}")
            return None
    
    # Get rows with missing IDs - include empty strings and NaN values
    missing_id_rows = df[(df[id_col].isna()) | (df[id_col] == '')]
    
    # Reconstruct missing IDs if we have reception date
    if len(missing_id_rows) > 0 and datum_prijmu_col is not None:
        results.append("RECONSTRUCTING MISSING IDs")
        
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
        
        # Find time column for sorting
        cas_prijmu_col = None
        for col in df.columns:
            if 'cas' in str(col).lower() and ('prijm' in str(col).lower() or 'prij' in str(col).lower()):
                cas_prijmu_col = col
                break
        
        if cas_prijmu_col:
            results.append(f"Using '{cas_prijmu_col}' as reception time column for sorting")
        
        # Process each missing ID row
        for idx, row in missing_id_rows.iterrows():
            reception_date = row[datum_prijmu_col]
            
            if pd.isna(reception_date):
                results.append(f"Row {idx}: Cannot reconstruct ID - missing reception date")
                continue
            
            # Find all rows with the same reception date
            same_date_rows = df[df[datum_prijmu_col] == reception_date].copy()
            
            # METHOD 1: If multiple records with same date, use ranking by time
            if len(same_date_rows) >= 2 and cas_prijmu_col is not None:
                # Extract valid IDs from same date (to use as template)
                valid_ids = same_date_rows[(same_date_rows[id_col].notna()) & (same_date_rows[id_col] != '')][id_col]
                
                if not valid_ids.empty:
                    # Extract a template ID - check if we need to use old or new format logic
                    template_id = valid_ids.iloc[0]
                    if old_format and len(template_id) == 9:
                        # For old format, construct new format prefix
                        id_prefix = extract_date_parts(reception_date)
                        if not id_prefix:
                            results.append(f"Row {idx}: Failed to extract date parts from {reception_date}")
                            continue
                    else:
                        # For new format, use first 6 characters (YYMMDD)
                        id_prefix = template_id[:6]  # Extract YYMMDD part
                    
                    # Add time processing for ordering
                    same_date_rows['time_int'] = same_date_rows[cas_prijmu_col].apply(time_to_int)
                    
                    # Add original row position as tie-breaker for identical timestamps
                    same_date_rows['orig_position'] = range(len(same_date_rows))
                    
                    # Sort by time_int first, then by original position
                    same_date_rows = same_date_rows.sort_values(['time_int', 'orig_position'])
                    
                    # Add a rank column (1-based)
                    same_date_rows['rank'] = range(1, len(same_date_rows) + 1)
                    
                    # Find the rank of the current missing ID row
                    missing_row_in_sorted = same_date_rows[same_date_rows.index == idx]
                    
                    if not missing_row_in_sorted.empty:
                        rank = missing_row_in_sorted['rank'].iloc[0]
                        
                        # Create a new ID by combining the prefix with the rank (as 4 digits with leading zeros)
                        new_id = f"{id_prefix}{rank:04d}"
                        
                        results.append(f"Row {idx}: Reconstructed ID {new_id} from template/date with rank {rank}")
                        
                        # Update the ID in the dataframe
                        df.at[idx, id_col] = new_id
                        reconstructed_count += 1
                        continue
            
            # METHOD 2: If this is the only record for this date or METHOD 1 failed, 
            # directly extract date parts and create an ID
            id_prefix = extract_date_parts(reception_date)
            
            if id_prefix:
                # Since this is the only record or we couldn't use ranking, use 0001 as default
                new_id = f"{id_prefix}0001"
                results.append(f"Row {idx}: Created ID {new_id} directly from date {reception_date}")
                
                # Update the ID in the dataframe
                df.at[idx, id_col] = new_id
                solo_reconstructed_count += 1
            else:
                results.append(f"Row {idx}: Failed to extract date parts from {reception_date}")
        
        results.append(f"ID Reconstruction complete.")
        results.append(f"- IDs reconstructed using time ranking: {reconstructed_count}")
        results.append(f"- IDs reconstructed directly from date: {solo_reconstructed_count}")
        results.append(f"- Total reconstructed: {reconstructed_count + solo_reconstructed_count} of {len(missing_id_rows)}")
    
    # If we need to convert existing IDs from old to new format
    if old_format and datum_prijmu_col is not None:
        results.append("CONVERTING EXISTING IDs FROM OLD TO NEW FORMAT")
        
        # Get rows with valid IDs in old format
        valid_id_mask = df[id_col].notna() & (df[id_col] != '')
        old_format_ids = valid_id_mask & df[id_col].str.len().isin([8, 9])  # Allow for 8 or 9 digits
        
        if old_format_ids.sum() > 0:
            convert_count = 0
            
            for idx, row in df[old_format_ids].iterrows():
                old_id = row[id_col]
                reception_date = row[datum_prijmu_col]
                
                if pd.isna(reception_date):
                    results.append(f"Row {idx}: Cannot convert ID {old_id} - missing reception date")
                    continue
                
                # Extract new date prefix (YYMMDD)
                new_prefix = extract_date_parts(reception_date)
                
                if not new_prefix:
                    results.append(f"Row {idx}: Failed to extract date parts from {reception_date} for ID {old_id}")
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
                    results.append(f"Row {idx}: Converted {old_id} → {new_id}")
            
            results.append(f"Converted {convert_count} IDs from old format to new format.")
    
    # Find validovany vysledok column
    validovany_col = None
    for col in df.columns:
        if 'validovany' in str(col).lower() and 'vysledok' in str(col).lower():
            validovany_col = col
            break
    
    # Remove rows with blank validovany vysledok
    if validovany_col and remove_blank_validovany:
        initial_rows = len(df)
        df = df[df[validovany_col].notna() & (df[validovany_col] != '')]
        removed_rows = initial_rows - len(df)
        results.append(f"Removed {removed_rows} rows with blank values in '{validovany_col}' column")
    
    # Find diagnoza column
    diagnoza_col = None
    for col in df.columns:
        if 'diagnoza' in str(col).lower() or 'mkch' in str(col).lower():
            diagnoza_col = col
            break
    
    # Remove rows with blank diagnoza
    if diagnoza_col and remove_blank_diagnoza:
        initial_rows = len(df)
        df = df[df[diagnoza_col].notna() & (df[diagnoza_col] != '')]
        removed_rows = initial_rows - len(df)
        results.append(f"Removed {removed_rows} rows with blank values in '{diagnoza_col}' column")
    
    # Find HFE columns
    hfe_columns = []
    for col in df.columns:
        if 'hfe' in str(col).lower():
            hfe_columns.append(col)
    
    # Remove rows with blank HFE values
    if hfe_columns and remove_blank_hfe:
        initial_rows = len(df)
        for hfe_col in hfe_columns:
            df = df[df[hfe_col].notna() & (df[hfe_col] != '')]
        removed_rows = initial_rows - len(df)
        results.append(f"Removed {removed_rows} rows with blank values in HFE columns: {', '.join(hfe_columns)}")
    
    # Filter by age if requested
    if filter_by_age:
        # Find age column - try vek and other common age-related names
        age_col = None
        age_column_possibilities = ['vek', 'age', 'wiek', 'alter', 'edad', 'rok']
        
        for col_name in age_column_possibilities:
            possible_cols = [col for col in df.columns if col_name.lower() in str(col).lower()]
            if possible_cols:
                age_col = possible_cols[0]
                break
        
        if age_col:
            results.append(f"Found age column for filtering: '{age_col}'")
            initial_rows = len(df)
            
            try:
                # Convert to numeric, handling possible decimal separators
                age_values = df[age_col].astype(str).str.replace(',', '.').pipe(pd.to_numeric, errors='coerce')
                
                # Apply age filters
                df = df[(age_values >= min_age) & (age_values <= max_age)]
                removed_rows = initial_rows - len(df)
                
                results.append(f"Filtered age values between {min_age} and {max_age}")
                results.append(f"Removed {removed_rows} rows with age values outside the specified range")
            except Exception as e:
                results.append(f"Error filtering by age: {str(e)}")
        else:
            results.append("Could not find a suitable age column for filtering")
    
    results.append(f"Final rows after cleaning: {len(df)}")
    return df, results

def analyze_age_column(df):
    """
    Specifically analyze the 'vek' (age) column to extract min and max values
    """
    results = []
    results.append("\nAge Column Analysis:")
    
    # Find age column - try vek and other common age-related names
    age_col = None
    age_column_possibilities = ['vek', 'age', 'wiek', 'alter', 'edad', 'rok']
    
    for col_name in age_column_possibilities:
        possible_cols = [col for col in df.columns if col_name.lower() in str(col).lower()]
        if possible_cols:
            age_col = possible_cols[0]
            break
    
    if age_col is None:
        results.append("No column named 'vek' or other common age name found in the dataset")
        return results
    
    results.append(f"Found age column: '{age_col}'")
    
    # Show sample values for diagnostics
    sample_values = df[age_col].head(5).tolist()
    results.append(f"Sample values: {sample_values}")
    results.append(f"Column data type: {df[age_col].dtype}")
    
    # Ensure the column is numeric
    try:
        # If it's a date column, extract year and calculate age
        if pd.api.types.is_datetime64_any_dtype(df[age_col]):
            current_year = datetime.now().year
            df['calculated_age'] = current_year - df[age_col].dt.year
            age_values = df['calculated_age']
            results.append(f"Column is a date - calculated age from year")
        else:
            # Force to string first to handle various formats, then to numeric
            # This helps with both comma/period decimal separators
            age_values = df[age_col].astype(str).str.replace(',', '.').pipe(pd.to_numeric, errors='coerce')
            non_numeric_count = age_values.isna().sum() - df[age_col].isna().sum()
            if non_numeric_count > 0:
                results.append(f"Warning: {non_numeric_count} values could not be converted to numbers")
        
        # Basic validation check - report NaN count
        na_count = age_values.isna().sum()
        results.append(f"Missing or invalid values: {na_count} ({na_count/len(df)*100:.1f}%)")
                    
        valid_values = age_values.dropna()
        
        if len(valid_values) > 0:
            min_val = valid_values.min()
            max_val = valid_values.max()
            mean_val = valid_values.mean()
            median_val = valid_values.median()
            
            # Check if values are very large (birth years) or negative
            if median_val > 120:  # Likely birth years
                current_year = datetime.now().year
                valid_values = current_year - valid_values
                min_val = valid_values.min()
                max_val = valid_values.max()
                mean_val = valid_values.mean()
                median_val = valid_values.median()
                results.append(f"Column appears to contain birth years - converted to ages")
                
            results.append(f"Min age: {min_val:.1f}")
            results.append(f"Max age: {max_val:.1f}")
            results.append(f"Mean age: {mean_val:.1f}")
            results.append(f"Median age: {median_val:.1f}")
            
            # Age distribution
            age_counts = valid_values.value_counts().sort_index()
            if len(age_counts) <= 10:  # Only show full distribution if 10 or fewer unique values
                results.append("\nAge distribution:")
                for age, count in age_counts.items():
                    results.append(f"Age {age}: {count} patients ({count/len(valid_values)*100:.1f}%)")
            else:
                # Show age distribution by decade
                age_groups = pd.cut(valid_values, bins=[0, 20, 30, 40, 50, 60, 70, 80, 100], 
                                    labels=['0-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81+'])
                group_counts = age_groups.value_counts().sort_index()
                results.append("\nAge distribution by groups:")
                for group, count in group_counts.items():
                    results.append(f"{group}: {count} patients ({count/len(valid_values)*100:.1f}%)")
        else:
            results.append("No valid numeric values found in the age column")
            
    except Exception as e:
        results.append(f"Error analyzing age column: {str(e)}")
        import traceback
        results.append(traceback.format_exc())
    
    return results

def find_hfe_columns(df):
    """Helper function to find HFE-related columns in the dataset"""
    hfe_columns = []
    for col in df.columns:
        if 'hfe' in str(col).lower():
            hfe_columns.append(col)
    return hfe_columns 