#!/usr/bin/env python3
# Data Cleaning Script for SSBU25_dataset.xls

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load the dataset
file_path = Path("Info/SSBU25_dataset.xls")
print(f"Loading dataset from {file_path.absolute()}")

try:
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Display basic information about the dataset
    print("\nDataset Info:")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    for col in df.columns:
        print(f"- {col}")
    
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    # Basic data cleaning
    # 1. Handle missing values
    print("\nCleaning Data...")
    
    # For numeric columns, fill NaN with median
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
            print(f"Filled missing values in {col} with median: {median_value}")
    
    # For categorical/text columns, fill with most frequent value
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0]
            df[col].fillna(mode_value, inplace=True)
            print(f"Filled missing values in {col} with mode: {mode_value}")
    
    # 2. Remove duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        print(f"Removed {duplicates} duplicate rows")
    
    # 3. Save the cleaned dataset
    output_path = "cleaned_dataset.xlsx"
    df.to_excel(output_path, index=False)
    print(f"\nCleaned dataset saved to {output_path}")
    
    print("\nCleaning Complete!")
    
except Exception as e:
    print(f"Error processing the dataset: {e}") 