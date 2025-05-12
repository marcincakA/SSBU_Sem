#!/usr/bin/env python3
# Dataset Analysis Script for SSBU25_dataset.xls

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Analyze SSBU25 dataset")
parser.add_argument("--path", "-p", type=str, help="Path to the dataset file")
args = parser.parse_args()

# Load the dataset
default_path = "SSBU25_dataset_modified_new.xlsx"
file_path = Path(args.path) if args.path else Path(default_path)
print(f"Loading dataset from {file_path.absolute()}")

try:
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Basic dataset information
    print("\n" + "="*50)
    print("DATASET OVERVIEW")
    print("="*50)
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    
    # Column names and indices
    print("\n" + "="*50)
    print("COLUMN NAMES AND INDICES")
    print("="*50)
    for i, col in enumerate(df.columns):
        print(f"Column {i}: '{col}'")
    
    # Data types
    print("\n" + "="*50)
    print("DATA TYPES")
    print("="*50)
    print(df.dtypes)
    
    # Missing values analysis
    print("\n" + "="*50)
    print("MISSING VALUES ANALYSIS")
    print("="*50)
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    
    missing_data = pd.DataFrame({
        'Missing Values': missing,
        'Percentage (%)': missing_percent
    })
    
    print(missing_data[missing_data['Missing Values'] > 0].sort_values('Missing Values', ascending=False))
    
    # Basic statistics for numeric columns
    print("\n" + "="*50)
    print("NUMERIC COLUMNS STATISTICS")
    print("="*50)
    print(df.describe())
    
    # Value counts for categorical columns (top 5 values)
    print("\n" + "="*50)
    print("CATEGORICAL COLUMNS VALUE COUNTS (TOP 5)")
    print("="*50)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\nColumn: '{col}'")
        print("-" * 30)
        value_counts = df[col].value_counts().head(5)
        print(value_counts)
        print(f"Unique values: {df[col].nunique()}")
    
    # Check for duplicate rows
    print("\n" + "="*50)
    print("DUPLICATE ROWS")
    print("="*50)
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")
    
    print("\nAnalysis complete!")
    
except Exception as e:
    print(f"Error analyzing the dataset: {e}") 