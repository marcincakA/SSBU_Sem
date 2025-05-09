# Dataset Cleaning Script

This script analyzes and cleans the SSBU25_dataset.xls file.

## Features

- Loads the Excel dataset
- Displays basic information and statistics
- Identifies and handles missing values
- Removes duplicate entries
- Saves the cleaned dataset as a new Excel file

## Requirements

Install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

Run the script with:

```
python data_cleaning.py
```

The cleaned dataset will be saved as `cleaned_dataset.xlsx` in the current directory.

## Cleaning Operations

1. **Missing Values**:
   - Numeric columns: filled with median values
   - Categorical columns: filled with most frequent values

2. **Duplicate Removal**:
   - Removes duplicate rows from the dataset

3. **Output**:
   - Creates a new Excel file with the cleaned data 