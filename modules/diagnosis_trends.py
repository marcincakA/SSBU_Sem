#!/usr/bin/env python3
# Diagnosis trends analysis module for HFE mutation analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
import re
import requests
from bs4 import BeautifulSoup

# Define ICD-10 categories
ICD10_CATEGORIES = {
    'A': 'Certain infectious and parasitic diseases',
    'B': 'Certain infectious and parasitic diseases',
    'C': 'Neoplasms',
    'D00-D48': 'Neoplasms',
    'D50-D89': 'Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism',
    'E': 'Endocrine, nutritional and metabolic diseases',
    'F': 'Mental and behavioural disorders',
    'G': 'Diseases of the nervous system',
    'H00-H59': 'Diseases of the eye and adnexa',
    'H60-H95': 'Diseases of the ear and mastoid process',
    'I': 'Diseases of the circulatory system',
    'J': 'Diseases of the respiratory system',
    'K': 'Diseases of the digestive system',
    'L': 'Diseases of the skin and subcutaneous tissue',
    'M': 'Diseases of the musculoskeletal system and connective tissue',
    'N': 'Diseases of the genitourinary system',
    'O': 'Pregnancy, childbirth and the puerperium',
    'P': 'Certain conditions originating in the perinatal period',
    'Q': 'Congenital malformations, deformations and chromosomal abnormalities',
    'R': 'Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified',
    'S': 'Injury, poisoning and certain other consequences of external causes',
    'T': 'Injury, poisoning and certain other consequences of external causes',
    'V': 'External causes of morbidity and mortality',
    'W': 'External causes of morbidity and mortality',
    'X': 'External causes of morbidity and mortality',
    'Y': 'External causes of morbidity and mortality',
    'Z': 'Factors influencing health status and contact with health services'
}

# Define specialized categories related to liver diseases and HFE relevance
LIVER_CATEGORIES = {
    'K70': 'Alcoholic liver disease',
    'K71': 'Toxic liver disease',
    'K72': 'Hepatic failure',
    'K73': 'Chronic hepatitis',
    'K74': 'Fibrosis and cirrhosis of liver',
    'K75': 'Other inflammatory liver diseases',
    'K76': 'Other diseases of liver',
    'K77': 'Liver disorders in diseases classified elsewhere',
    'E83.1': 'Disorders of iron metabolism (incl. hemochromatosis)'
}

def categorize_icd10(code):
    """
    Categorize an ICD-10 code into its main category.
    
    Args:
        code: ICD-10 diagnosis code
    
    Returns:
        Category name
    """
    if pd.isna(code) or code == '':
        return "Unknown"
    
    # Normalize the code (remove spaces, dots in some positions)
    code = str(code).strip().upper()
    letter_part = code[0]
    
    # First check specialized liver categories
    for prefix, category in LIVER_CATEGORIES.items():
        if code.startswith(prefix):
            return category
    
    # Then check general categories
    for prefix, category in ICD10_CATEGORIES.items():
        if '-' in prefix:  # Handle ranges like D00-D48
            range_start, range_end = prefix.split('-')
            base_letter = range_start[0]
            if letter_part == base_letter:
                # Extract numeric parts
                if len(code) > 1 and code[1:].strip() and code[1:].strip()[0].isdigit():
                    num_part = int(re.findall(r'\d+', code)[0])
                    range_start_num = int(re.findall(r'\d+', range_start)[0])
                    range_end_num = int(re.findall(r'\d+', range_end)[0])
                    if range_start_num <= num_part <= range_end_num:
                        return category
        elif letter_part == prefix:
            return category
    
    # If no match found
    return "Other/Unspecified"

def extract_exam_year(date_val):
    """
    Extract the year from a date value.
    
    Args:
        date_val: Date value (str, datetime, etc.)
    
    Returns:
        Year as integer or None if extraction fails
    """
    if pd.isna(date_val):
        return None
    
    try:
        # If already a datetime
        if isinstance(date_val, (pd.Timestamp, datetime)):
            return date_val.year
        
        # Try parsing string with different formats
        for fmt in ['%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d']:
            try:
                dt = pd.to_datetime(date_val, format=fmt)
                return dt.year
            except:
                continue
        
        # If none of the specific formats worked, let pandas guess
        dt = pd.to_datetime(date_val)
        return dt.year
    except:
        # Look for year patterns in the string
        if isinstance(date_val, str):
            year_match = re.search(r'(19|20)\d{2}', date_val)
            if year_match:
                return int(year_match.group(0))
    
    return None

def analyze_diagnosis_trends(df, specified_date_col=None):
    """
    Analyze diagnoses by ICD-10 code and their changes over time.
    
    Args:
        df: DataFrame containing the data
        specified_date_col: Optional user-specified date column name
    
    Returns:
        Tuple of (results text, visualization data)
    """
    results = []
    plots = []
    
    # Find diagnosis and date columns
    diagnoza_col = next((col for col in df.columns 
                       if any(term in str(col).lower() for term in 
                             ['diagnoza', 'mkch', 'diagnosis', 'icd', 'kod'])), None)
    
    # Use the specified date column if provided, otherwise auto-detect
    if specified_date_col and specified_date_col in df.columns:
        date_col = specified_date_col
        results.append(f"Using user-selected date column: {date_col}")
    else:
        date_col = next((col for col in df.columns 
                        if any(term in str(col).lower() for term in 
                              ['datum', 'date', 'rok', 'year', 'cas_prijmu', 'validation_date'])), None)
        if specified_date_col:
            results.append(f"Warning: Specified date column '{specified_date_col}' not found in dataset.")
            results.append(f"Auto-detected date column: {date_col}")
        else:
            results.append(f"Auto-detected date column: {date_col}")
    
    if not diagnoza_col or not date_col:
        results.append("Could not find necessary columns for diagnosis trend analysis")
        if not diagnoza_col:
            results.append("- No diagnosis column found")
        if not date_col:
            results.append("- No date column found")
        return results, plots
    
    results.append(f"Diagnosis column: {diagnoza_col}")
    
    # Create copies to avoid modifying the original
    df_analysis = df[[diagnoza_col, date_col]].copy()
    
    # Extract years
    df_analysis['exam_year'] = df_analysis[date_col].apply(extract_exam_year)
    
    # Categorize diagnoses
    df_analysis['diagnosis_category'] = df_analysis[diagnoza_col].apply(categorize_icd10)
    
    # Filter out missing years or diagnoses
    df_analysis = df_analysis.dropna(subset=['exam_year', 'diagnosis_category'])
    
    # Convert year to integer
    df_analysis['exam_year'] = df_analysis['exam_year'].astype(int)
    
    # Get range of years
    years = sorted(df_analysis['exam_year'].unique())
    
    if len(years) <= 1:
        results.append(f"Insufficient time range for trend analysis: only found year {years[0] if years else 'N/A'}")
        return results, plots
    
    results.append(f"Analysis period: {min(years)} - {max(years)}")
    
    # Analyze overall distribution of diagnosis categories
    results.append("\nOverall Distribution of Diagnosis Categories:")
    category_counts = df_analysis['diagnosis_category'].value_counts()
    for category, count in category_counts.items():
        results.append(f"{category}: {count} patients ({count/len(df_analysis)*100:.1f}%)")
    
    # Create time trend plot for major categories
    plt.figure(figsize=(12, 8))
    # Count diagnoses by category and year
    diagnosis_by_year = pd.crosstab(df_analysis['exam_year'], df_analysis['diagnosis_category'])
    
    # Normalize by year to get percentages
    diagnosis_by_year_pct = diagnosis_by_year.div(diagnosis_by_year.sum(axis=1), axis=0) * 100
    
    # Plot trends for top categories
    top_categories = category_counts.nlargest(5).index.tolist()
    diagnosis_by_year_pct[top_categories].plot(marker='o')
    
    plt.title('Diagnosis Category Trends Over Time')
    plt.xlabel('Year')
    plt.ylabel('Percentage of Diagnoses (%)')
    plt.legend(title='Diagnosis Category')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Convert to base64 for display
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plots.append(("Diagnosis Trends", img_str))
    plt.close()
    
    # Create heatmap for diagnosis categories by year
    plt.figure(figsize=(14, 10))
    
    # Only include years and categories with sufficient data
    min_count = 3  # Minimum count to include
    filtered_diagnosis_by_year = diagnosis_by_year.copy()
    
    # Drop categories with insufficient data
    category_totals = filtered_diagnosis_by_year.sum()
    categories_to_keep = category_totals[category_totals >= min_count].index
    filtered_diagnosis_by_year = filtered_diagnosis_by_year[categories_to_keep]
    
    # Create heatmap
    sns.heatmap(filtered_diagnosis_by_year, cmap="YlGnBu", annot=True, fmt="g", linewidths=.5)
    plt.title('Number of Diagnoses by Category and Year')
    plt.ylabel('Year')
    plt.xlabel('Diagnosis Category')
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right')
    
    # Convert to base64 for display
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plots.append(("Diagnosis Heatmap", img_str))
    plt.close()
    
    # Analyze liver-specific diagnoses
    liver_diagnoses = df_analysis[df_analysis[diagnoza_col].apply(
        lambda x: any(str(x).startswith(code) for code in LIVER_CATEGORIES.keys() if isinstance(x, (str, int)))
    )]
    
    if len(liver_diagnoses) > 0:
        results.append("\nLiver-Related Diagnoses Analysis:")
        liver_counts = liver_diagnoses['diagnosis_category'].value_counts()
        for category, count in liver_counts.items():
            results.append(f"{category}: {count} patients ({count/len(df_analysis)*100:.1f}% of all diagnoses)")
        
        # Time trend for liver diagnoses
        if len(years) > 1 and len(liver_diagnoses) >= 5:
            plt.figure(figsize=(12, 8))
            liver_by_year = pd.crosstab(liver_diagnoses['exam_year'], liver_diagnoses['diagnosis_category'])
            
            # Fill missing years with 0
            for year in years:
                if year not in liver_by_year.index:
                    liver_by_year.loc[year] = 0
            liver_by_year = liver_by_year.sort_index()
            
            # Plot
            liver_by_year.plot(marker='o')
            plt.title('Liver-Related Diagnoses Over Time')
            plt.xlabel('Year')
            plt.ylabel('Number of Patients')
            plt.legend(title='Liver Diagnosis')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Convert to base64 for display
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plots.append(("Liver Diagnosis Trends", img_str))
            plt.close()
    else:
        results.append("\nNo liver-related diagnoses found in the dataset.")
    
    # Check for obsolete codes
    results.append("\nValidation of ICD-10 Codes:")
    
    # Get unique diagnosis codes
    unique_codes = df_analysis[diagnoza_col].dropna().unique()
    
    # TODO: Implement full validation against NCZI or WHO ICD-10 database
    # For now, we'll just check basic format validity
    
    valid_codes = 0
    invalid_format_codes = []
    potentially_obsolete_codes = []
    
    for code in unique_codes:
        code_str = str(code).strip().upper()
        
        # Check basic format (letter followed by numbers, possibly with a dot)
        if re.match(r'^[A-Z]\d+(\.\d+)?$', code_str):
            valid_codes += 1
            
            # Check for potentially obsolete codes (this is a placeholder; real check would query NCZI database)
            if code_str.startswith('R99') or code_str.startswith('Z98'):
                potentially_obsolete_codes.append(code_str)
        else:
            invalid_format_codes.append(code_str)
    
    results.append(f"Total unique diagnosis codes: {len(unique_codes)}")
    results.append(f"Codes with valid format: {valid_codes}")
    
    if invalid_format_codes:
        results.append("\nCodes with invalid format:")
        for code in invalid_format_codes[:10]:  # Show first 10
            results.append(f"- {code}")
        if len(invalid_format_codes) > 10:
            results.append(f"... and {len(invalid_format_codes)-10} more")
    
    if potentially_obsolete_codes:
        results.append("\nPotentially obsolete or special codes (require manual verification):")
        for code in potentially_obsolete_codes:
            results.append(f"- {code}")
        results.append("\nPlease verify these codes against the current NCZI standards at:")
        results.append("https://www.nczisk.sk/Standardy-v-zdravotnictve/Pages/MKCH-10-Revizia.aspx")
    
    return results, plots

def check_icd10_code_validity(code, check_online=False):
    """
    Check if an ICD-10 code is currently valid according to standards.
    
    Args:
        code: ICD-10 code to check
        check_online: Whether to attempt an online check against official sources
    
    Returns:
        Dictionary with validation results
    """
    result = {
        "code": code,
        "valid_format": False,
        "description": None,
        "status": "unknown",
        "message": ""
    }
    
    # Normalize the code
    code_str = str(code).strip().upper()
    
    # Check basic format
    if not re.match(r'^[A-Z]\d+(\.\d+)?$', code_str):
        result["message"] = "Invalid ICD-10 format. Should be a letter followed by numbers, with optional decimal."
        return result
    
    result["valid_format"] = True
    
    # For offline validation, we can check against our category mappings
    base_code = code_str[0]
    
    for prefix, category in ICD10_CATEGORIES.items():
        if base_code == prefix or (len(prefix) > 1 and code_str.startswith(prefix)):
            result["description"] = category
            result["status"] = "valid"
            break
    
    # Check liver-specific codes more precisely
    for prefix, category in LIVER_CATEGORIES.items():
        if code_str.startswith(prefix):
            result["description"] = category
            result["status"] = "valid"
            break
    
    # Online check is optional and might be slow/unreliable
    if check_online:
        try:
            # This is a placeholder - would need to be replaced with actual API call to NCZI
            # For demonstration purposes only
            result["message"] += " (Online validation attempted but not implemented)"
        except Exception as e:
            result["message"] += f" Online check error: {str(e)}"
    
    return result

def generate_diagnosis_validation_report(df, diagnoza_col):
    """
    Generate a detailed report on the validity of diagnosis codes in the dataset.
    
    Args:
        df: DataFrame containing the data
        diagnoza_col: Column name containing diagnosis codes
    
    Returns:
        Text report and validation statistics
    """
    results = []
    stats = {}
    
    results.append("ICD-10 CODE VALIDATION REPORT")
    results.append("-" * 30)
    
    # Get unique codes
    unique_codes = df[diagnoza_col].dropna().unique()
    total_codes = len(unique_codes)
    
    results.append(f"Total unique diagnosis codes: {total_codes}")
    
    # Categorize codes
    valid_format_count = 0
    valid_code_count = 0
    invalid_codes = []
    valid_codes = []
    obsolete_codes = []
    special_codes = []
    
    for code in unique_codes:
        validation = check_icd10_code_validity(code)
        
        if validation["valid_format"]:
            valid_format_count += 1
            
            if validation["status"] == "valid":
                valid_code_count += 1
                valid_codes.append((code, validation["description"]))
            elif validation["status"] == "obsolete":
                obsolete_codes.append(code)
            elif validation["status"] == "special":
                special_codes.append(code)
        else:
            invalid_codes.append((code, validation["message"]))
    
    # Compile statistics
    stats["total_codes"] = total_codes
    stats["valid_format_percentage"] = (valid_format_count / total_codes * 100) if total_codes > 0 else 0
    stats["valid_code_percentage"] = (valid_code_count / total_codes * 100) if total_codes > 0 else 0
    stats["invalid_codes"] = len(invalid_codes)
    stats["obsolete_codes"] = len(obsolete_codes)
    stats["special_codes"] = len(special_codes)
    
    # Add to results
    results.append(f"\nFormat Validation:")
    results.append(f"- Codes with valid format: {valid_format_count} ({stats['valid_format_percentage']:.1f}%)")
    results.append(f"- Codes with invalid format: {len(invalid_codes)} ({100-stats['valid_format_percentage']:.1f}%)")
    
    results.append(f"\nContent Validation:")
    results.append(f"- Valid codes: {valid_code_count} ({stats['valid_code_percentage']:.1f}%)")
    results.append(f"- Potentially obsolete codes: {len(obsolete_codes)}")
    results.append(f"- Special/administrative codes: {len(special_codes)}")
    
    # Show examples of invalid codes
    if invalid_codes:
        results.append("\nSample of Invalid Format Codes:")
        for code, message in invalid_codes[:5]:  # Show first 5
            results.append(f"- {code}: {message}")
        if len(invalid_codes) > 5:
            results.append(f"... and {len(invalid_codes)-5} more")
    
    # Show examples of obsolete codes
    if obsolete_codes:
        results.append("\nPotentially Obsolete Codes (require verification):")
        for code in obsolete_codes[:5]:  # Show first 5
            results.append(f"- {code}")
        if len(obsolete_codes) > 5:
            results.append(f"... and {len(obsolete_codes)-5} more")
    
    # Distribution by major categories
    results.append("\nDistribution by Major Categories:")
    category_counts = {}
    
    for code, description in valid_codes:
        if description in category_counts:
            category_counts[description] += 1
        else:
            category_counts[description] = 1
    
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        results.append(f"- {category}: {count} codes ({count/len(valid_codes)*100:.1f}%)")
    
    # Verification resources
    results.append("\nVerification Resources:")
    results.append("- NCZI ICD-10 Standards: https://www.nczisk.sk/Standardy-v-zdravotnictve/Pages/MKCH-10-Revizia.aspx")
    results.append("- WHO ICD-10 Browser: https://icd.who.int/browse10/2019/en")
    
    return results, stats 