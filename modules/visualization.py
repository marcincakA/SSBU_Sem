#!/usr/bin/env python3
# Visualization module for HFE mutation analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime

def generate_genotype_distribution_plot(df, hfe_columns):
    """
    Generate plots for genotype distribution
    """
    plots = []
    
    # Create a plot for each HFE column
    for column in hfe_columns:
        plt.figure(figsize=(10, 6))
        genotype_counts = df[column].value_counts()
        total = len(df)
        
        # Skip if no data
        if total == 0:
            plt.close()
            continue
        
        # Create bar chart
        ax = sns.barplot(x=genotype_counts.index, y=genotype_counts.values)
        
        # Add count and percentage labels
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            percentage = height / total * 100
            ax.annotate(f'{int(height)} ({percentage:.1f}%)', 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', xytext=(0, 10),
                        textcoords='offset points')
        
        # Add labels and title
        plt.xlabel('Genotype')
        plt.ylabel('Number of Patients')
        column_name = column.split('\n')[0] if '\n' in column else column
        plt.title(f'Genotype Distribution for {column_name}')
        plt.tight_layout()
        
        # Convert to base64 for display in Shiny
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plots.append((column_name, img_str))
        plt.close()
    
    return plots

def generate_risk_distribution_plot(df):
    """
    Generate plot for HH risk distribution
    """
    if 'Risk_Category' not in df.columns:
        return None
    
    plt.figure(figsize=(10, 6))
    risk_counts = df['Risk_Category'].value_counts()
    total = len(df)
    
    # Skip if no data
    if total == 0:
        plt.close()
        return None
    
    # Create bar chart
    ax = sns.barplot(x=risk_counts.index, y=risk_counts.values)
    
    # Add count and percentage labels
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        percentage = height / total * 100
        ax.annotate(f'{int(height)} ({percentage:.1f}%)', 
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', xytext=(0, 10),
                    textcoords='offset points')
    
    # Add labels and title
    plt.xlabel('Risk Category')
    plt.ylabel('Number of Patients')
    plt.title('Hereditary Hemochromatosis Risk Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Convert to base64 for display in Shiny
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_str

def generate_diagnosis_association_plot(df):
    """
    Generate plot for diagnosis association with HFE mutations
    """
    if 'Diagnosis_Category' not in df.columns or 'Any_HFE_Mutation' not in df.columns:
        return None
    
    plt.figure(figsize=(12, 8))
    diagnosis_mutation = pd.crosstab(df['Diagnosis_Category'], df['Any_HFE_Mutation'], normalize='index') * 100
    diagnosis_mutation.plot(kind='bar', stacked=True)
    plt.title('Diagnosis Categories by HFE Mutation Status')
    plt.xlabel('Diagnosis Category')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Mutation Status')
    plt.tight_layout()
    
    # Convert to base64 for display in Shiny
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_str

def generate_liver_disease_plot(df):
    """
    Generate plot for liver disease prevalence by risk category
    """
    if 'Risk_Category' not in df.columns or 'Has_K760' not in df.columns or 'Has_K759' not in df.columns:
        return None
    
    k760_by_risk = df.groupby('Risk_Category')['Has_K760'].mean() * 100
    k759_by_risk = df.groupby('Risk_Category')['Has_K759'].mean() * 100
    
    liver_prevalence = pd.DataFrame({
        'K76.0 (Fatty liver)': k760_by_risk,
        'K75.9 (Inflammatory liver disease)': k759_by_risk
    })
    
    plt.figure(figsize=(12, 8))
    liver_prevalence.plot(kind='bar')
    plt.title('Liver Disease Prevalence by HH Risk Category')
    plt.xlabel('Risk Category')
    plt.ylabel('Prevalence (%)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Liver Disease')
    plt.tight_layout()
    
    # Convert to base64 for display in Shiny
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_str

def generate_hardy_weinberg_plots(hwe_results_list):
    """
    Create bar charts comparing observed vs expected genotype distributions for HWE analysis
    """
    plots = []
    
    for result in hwe_results_list:
        mutation = result["Mutation"]
        
        # Data for plotting
        genotypes = ["Normal", "Heterozygote", "Mutant"]
        observed = [result["Normal (observed)"], result["Heterozygote (observed)"], result["Mutant (observed)"]]
        expected = [result["Normal (expected)"], result["Heterozygote (expected)"], result["Mutant (expected)"]]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Set up bar width and positions
        bar_width = 0.35
        indices = np.arange(len(genotypes))
        
        # Create bars
        plt.bar(indices - bar_width/2, observed, bar_width, label='Observed', color='blue', alpha=0.7)
        plt.bar(indices + bar_width/2, expected, bar_width, label='Expected (HWE)', color='green', alpha=0.7)
        
        # Add labels, title, and legend
        plt.xlabel('Genotype')
        plt.ylabel('Count')
        plt.title(f'Hardy-Weinberg Equilibrium Analysis: {mutation}')
        plt.xticks(indices, genotypes)
        plt.legend()
        
        # Add HWE status and p-value to the plot
        hwe_status = result["In Hardy-Weinberg equilibrium"]
        p_value = result["p-value"]
        
        if isinstance(p_value, float):
            plt.figtext(0.5, 0.01, f"HWE Status: {hwe_status} (p = {p_value:.4f})", 
                        ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
        else:
            plt.figtext(0.5, 0.01, f"HWE Status: {hwe_status}", 
                        ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
        
        # Convert to base64 for display in Shiny
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plots.append((mutation, img_str))
        plt.close()
    
    return plots

def generate_genotype_by_age_plot(df, hfe_columns):
    """Generate plots showing the relationship between genotypes and age"""
    plots = []
    results = []
    
    # Find age column - expanded search terms
    age_col = next((col for col in df.columns if any(term in str(col).lower() for term in 
                   ['vek', 'age', 'rok', 'year', 'date', 'datum', 'nar', 'birth'])), None)
    
    results.append(f"Age column search: {'Found: ' + age_col if age_col else 'Not found'}")
    
    if age_col is None:
        # Try to infer from date columns
        date_cols = [col for col in df.columns if any(term in str(col).lower() for term in 
                    ['date', 'datum', 'cas', 'time'])]
        results.append(f"Checked date columns: {date_cols}")
        return [], results
    
    # Print sample values from age column for diagnostics
    sample_values = df[age_col].dropna().head(5).tolist()
    results.append(f"Sample values in age column: {sample_values}")
    
    # Ensure age column is numeric
    try:
        # If it's a date column, extract year
        if pd.api.types.is_datetime64_any_dtype(df[age_col]):
            results.append(f"Detected datetime in age column")
            current_year = datetime.now().year
            df['calculated_age'] = current_year - df[age_col].dt.year
            age_col = 'calculated_age'
            results.append(f"Created calculated age based on year")
        else:
            # Try to convert to numeric
            df[age_col] = df[age_col].astype(str).str.replace(',', '.').pipe(pd.to_numeric, errors='coerce')
        
        # Check if conversion worked
        if df[age_col].isna().all():
            results.append(f"Age column contains no valid numeric values after conversion")
            return [], results
            
        # Check the range of ages to see if they're reasonable
        min_age = df[age_col].min()
        max_age = df[age_col].max()
        results.append(f"Age range: {min_age} to {max_age}")
        
        # If "ages" are very large (e.g., years like 1990), they might be birth years
        if min_age > 120:  # Assuming no one is older than 120
            current_year = datetime.now().year
            df['calculated_age'] = current_year - df[age_col]
            age_col = 'calculated_age'
            results.append(f"Converted birth years to ages. New range: {df[age_col].min()} to {df[age_col].max()}")
    except Exception as e:
        results.append(f"Error converting age column: {str(e)}")
        return [], results
    
    # Create age groups for better visualization
    try:
        df['Age_Group'] = pd.cut(df[age_col], bins=[0, 30, 45, 60, 75, 100], 
                                labels=['<30', '30-45', '45-60', '60-75', '>75'])
        results.append(f"Created age groups successfully")
    except Exception as e:
        results.append(f"Error creating standard age groups: {str(e)}")
        try:
            # Alternative approach with quantiles
            df['Age_Group'] = pd.qcut(df[age_col].dropna(), q=4, duplicates='drop')
            results.append(f"Created quartile-based age groups")
        except Exception as e2:
            results.append(f"Error creating alternative age groups: {str(e2)}")
            return [], results
    
    # Count how many individuals in each age group
    age_group_counts = df['Age_Group'].value_counts()
    results.append(f"Age group distribution: {dict(age_group_counts)}")
    
    for column in hfe_columns:
        try:
            plt.figure(figsize=(12, 8))
            
            # Filter out NaN values in both columns
            valid_data = df.dropna(subset=[column, 'Age_Group'])
            results.append(f"Valid data points for {column}: {len(valid_data)}")
            
            # Skip if not enough data (reduced threshold)
            if len(valid_data) < 3:
                results.append(f"Not enough data for {column}")
                plt.close()
                continue
                
            mutation_by_age = pd.crosstab(valid_data['Age_Group'], valid_data[column], normalize='index') * 100
            
            # Skip if crosstab is empty
            if mutation_by_age.empty:
                results.append(f"Empty crosstab for {column}")
                plt.close()
                continue
                
            mutation_by_age.plot(kind='bar', stacked=True)
            
            plt.title(f'Age Distribution by {column} Genotype')
            plt.xlabel('Age Group')
            plt.ylabel('Percentage (%)')
            plt.legend(title='Genotype')
            plt.tight_layout()
            
            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plots.append((f"{column} by Age", img_str))
            plt.close()
            results.append(f"Successfully created plot for {column}")
        except Exception as e:
            results.append(f"Error creating plot for {column}: {str(e)}")
            plt.close()
            continue
        
    return plots, results

def generate_genotype_by_gender_plot(df, hfe_columns):
    """Generate plots showing the relationship between genotypes and gender"""
    plots = []
    results = []
    
    # Find gender column - expanded search terms
    gender_col = next((col for col in df.columns 
                      if any(term in str(col).lower() for term in 
                            ['pohlavie', 'gender', 'sex', 'pohlavi', 'sexus', 'm/f', 'm/ž', 'm/z'])), None)
    
    results.append(f"Gender column search: {'Found: ' + gender_col if gender_col else 'Not found'}")
    
    if gender_col is None:
        # Check for columns with short values that might be gender
        short_value_cols = []
        for col in df.columns:
            # Check if column contains mostly short strings (potential gender values)
            if df[col].dtype == object:
                val_lengths = df[col].astype(str).str.len()
                if val_lengths.mean() < 3 and val_lengths.max() < 5:
                    short_value_cols.append(col)
        
        results.append(f"Potential gender columns based on value length: {short_value_cols}")
        
        # If no gender column found, try the first short value column
        if short_value_cols and not gender_col:
            gender_col = short_value_cols[0]
            results.append(f"Using column {gender_col} as potential gender column")
    
    if gender_col is None:
        return [], results
    
    # Print sample values from gender column for diagnostics
    sample_values = df[gender_col].dropna().head(5).tolist()
    results.append(f"Sample values in gender column: {sample_values}")
    
    # Standardize gender values
    try:
        # Create a copy to avoid modifying the original DataFrame
        gender_mapping = {}
        unique_values = df[gender_col].dropna().unique()
        
        for val in unique_values:
            str_val = str(val).lower().strip()
            if str_val in ['m', 'male', 'muz', 'muž', '1']:
                gender_mapping[val] = 'Male'
            elif str_val in ['f', 'female', 'zena', 'žena', 'z', '2']:
                gender_mapping[val] = 'Female'
                
        # If we found mappings, create a standardized gender column
        if gender_mapping:
            df['Standardized_Gender'] = df[gender_col].map(gender_mapping)
            gender_col = 'Standardized_Gender'
            results.append(f"Standardized gender values: {gender_mapping}")
        
        # Check the number of unique gender values
        gender_counts = df[gender_col].value_counts()
        results.append(f"Gender distribution: {dict(gender_counts)}")
    except Exception as e:
        results.append(f"Error standardizing gender: {str(e)}")
    
    for column in hfe_columns:
        try:
            plt.figure(figsize=(10, 6))
            
            # Filter out NaN values in both columns
            valid_data = df.dropna(subset=[column, gender_col])
            results.append(f"Valid data points for {column}: {len(valid_data)}")
            
            # Skip if not enough data (reduced threshold)
            if len(valid_data) < 3:
                results.append(f"Not enough data for {column}")
                plt.close()
                continue
                
            gender_by_mutation = pd.crosstab(valid_data[gender_col], valid_data[column], normalize='index') * 100
            
            # Skip if crosstab is empty
            if gender_by_mutation.empty:
                results.append(f"Empty crosstab for {column}")
                plt.close()
                continue
                
            gender_by_mutation.plot(kind='bar', stacked=True)
            
            plt.title(f'Gender Distribution by {column} Genotype')
            plt.xlabel('Gender')
            plt.ylabel('Percentage (%)')
            plt.legend(title='Genotype')
            plt.tight_layout()
            
            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plots.append((f"{column} by Gender", img_str))
            plt.close()
            results.append(f"Successfully created plot for {column}")
        except Exception as e:
            results.append(f"Error creating plot for {column}: {str(e)}")
            plt.close()
            continue
        
    return plots, results

def generate_genotype_diagnosis_gender_plot(df, hfe_columns):
    """Generate plots showing relationships between genotypes, diagnoses, and gender"""
    plots = []
    results = []
    
    # Find gender and diagnosis columns - expanded search terms
    gender_col = next((col for col in df.columns 
                      if any(term in str(col).lower() for term in 
                            ['pohlavie', 'gender', 'sex', 'pohlavi', 'sexus', 'm/f', 'm/ž', 'm/z'])), None)
    
    diagnoza_col = next((col for col in df.columns 
                        if any(term in str(col).lower() for term in 
                              ['diagnoza', 'mkch', 'diagnosis', 'icd', 'kod'])), None)
    
    results.append(f"Gender column search: {'Found: ' + gender_col if gender_col else 'Not found'}")
    results.append(f"Diagnosis column search: {'Found: ' + diagnoza_col if diagnoza_col else 'Not found'}")
    
    if gender_col is None:
        # Check for columns with short values that might be gender
        short_value_cols = []
        for col in df.columns:
            # Check if column contains mostly short strings (potential gender values)
            if df[col].dtype == object:
                val_lengths = df[col].astype(str).str.len()
                if val_lengths.mean() < 3 and val_lengths.max() < 5:
                    short_value_cols.append(col)
        
        results.append(f"Potential gender columns based on value length: {short_value_cols}")
        
        # If no gender column found, try the first short value column
        if short_value_cols and not gender_col:
            gender_col = short_value_cols[0]
            results.append(f"Using column {gender_col} as potential gender column")
    
    if gender_col is None or diagnoza_col is None:
        if gender_col is None:
            results.append("Cannot create gender-diagnosis plots: gender column not found")
        if diagnoza_col is None:
            results.append("Cannot create gender-diagnosis plots: diagnosis column not found")
        return [], results
    
    # Print sample values for diagnostics
    gender_samples = df[gender_col].dropna().head(5).tolist()
    diag_samples = df[diagnoza_col].dropna().head(5).tolist()
    results.append(f"Sample gender values: {gender_samples}")
    results.append(f"Sample diagnosis values: {diag_samples}")
    
    # Standardize gender values
    try:
        # Create a copy to avoid modifying the original DataFrame
        gender_mapping = {}
        unique_values = df[gender_col].dropna().unique()
        
        for val in unique_values:
            str_val = str(val).lower().strip()
            if str_val in ['m', 'male', 'muz', 'muž', '1']:
                gender_mapping[val] = 'Male'
            elif str_val in ['f', 'female', 'zena', 'žena', 'z', '2']:
                gender_mapping[val] = 'Female'
                
        # If we found mappings, create a standardized gender column
        if gender_mapping:
            df['Standardized_Gender'] = df[gender_col].map(gender_mapping)
            gender_col = 'Standardized_Gender'
            results.append(f"Standardized gender values: {gender_mapping}")
        
        # Check the number of unique gender values
        gender_counts = df[gender_col].value_counts()
        results.append(f"Gender distribution: {dict(gender_counts)}")
    except Exception as e:
        results.append(f"Error standardizing gender: {str(e)}")
    
    # Create diagnosis categories if not already there
    if 'Diagnosis_Category' not in df.columns:
        try:
            def categorize_diagnosis(diagnosis):
                if pd.isna(diagnosis):
                    return "Unknown"
                
                # Check for liver diseases
                liver_disease_codes = ['K76.0', 'K75.9', 'K70', 'K71', 'K72', 'K73', 'K74', 'K76', 'K77']
                for code in liver_disease_codes:
                    if str(diagnosis).startswith(code):
                        return "Liver Disease"
                
                # Check for other major categories
                if str(diagnosis).startswith('K'):
                    return "Other Digestive System"
                elif str(diagnosis).startswith('E'):
                    return "Endocrine/Metabolic"
                elif str(diagnosis).startswith('B'):
                    return "Infectious Disease"
                elif str(diagnosis).startswith('D'):
                    return "Blood Disorders"
                elif str(diagnosis).startswith('C'):
                    return "Neoplasms"
                else:
                    return "Other"
            
            df['Diagnosis_Category'] = df[diagnoza_col].apply(categorize_diagnosis)
            diag_cat_counts = df['Diagnosis_Category'].value_counts()
            results.append(f"Created diagnosis categories: {dict(diag_cat_counts)}")
        except Exception as e:
            results.append(f"Error creating diagnosis categories: {str(e)}")
            df['Diagnosis_Category'] = "Unknown"
    
    # For each HFE mutation
    for column in hfe_columns:
        # Create separate plots for each gender
        for gender_value in df[gender_col].dropna().unique():
            try:
                if pd.isna(gender_value):
                    continue
                    
                gender_subset = df[df[gender_col] == gender_value]
                results.append(f"Data for {gender_value}: {len(gender_subset)} rows")
                
                # Skip if too few samples (reduced threshold)
                if len(gender_subset) < 3:
                    results.append(f"Not enough data for {gender_value}")
                    continue
                
                # Filter out NaN values
                valid_data = gender_subset.dropna(subset=[column, 'Diagnosis_Category'])
                results.append(f"Valid data points for {column} and {gender_value}: {len(valid_data)}")
                
                # Skip if not enough data after filtering
                if len(valid_data) < 3:
                    results.append(f"Not enough valid data after filtering")
                    continue
                
                # Check if we have sufficient diagnosis categories
                diag_counts = valid_data['Diagnosis_Category'].value_counts()
                results.append(f"Diagnosis counts for {gender_value}: {dict(diag_counts)}")
                
                if len(diag_counts) < 2:
                    results.append(f"Not enough diagnosis categories for {gender_value}")
                    continue
                
                plt.figure(figsize=(12, 8))
                
                # Create crosstab 
                try:
                    diagnosis_by_mutation = pd.crosstab(
                        valid_data['Diagnosis_Category'], 
                        valid_data[column], 
                        normalize='index'
                    ) * 100
                    
                    # Skip if crosstab is empty or has only one row/column
                    if diagnosis_by_mutation.empty or diagnosis_by_mutation.shape[0] < 2 or diagnosis_by_mutation.shape[1] < 2:
                        results.append(f"Insufficient crosstab dimensions: {diagnosis_by_mutation.shape}")
                        plt.close()
                        continue
                        
                    diagnosis_by_mutation.plot(kind='bar', stacked=True)
                except Exception as e:
                    results.append(f"Error creating crosstab: {str(e)}")
                    plt.close()
                    continue
                
                plt.title(f'Diagnosis Distribution by {column} Genotype - {gender_value}')
                plt.xlabel('Diagnosis Category')
                plt.ylabel('Percentage (%)')
                plt.legend(title='Genotype')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # Convert to base64
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plots.append((f"{column} by Diagnosis - {gender_value}", img_str))
                plt.close()
                results.append(f"Successfully created plot for {column} and {gender_value}")
            except Exception as e:
                results.append(f"Error creating plot for {column} and {gender_value}: {str(e)}")
                plt.close()
                continue
    
    return plots, results 