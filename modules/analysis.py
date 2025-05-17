#!/usr/bin/env python3
# Analysis module for HFE mutation analysis

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def analyze_dataset(df):
    """
    Perform basic analysis on the dataset and return results
    """
    results = []
    results.append(f"Dataset Overview:")
    results.append(f"Number of rows: {df.shape[0]}")
    results.append(f"Number of columns: {df.shape[1]}")
    
    # Column names and indices
    results.append("\nColumn Names and Indices:")
    for i, col in enumerate(df.columns):
        results.append(f"Column {i}: '{col}'")
    
    # Missing values analysis
    results.append("\nMissing Values Analysis:")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_data = pd.DataFrame({
        'Column': missing.index,
        'Missing Values': missing.values,
        'Percentage (%)': missing_percent.values
    })
    missing_data = missing_data[missing_data['Missing Values'] > 0].reset_index(drop=True)
    if not missing_data.empty:
        for i, row in missing_data.iterrows():
            results.append(f"{row['Column']}: {row['Missing Values']} ({row['Percentage (%)']:.2f}%)")
    else:
        results.append("No missing values found")
    
    # Find HFE columns
    hfe_columns = []
    for col in df.columns:
        if 'hfe' in str(col).lower():
            hfe_columns.append(col)
    
    # Analyze HFE mutation distributions
    if hfe_columns:
        results.append("\nHFE Mutation Distribution:")
        for column in hfe_columns:
            results.append(f"\n{column}:")
            genotype_counts = df[column].value_counts()
            genotype_percentages = (genotype_counts / len(df) * 100).round(2)
            for genotype, count in genotype_counts.items():
                percentage = genotype_percentages[genotype]
                results.append(f"{genotype}: {count} patients ({percentage:.2f}%)")
    
    return results

def classify_hh_risk(row):
    """
    Classify hereditary hemochromatosis risk based on genotype combinations
    """
    # Extract genotypes, handling possible different column names
    c282y = h63d = s65c = None
    
    # Find the columns containing each mutation
    for col in row.index:
        if 'C282Y' in col:
            c282y = row[col]
        elif 'H63D' in col:
            h63d = row[col]
        elif 'S65C' in col:
            s65c = row[col]
    
    if not all([c282y, h63d, s65c]):
        return "Unknown (missing data)"
    
    # Check for C282Y homozygotes (highest risk)
    if c282y == "mutant":
        return "High Risk (C282Y homozygote)"
    
    # Check for compound heterozygotes (moderate risk)
    if c282y == "heterozygot" and h63d == "heterozygot":
        return "Moderate Risk (C282Y/H63D compound heterozygote)"
    if c282y == "heterozygot" and s65c == "heterozygot":
        return "Moderate Risk (C282Y/S65C compound heterozygote)"
    if h63d == "heterozygot" and s65c == "heterozygot":
        return "Lower Risk (H63D/S65C compound heterozygote)"
    
    # Check for H63D homozygotes (low to moderate risk)
    if h63d == "mutant":
        return "Lower Risk (H63D homozygote)"
    
    # Check for S65C homozygotes (rare, but considered lower risk)
    if s65c == "mutant":
        return "Lower Risk (S65C homozygote)"
    
    # Check for carriers (lower risk)
    if c282y == "heterozygot":
        return "Carrier (C282Y heterozygote)"
    if h63d == "heterozygot":
        return "Carrier (H63D heterozygote)"
    if s65c == "heterozygot":
        return "Carrier (S65C heterozygote)"
    
    # Everyone else
    return "Minimal Risk (no mutations)"

def analyze_hh_risk(df, hfe_columns):
    """
    Analyze hereditary hemochromatosis risk in the dataset
    """
    results = []
    results.append("HEREDITARY HEMOCHROMATOSIS RISK ANALYSIS")
    
    # Classify each patient's risk
    df["HH_Risk"] = df.apply(classify_hh_risk, axis=1)
    
    # Create a simplified risk category for statistical analysis
    def simplify_risk(risk):
        if "High Risk" in risk or "Moderate Risk" in risk:
            return "High/Moderate Risk"
        elif "Lower Risk" in risk:
            return "Lower Risk"
        elif "Carrier" in risk:
            return "Carrier"
        else:
            return "Minimal Risk"
    
    df["Risk_Category"] = df["HH_Risk"].apply(simplify_risk)
    
    # Count and percentage for each risk category
    risk_counts = df["HH_Risk"].value_counts()
    risk_percentages = (risk_counts / len(df) * 100).round(2)
    
    results.append("\nRisk Category Distribution:")
    for risk_category, count in risk_counts.items():
        results.append(f"{risk_category}: {count} patients ({risk_percentages[risk_category]:.2f}%)")
    
    # Summarize carriers and at-risk individuals
    carriers = df["HH_Risk"].str.contains("Carrier").sum()
    high_risk = df["HH_Risk"].str.contains("High Risk").sum()
    moderate_risk = df["HH_Risk"].str.contains("Moderate Risk").sum()
    lower_risk = df["HH_Risk"].str.contains("Lower Risk").sum()
    
    total_with_predisposition = high_risk + moderate_risk + lower_risk
    
    results.append("\nSUMMARY:")
    results.append(f"Total Carriers: {carriers} patients ({carriers/len(df)*100:.2f}%)")
    results.append(f"Total with Genetic Predisposition: {total_with_predisposition} patients ({total_with_predisposition/len(df)*100:.2f}%)")
    results.append(f"- High Risk: {high_risk} patients ({high_risk/len(df)*100:.2f}%)")
    results.append(f"- Moderate Risk: {moderate_risk} patients ({moderate_risk/len(df)*100:.2f}%)")
    results.append(f"- Lower Risk: {lower_risk} patients ({lower_risk/len(df)*100:.2f}%)")
    
    return df, results

def analyze_diagnosis_associations(df, hfe_columns):
    """
    Analyze associations between HFE mutations and diagnoses
    """
    results = []
    results.append("HFE MUTATIONS AND DIAGNOSIS ASSOCIATION ANALYSIS")
    
    # Define diagnosis categories of interest
    liver_disease_codes = ['K76.0', 'K75.9', 'K70', 'K71', 'K72', 'K73', 'K74', 'K76', 'K77']
    
    # Create a helper function to categorize diagnoses
    def categorize_diagnosis(diagnosis):
        if pd.isna(diagnosis):
            return "Unknown"
        
        # Check for liver diseases of interest
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
        else:
            return "Other"
    
    # Find diagnoza column
    diagnoza_col = None
    for col in df.columns:
        if 'diagnoza' in str(col).lower() or 'mkch' in str(col).lower():
            diagnoza_col = col
            break
    
    if not diagnoza_col:
        results.append("No diagnosis column found in the dataset")
        return df, results
    
    # Add diagnosis category column
    df['Diagnosis_Category'] = df[diagnoza_col].apply(categorize_diagnosis)
    
    # Also create a specific column for our two main liver diseases of interest
    def is_specific_liver_disease(diagnosis):
        if pd.isna(diagnosis):
            return "Other"
        if str(diagnosis).startswith('K76.0'):
            return "K76.0 (Fatty liver)"
        elif str(diagnosis).startswith('K75.9'):
            return "K75.9 (Inflammatory liver disease)"
        else:
            return "Other"
    
    df['Specific_Liver_Disease'] = df[diagnoza_col].apply(is_specific_liver_disease)
    
    # Create a column for any HFE mutation
    def has_any_mutation(row):
        for col in hfe_columns:
            if row[col] != 'normal':
                return "Mutation Present"
        return "No Mutation"
        
    df['Any_HFE_Mutation'] = df.apply(has_any_mutation, axis=1)
    
    # Print basic statistics
    results.append("\nDiagnosis category distribution:")
    diag_dist = df['Diagnosis_Category'].value_counts()
    for category, count in diag_dist.items():
        results.append(f"{category}: {count} patients ({count/len(df)*100:.2f}%)")
    
    # Create a binary column for each specific liver disease
    df['Has_K760'] = df[diagnoza_col].apply(lambda x: 1 if str(x).startswith('K76.0') else 0)
    df['Has_K759'] = df[diagnoza_col].apply(lambda x: 1 if str(x).startswith('K75.9') else 0)
    
    # Calculate prevalence in patients with and without mutations
    mutation_present = df['Any_HFE_Mutation'] == 'Mutation Present'
    no_mutation = df['Any_HFE_Mutation'] == 'No Mutation'
    
    # Check for division by zero
    if mutation_present.sum() > 0:
        k760_in_mutation = df[mutation_present]['Has_K760'].mean() * 100
        k759_in_mutation = df[mutation_present]['Has_K759'].mean() * 100
        results.append(f"\nK76.0 (Fatty liver) prevalence:")
        results.append(f"- In patients with HFE mutations: {k760_in_mutation:.2f}%")
        results.append(f"\nK75.9 (Inflammatory liver disease) prevalence:")
        results.append(f"- In patients with HFE mutations: {k759_in_mutation:.2f}%")
    else:
        results.append("\nNo patients with HFE mutations found for prevalence calculation")
    
    if no_mutation.sum() > 0:
        k760_in_no_mutation = df[no_mutation]['Has_K760'].mean() * 100
        k759_in_no_mutation = df[no_mutation]['Has_K759'].mean() * 100
        if mutation_present.sum() > 0:  # Only add this if we already have mutation data
            results.append(f"- In patients without HFE mutations: {k760_in_no_mutation:.2f}%")
            results.append(f"- In patients without HFE mutations: {k759_in_no_mutation:.2f}%")
        else:
            results.append(f"\nK76.0 (Fatty liver) prevalence:")
            results.append(f"- In patients without HFE mutations: {k760_in_no_mutation:.2f}%")
            results.append(f"\nK75.9 (Inflammatory liver disease) prevalence:")
            results.append(f"- In patients without HFE mutations: {k759_in_no_mutation:.2f}%")
    else:
        results.append("\nNo patients without HFE mutations found for prevalence calculation")
    
    # Create a 2x2 contingency table for each specific disease
    k760_table = pd.crosstab(df['Any_HFE_Mutation'], df['Has_K760'])
    k759_table = pd.crosstab(df['Any_HFE_Mutation'], df['Has_K759'])
    
    # Perform chi-square tests on the 2x2 tables if possible
    try:
        # Check if table is valid for chi-square test (no zeros)
        if k760_table.shape == (2, 2) and not (k760_table == 0).any().any():
            k760_chi2, k760_p, k760_dof, k760_expected = chi2_contingency(k760_table)
            results.append(f"\nChi-Square Test for K76.0: chi2={k760_chi2:.2f}, p={k760_p:.4f}")
            if k760_p < 0.05:
                results.append("There is a significant association between HFE mutations and K76.0 (Fatty liver).")
            else:
                results.append("No significant association found between HFE mutations and K76.0 (Fatty liver).")
        else:
            results.append("\nCould not perform chi-square test for K76.0 - insufficient or unbalanced data.")
    except Exception as e:
        results.append(f"\nError in chi-square test for K76.0: {str(e)}")
    
    try:
        # Check if table is valid for chi-square test (no zeros)
        if k759_table.shape == (2, 2) and not (k759_table == 0).any().any():
            k759_chi2, k759_p, k759_dof, k759_expected = chi2_contingency(k759_table)
            results.append(f"\nChi-Square Test for K75.9: chi2={k759_chi2:.2f}, p={k759_p:.4f}")
            if k759_p < 0.05:
                results.append("There is a significant association between HFE mutations and K75.9 (Inflammatory liver disease).")
            else:
                results.append("No significant association found between HFE mutations and K75.9 (Inflammatory liver disease).")
        else:
            results.append("\nCould not perform chi-square test for K75.9 - insufficient or unbalanced data.")
    except Exception as e:
        results.append(f"\nError in chi-square test for K75.9: {str(e)}")
    
    return df, results

def calculate_hwe(df, mutation_column, column_name):
    """
    Calculate Hardy-Weinberg equilibrium for a given mutation column.
    
    Args:
        df: DataFrame containing the genotype data
        mutation_column: Column name containing genotype data
        column_name: Human-readable name for the mutation
        
    Returns:
        Dictionary with HWE analysis results
    """
    # Count genotypes
    genotype_counts = df[mutation_column].value_counts()
    
    # Extract counts for each genotype (handle case where a genotype might be missing)
    normal_count = genotype_counts.get('normal', 0)
    heterozygote_count = genotype_counts.get('heterozygot', 0)
    mutant_count = genotype_counts.get('mutant', 0)
    
    total = normal_count + heterozygote_count + mutant_count
    
    # Check if total is zero to avoid division by zero
    if total == 0:
        return {
            "Mutation": column_name,
            "Total": 0,
            "Normal (observed)": 0,
            "Heterozygote (observed)": 0,
            "Mutant (observed)": 0,
            "Normal allele frequency (p)": 0,
            "Mutant allele frequency (q)": 0,
            "p + q": 0,
            "Normal (expected)": 0,
            "Heterozygote (expected)": 0,
            "Mutant (expected)": 0,
            "Chi-square": np.nan,
            "p-value": np.nan,
            "Degrees of freedom": 0,
            "In Hardy-Weinberg equilibrium": "Cannot determine (no data)"
        }
    
    # Calculate allele frequencies
    p = (2 * normal_count + heterozygote_count) / (2 * total)  # Normal allele frequency
    q = (2 * mutant_count + heterozygote_count) / (2 * total)  # Mutant allele frequency
    
    # Calculate expected genotype counts under Hardy-Weinberg equilibrium
    expected_normal = (p**2) * total
    expected_heterozygote = 2 * p * q * total
    expected_mutant = (q**2) * total
    
    # Perform chi-square test
    observed = np.array([normal_count, heterozygote_count, mutant_count])
    expected = np.array([expected_normal, expected_heterozygote, expected_mutant])
    
    # Filter out zeros to avoid division by zero in chi-square test
    valid_indices = expected > 0
    
    if sum(valid_indices) <= 1:
        # Not enough valid categories for chi-square test
        chi2 = np.nan
        p_value = np.nan
        df_freedom = 0
        is_in_hwe = "Cannot determine (insufficient data)"
    else:
        # Calculate chi-square and p-value using scipy.stats.chisquare
        from scipy import stats
        chi2, p_value = stats.chisquare(
            observed[valid_indices], 
            expected[valid_indices]
        )
        
        # Degrees of freedom: number of categories - 1 (allele frequency) - 1 = #categories - 2
        df_freedom = sum(valid_indices) - 1 - 1
        df_freedom = max(1, df_freedom)  # Ensure at least 1 degree of freedom
        
        # Determine if in Hardy-Weinberg equilibrium
        is_in_hwe = "Yes" if p_value > 0.05 else "No"
    
    # Return results
    results = {
        "Mutation": column_name,
        "Total": total,
        "Normal (observed)": normal_count,
        "Heterozygote (observed)": heterozygote_count,
        "Mutant (observed)": mutant_count,
        "Normal allele frequency (p)": p,
        "Mutant allele frequency (q)": q,
        "p + q": p + q,
        "Normal (expected)": expected_normal,
        "Heterozygote (expected)": expected_heterozygote,
        "Mutant (expected)": expected_mutant,
        "Chi-square": chi2,
        "p-value": p_value,
        "Degrees of freedom": df_freedom,
        "In Hardy-Weinberg equilibrium": is_in_hwe
    }
    
    return results

def analyze_hardy_weinberg(df, hfe_columns):
    """
    Analyze Hardy-Weinberg equilibrium for HFE mutations in the dataset
    """
    results = []
    results.append("HARDY-WEINBERG EQUILIBRIUM ANALYSIS")
    
    # Define human-readable names for mutations
    mutation_names = {
        "HFE C187G (H63D)\n[HFE]": "HFE H63D",
        "HFE A193T (S65C)\n[HFE]": "HFE S65C", 
        "HFE G845A (C282Y)\n[HFE]": "HFE C282Y"
    }
    
    # Store calculated HWE results for plotting
    hwe_results_list = []
    
    # Run Hardy-Weinberg equilibrium analysis for each mutation
    for column in hfe_columns:
        # Get human-readable name if available, otherwise use column name
        column_name = mutation_names.get(column, column)
        
        results.append(f"\n{'-'*50}")
        results.append(f"Analyzing {column_name}")
        results.append(f"{'-'*50}")
        
        # Print genotype distribution
        genotype_counts = df[column].value_counts()
        results.append("\nGenotype Distribution:")
        for genotype, count in genotype_counts.items():
            percentage = count / len(df) * 100
            results.append(f"{genotype}: {count} ({percentage:.2f}%)")
        
        # Calculate Hardy-Weinberg equilibrium
        hwe_results = calculate_hwe(df, column, column_name)
        hwe_results_list.append(hwe_results)
        
        # Print results
        results.append("\nHardy-Weinberg Equilibrium Analysis:")
        results.append(f"Total individuals: {hwe_results['Total']}")
        results.append(f"Normal allele frequency (p): {hwe_results['Normal allele frequency (p)']:.4f}")
        results.append(f"Mutant allele frequency (q): {hwe_results['Mutant allele frequency (q)']:.4f}")
        results.append(f"Sum (p + q): {hwe_results['p + q']:.4f} (should be close to 1.0)")
        
        results.append("\nObserved vs. Expected Counts:")
        results.append(f"Normal (wildtype): {hwe_results['Normal (observed)']:.1f} observed vs {hwe_results['Normal (expected)']:.1f} expected")
        results.append(f"Heterozygote: {hwe_results['Heterozygote (observed)']:.1f} observed vs {hwe_results['Heterozygote (expected)']:.1f} expected")
        results.append(f"Mutant (homozygote): {hwe_results['Mutant (observed)']:.1f} observed vs {hwe_results['Mutant (expected)']:.1f} expected")
        
        results.append("\nStatistical Test:")
        if np.isnan(hwe_results['Chi-square']):
            results.append("Chi-square test could not be performed (insufficient data)")
        else:
            results.append(f"Chi-square: {hwe_results['Chi-square']:.4f}")
            results.append(f"p-value: {hwe_results['p-value']:.4f}")
            results.append(f"Degrees of freedom: {hwe_results['Degrees of freedom']}")
        
        results.append(f"\nIn Hardy-Weinberg equilibrium: {hwe_results['In Hardy-Weinberg equilibrium']}")
    
    return df, results, hwe_results_list 