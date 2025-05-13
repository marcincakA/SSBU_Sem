#!/usr/bin/env python3
# Script to analyze associations between HFE mutations and diagnoses

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

def analyze_genotype_distribution(df, hfe_columns):
    """
    Calculate and display the percentage distribution of genotypes for each mutation
    """
    results = {}
    
    print("\n" + "="*60)
    print("GENOTYPE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    for column in hfe_columns:
        print(f"\n{'-'*50}")
        print(f"Analyzing {column}")
        print(f"{'-'*50}")
        
        # Get genotype counts
        genotype_counts = df[column].value_counts()
        total = len(df)
        
        # Calculate percentages
        genotype_percentages = (genotype_counts / total * 100).round(2)
        
        print("\nGenotype Distribution:")
        for genotype, count in genotype_counts.items():
            percentage = genotype_percentages[genotype]
            print(f"{genotype}: {count} patients ({percentage:.2f}%)")
        
        # Store results for later use
        results[column] = {
            'counts': genotype_counts,
            'percentages': genotype_percentages
        }
    
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
    print("\n" + "="*60)
    print("HEREDITARY HEMOCHROMATOSIS RISK ANALYSIS")
    print("="*60)
    
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
    
    print("\nRisk Category Distribution:")
    for risk_category, count in risk_counts.items():
        print(f"{risk_category}: {count} patients ({risk_percentages[risk_category]:.2f}%)")
    
    # Summarize carriers and at-risk individuals
    carriers = df["HH_Risk"].str.contains("Carrier").sum()
    high_risk = df["HH_Risk"].str.contains("High Risk").sum()
    moderate_risk = df["HH_Risk"].str.contains("Moderate Risk").sum()
    lower_risk = df["HH_Risk"].str.contains("Lower Risk").sum()
    
    total_with_predisposition = high_risk + moderate_risk + lower_risk
    
    print("\nSUMMARY:")
    print(f"Total Carriers: {carriers} patients ({carriers/len(df)*100:.2f}%)")
    print(f"Total with Genetic Predisposition: {total_with_predisposition} patients ({total_with_predisposition/len(df)*100:.2f}%)")
    print(f"- High Risk: {high_risk} patients ({high_risk/len(df)*100:.2f}%)")
    print(f"- Moderate Risk: {moderate_risk} patients ({moderate_risk/len(df)*100:.2f}%)")
    print(f"- Lower Risk: {lower_risk} patients ({lower_risk/len(df)*100:.2f}%)")
    
    return {
        'risk_counts': risk_counts,
        'risk_percentages': risk_percentages,
        'carriers': carriers,
        'high_risk': high_risk,
        'moderate_risk': moderate_risk,
        'lower_risk': lower_risk,
        'total_with_predisposition': total_with_predisposition
    }

def analyze_diagnosis_associations(df, hfe_columns):
    """
    Analyze associations between HFE mutations and diagnoses
    """
    print("\n" + "="*60)
    print("HFE MUTATIONS AND DIAGNOSIS ASSOCIATION ANALYSIS")
    print("="*60)
    
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
    
    # Add diagnosis category column
    df['Diagnosis_Category'] = df['diagnoza MKCH-10'].apply(categorize_diagnosis)
    
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
    
    df['Specific_Liver_Disease'] = df['diagnoza MKCH-10'].apply(is_specific_liver_disease)
    
    # Create a column for any HFE mutation
    def has_any_mutation(row):
        for col in hfe_columns:
            if row[col] != 'normal':
                return "Mutation Present"
        return "No Mutation"
        
    df['Any_HFE_Mutation'] = df.apply(has_any_mutation, axis=1)
    
    # Print basic statistics
    print("\nDiagnosis category distribution:")
    diag_dist = df['Diagnosis_Category'].value_counts()
    for category, count in diag_dist.items():
        print(f"{category}: {count} patients ({count/len(df)*100:.2f}%)")
    
    # Analyze associations between any HFE mutation and diagnoses
    print(f"\n{'-'*50}")
    print(f"Analyzing Any HFE Mutation vs Diagnoses")
    print(f"{'-'*50}")
    
    # Create contingency table
    mutation_diag_table = pd.crosstab(df['Any_HFE_Mutation'], df['Diagnosis_Category'])
    print("\nContingency table (counts):")
    print(mutation_diag_table)
    
    # Calculate percentages within mutation groups
    mutation_diag_pct = pd.crosstab(df['Any_HFE_Mutation'], df['Diagnosis_Category'], normalize='index') * 100
    print("\nPercentages within mutation groups:")
    print(mutation_diag_pct.round(2))
    
    # Perform chi-square test
    chi2, p, dof, expected = chi2_contingency(mutation_diag_table)
    print(f"\nChi-Square Test: chi2={chi2:.2f}, p={p:.4f}, dof={dof}")
    if p < 0.05:
        print("There is a significant association between HFE mutations and diagnoses.")
    else:
        print("No significant association found between HFE mutations and diagnoses.")
    
    # Focus on specific liver diseases (K76.0 and K75.9)
    print(f"\n{'-'*50}")
    print(f"Analysis of Specific Liver Diseases (K76.0 and K75.9)")
    print(f"{'-'*50}")
    
    # Create a binary column for each specific liver disease
    df['Has_K760'] = df['diagnoza MKCH-10'].apply(lambda x: 1 if str(x).startswith('K76.0') else 0)
    df['Has_K759'] = df['diagnoza MKCH-10'].apply(lambda x: 1 if str(x).startswith('K75.9') else 0)
    
    # Calculate prevalence in patients with and without mutations
    k760_in_mutation = df[df['Any_HFE_Mutation'] == 'Mutation Present']['Has_K760'].mean() * 100
    k760_in_no_mutation = df[df['Any_HFE_Mutation'] == 'No Mutation']['Has_K760'].mean() * 100
    
    k759_in_mutation = df[df['Any_HFE_Mutation'] == 'Mutation Present']['Has_K759'].mean() * 100
    k759_in_no_mutation = df[df['Any_HFE_Mutation'] == 'No Mutation']['Has_K759'].mean() * 100
    
    print(f"K76.0 (Fatty liver) prevalence:")
    print(f"- In patients with HFE mutations: {k760_in_mutation:.2f}%")
    print(f"- In patients without HFE mutations: {k760_in_no_mutation:.2f}%")
    print(f"- Ratio: {k760_in_mutation/k760_in_no_mutation if k760_in_no_mutation else 'N/A':.2f}")
    
    print(f"\nK75.9 (Inflammatory liver disease) prevalence:")
    print(f"- In patients with HFE mutations: {k759_in_mutation:.2f}%")
    print(f"- In patients without HFE mutations: {k759_in_no_mutation:.2f}%")
    print(f"- Ratio: {k759_in_mutation/k759_in_no_mutation if k759_in_no_mutation else 'N/A':.2f}")
    
    # Chi-square test for specific liver diseases
    specific_table = pd.crosstab(df['Any_HFE_Mutation'], df['Specific_Liver_Disease'])
    print("\nContingency table for specific liver diseases (counts):")
    print(specific_table)
    
    # Create a 2x2 contingency table for each specific disease
    k760_table = pd.crosstab(df['Any_HFE_Mutation'], df['Has_K760'])
    k759_table = pd.crosstab(df['Any_HFE_Mutation'], df['Has_K759'])
    
    print("\nContingency table for K76.0 (counts):")
    print(k760_table)
    
    print("\nContingency table for K75.9 (counts):")
    print(k759_table)
    
    # Perform chi-square tests on the 2x2 tables
    try:
        k760_chi2, k760_p, k760_dof, k760_expected = chi2_contingency(k760_table)
        print(f"\nChi-Square Test for K76.0: chi2={k760_chi2:.2f}, p={k760_p:.4f}")
        if k760_p < 0.05:
            print("There is a significant association between HFE mutations and K76.0 (Fatty liver).")
        else:
            print("No significant association found between HFE mutations and K76.0 (Fatty liver).")
    except:
        print("Could not perform chi-square test for K76.0 - may have insufficient data.")
    
    try:
        k759_chi2, k759_p, k759_dof, k759_expected = chi2_contingency(k759_table)
        print(f"\nChi-Square Test for K75.9: chi2={k759_chi2:.2f}, p={k759_p:.4f}")
        if k759_p < 0.05:
            print("There is a significant association between HFE mutations and K75.9 (Inflammatory liver disease).")
        else:
            print("No significant association found between HFE mutations and K75.9 (Inflammatory liver disease).")
    except:
        print("Could not perform chi-square test for K75.9 - may have insufficient data.")
    
    # Analyze associations between risk categories and liver diseases
    print(f"\n{'-'*50}")
    print(f"Analyzing HH Risk Categories vs Liver Diseases")
    print(f"{'-'*50}")
    
    risk_liver_table = pd.crosstab(df['Risk_Category'], df['Diagnosis_Category'])
    print("\nContingency table for Risk Categories vs Diagnosis Categories (counts):")
    print(risk_liver_table)
    
    # Compute percentages within risk groups
    risk_liver_pct = pd.crosstab(df['Risk_Category'], df['Diagnosis_Category'], normalize='index') * 100
    print("\nPercentages within risk groups:")
    print(risk_liver_pct.round(2))
    
    # Chi-square test for risk categories and diagnoses
    try:
        risk_chi2, risk_p, risk_dof, risk_expected = chi2_contingency(risk_liver_table)
        print(f"\nChi-Square Test for Risk Categories vs Diagnoses: chi2={risk_chi2:.2f}, p={risk_p:.4f}")
        if risk_p < 0.05:
            print("There is a significant association between HH risk categories and diagnoses.")
        else:
            print("No significant association found between HH risk categories and diagnoses.")
    except:
        print("Could not perform chi-square test - may have insufficient data in some categories.")
    
    # Focus on high/moderate risk vs specific liver diseases
    high_mod_risk_k760 = df[df['Risk_Category'] == 'High/Moderate Risk']['Has_K760'].mean() * 100
    lower_risk_k760 = df[df['Risk_Category'] == 'Lower Risk']['Has_K760'].mean() * 100
    carrier_k760 = df[df['Risk_Category'] == 'Carrier']['Has_K760'].mean() * 100
    minimal_risk_k760 = df[df['Risk_Category'] == 'Minimal Risk']['Has_K760'].mean() * 100
    
    high_mod_risk_k759 = df[df['Risk_Category'] == 'High/Moderate Risk']['Has_K759'].mean() * 100
    lower_risk_k759 = df[df['Risk_Category'] == 'Lower Risk']['Has_K759'].mean() * 100
    carrier_k759 = df[df['Risk_Category'] == 'Carrier']['Has_K759'].mean() * 100
    minimal_risk_k759 = df[df['Risk_Category'] == 'Minimal Risk']['Has_K759'].mean() * 100
    
    print(f"\nK76.0 (Fatty liver) prevalence by risk category:")
    print(f"- High/Moderate Risk: {high_mod_risk_k760:.2f}%")
    print(f"- Lower Risk: {lower_risk_k760:.2f}%")
    print(f"- Carrier: {carrier_k760:.2f}%")
    print(f"- Minimal Risk: {minimal_risk_k760:.2f}%")
    
    print(f"\nK75.9 (Inflammatory liver disease) prevalence by risk category:")
    print(f"- High/Moderate Risk: {high_mod_risk_k759:.2f}%")
    print(f"- Lower Risk: {lower_risk_k759:.2f}%")
    print(f"- Carrier: {carrier_k759:.2f}%")
    print(f"- Minimal Risk: {minimal_risk_k759:.2f}%")
    
    return df

def plot_diagnosis_associations(df, output_dir=None):
    """
    Create visualizations for HFE mutation and diagnosis associations
    """
    # 1. Plot frequency of diagnoses by HFE mutation status
    plt.figure(figsize=(12, 8))
    diagnosis_mutation = pd.crosstab(df['Diagnosis_Category'], df['Any_HFE_Mutation'], normalize='index') * 100
    diagnosis_mutation.plot(kind='bar', stacked=True)
    plt.title('Diagnosis Categories by HFE Mutation Status')
    plt.xlabel('Diagnosis Category')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Mutation Status')
    
    if output_dir:
        output_path = Path(output_dir) / "diagnosis_by_mutation.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    plt.close()
    
    # 2. Plot specific liver diseases by mutation status
    plt.figure(figsize=(12, 8))
    liver_disease_mutation = pd.crosstab(df['Specific_Liver_Disease'], df['Any_HFE_Mutation'], normalize='index') * 100
    liver_disease_mutation.plot(kind='bar', stacked=True)
    plt.title('Specific Liver Diseases by HFE Mutation Status')
    plt.xlabel('Liver Disease')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Mutation Status')
    
    if output_dir:
        output_path = Path(output_dir) / "liver_disease_by_mutation.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    plt.close()
    
    # 3. Plot liver disease prevalence by risk category
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
    
    if output_dir:
        output_path = Path(output_dir) / "liver_disease_by_risk.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    plt.close()
    
    # 4. Plot distribution of diagnoses within high/moderate risk patients
    high_mod_patients = df[df['Risk_Category'] == 'High/Moderate Risk']
    if len(high_mod_patients) > 10:  # Only create plot if enough patients
        plt.figure(figsize=(12, 8))
        high_mod_diag = high_mod_patients['Diagnosis_Category'].value_counts()
        high_mod_diag_pct = (high_mod_diag / len(high_mod_patients) * 100).round(1)
        
        ax = high_mod_diag.plot(kind='bar')
        
        # Add percentage labels
        for i, p in enumerate(ax.patches):
            ax.annotate(f'{high_mod_diag.iloc[i]} ({high_mod_diag_pct.iloc[i]}%)', 
                      (p.get_x() + p.get_width()/2., p.get_height()), 
                      ha = 'center', va = 'bottom', rotation=0)
        
        plt.title('Diagnoses in High/Moderate Risk HH Patients')
        plt.xlabel('Diagnosis Category')
        plt.ylabel('Number of Patients')
        plt.xticks(rotation=45, ha='right')
        
        if output_dir:
            output_path = Path(output_dir) / "high_risk_diagnoses.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_path}")
        else:
            plt.tight_layout()
            plt.show()
        
        plt.close()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Analyze associations between HFE mutations and diagnoses")
    parser.add_argument("--path", "-p", type=str, default="SSBU25_dataset_modified_new.xlsx",
                        help="Path to the dataset file (default: SSBU25_dataset_modified_new.xlsx)")
    parser.add_argument("--plots", "-g", action="store_true", 
                        help="Generate plots for genotype and diagnosis associations")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Directory to save plots (default: display only)")
    args = parser.parse_args()
    
    # Create output directory if specified and doesn't exist
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True, parents=True)
    else:
        output_dir = None
    
    # Load the dataset
    file_path = Path(args.path)
    print(f"Loading dataset from {file_path.absolute()}")
    
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Find HFE mutation columns
        hfe_columns = []
        for col in df.columns:
            if 'hfe' in str(col).lower():
                hfe_columns.append(col)
        
        if not hfe_columns:
            print("Error: No HFE mutation columns found in the dataset")
            return
        
        print(f"\nFound {len(hfe_columns)} HFE mutation columns for analysis:")
        for col in hfe_columns:
            print(f"- {col}")
        
        # Analyze genotype distribution
        analyze_genotype_distribution(df, hfe_columns)
        
        # Analyze hereditary hemochromatosis risk
        analyze_hh_risk(df, hfe_columns)
        
        # Analyze associations with diagnoses
        df = analyze_diagnosis_associations(df, hfe_columns)
        
        # Generate plots if requested
        if args.plots:
            plot_diagnosis_associations(df, output_dir)
            
    except Exception as e:
        print(f"Error analyzing the dataset: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 