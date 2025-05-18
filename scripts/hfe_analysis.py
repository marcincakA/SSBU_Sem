#!/usr/bin/env python3
# Script to analyze HFE genotype distributions and hereditary hemochromatosis risk

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_genotype_distribution(df, hfe_columns, output_dir=None):
    """
    Create visualizations for genotype distributions
    """
    for column in hfe_columns:
        # Get data
        genotype_counts = df[column].value_counts()
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create bar chart
        ax = sns.barplot(x=genotype_counts.index, y=genotype_counts.values)
        
        # Add count and percentage labels
        total = len(df)
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            percentage = height / total * 100
            ax.annotate(f'{int(height)} ({percentage:.1f}%)', 
                        (p.get_x() + p.get_width() / 2., height),
                        ha = 'center', va = 'bottom', xytext = (0, 10),
                        textcoords = 'offset points')
        
        # Add labels and title
        plt.xlabel('Genotype')
        plt.ylabel('Number of Patients')
        column_name = column.split('\n')[0]
        plt.title(f'Genotype Distribution for {column_name}')
        
        # Save or display
        if output_dir:
            output_path = Path(output_dir) / f"genotype_{column_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_path}")
        else:
            plt.tight_layout()
            plt.show()
            
        plt.close()

def plot_hh_risk(df, output_dir=None):
    """
    Create visualization for hereditary hemochromatosis risk distribution
    """
    # Get risk categories, excluding "Minimal Risk" to focus on mutations
    risk_counts = df["HH_Risk"].value_counts()
    risk_counts = risk_counts[~risk_counts.index.str.contains("Minimal Risk")]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create bar chart
    ax = sns.barplot(x=risk_counts.index, y=risk_counts.values)
    
    # Rotate x labels for readability
    plt.xticks(rotation=45, ha='right')
    
    # Add count and percentage labels
    total = len(df)
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        percentage = height / total * 100
        ax.annotate(f'{int(height)} ({percentage:.1f}%)', 
                    (p.get_x() + p.get_width() / 2., height),
                    ha = 'center', va = 'bottom', xytext = (0, 10),
                    textcoords = 'offset points')
    
    # Add labels and title
    plt.xlabel('Risk Category')
    plt.ylabel('Number of Patients')
    plt.title('Hereditary Hemochromatosis Risk Distribution')
    
    # Save or display
    if output_dir:
        output_path = Path(output_dir) / "hh_risk_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.tight_layout()
        plt.show()
        
    plt.close()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Analyze HFE genotypes and hereditary hemochromatosis risk")
    parser.add_argument("--path", "-p", type=str, default="../SSBU25_dataset_modified_new.xlsx",
                        help="Path to the dataset file (default: SSBU25_dataset_modified_new.xlsx)")
    parser.add_argument("--plots", "-g", action="store_true", 
                        help="Generate plots for genotype and risk distributions")
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
        genotype_results = analyze_genotype_distribution(df, hfe_columns)
        
        # Analyze hereditary hemochromatosis risk
        risk_results = analyze_hh_risk(df, hfe_columns)
        
        # Generate plots if requested
        if args.plots:
            plot_genotype_distribution(df, hfe_columns, output_dir)
            plot_hh_risk(df, output_dir)
            
    except Exception as e:
        print(f"Error analyzing the dataset: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 