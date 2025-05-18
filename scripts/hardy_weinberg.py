#!/usr/bin/env python3
# Script to check Hardy-Weinberg equilibrium for HFE mutations

import pandas as pd
import numpy as np
from scipy import stats
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

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
        # Calculate chi-square and p-value
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

def plot_genotype_distribution(results_list, output_dir=None):
    """
    Create bar charts comparing observed vs expected genotype distributions
    
    Args:
        results_list: List of dictionaries with HWE analysis results
        output_dir: Directory to save plots (None for display only)
    """
    for result in results_list:
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
        
        # Save or show plot
        if output_dir:
            output_path = Path(output_dir) / f"hwe_{mutation.replace(' ', '_').replace('(', '').replace(')', '')}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_path}")
        else:
            plt.tight_layout()
            plt.show()
        
        plt.close()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Check Hardy-Weinberg equilibrium for mutations")
    parser.add_argument("--path", "-p", type=str, default="../SSBU25_dataset_modified_new.xlsx",
                        help="Path to the dataset file (default: SSBU25_dataset_modified_new.xlsx)")
    parser.add_argument("--plots", "-g", action="store_true", 
                        help="Generate plots for observed vs. expected genotype distributions")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Directory to save plots (default: display only)")
    args = parser.parse_args()
    
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
        
        # Define human-readable names for mutations
        mutation_names = {
            "HFE C187G (H63D)\n[HFE]": "HFE H63D",
            "HFE A193T (S65C)\n[HFE]": "HFE S65C", 
            "HFE G845A (C282Y)\n[HFE]": "HFE C282Y"
        }
        
        # Run Hardy-Weinberg equilibrium analysis for each mutation
        results_list = []
        
        for column in hfe_columns:
            # Get human-readable name if available, otherwise use column name
            column_name = mutation_names.get(column, column)
            
            print(f"\n{'-'*50}")
            print(f"Analyzing {column_name}")
            print(f"{'-'*50}")
            
            # Print genotype distribution
            genotype_counts = df[column].value_counts()
            print("\nGenotype Distribution:")
            print(genotype_counts)
            
            # Calculate Hardy-Weinberg equilibrium
            results = calculate_hwe(df, column, column_name)
            results_list.append(results)
            
            # Print results
            print("\nHardy-Weinberg Equilibrium Analysis:")
            print(f"Total individuals: {results['Total']}")
            print(f"Normal allele frequency (p): {results['Normal allele frequency (p)']:.4f}")
            print(f"Mutant allele frequency (q): {results['Mutant allele frequency (q)']:.4f}")
            print(f"Sum (p + q): {results['p + q']:.4f} (should be close to 1.0)")
            
            print("\nObserved vs. Expected Counts:")
            print(f"Normal (wildtype): {results['Normal (observed)']:.1f} observed vs {results['Normal (expected)']:.1f} expected")
            print(f"Heterozygote: {results['Heterozygote (observed)']:.1f} observed vs {results['Heterozygote (expected)']:.1f} expected")
            print(f"Mutant (homozygote): {results['Mutant (observed)']:.1f} observed vs {results['Mutant (expected)']:.1f} expected")
            
            print("\nStatistical Test:")
            if np.isnan(results['Chi-square']):
                print("Chi-square test could not be performed (insufficient data)")
            else:
                print(f"Chi-square: {results['Chi-square']:.4f}")
                print(f"p-value: {results['p-value']:.4f}")
                print(f"Degrees of freedom: {results['Degrees of freedom']}")
            
            print(f"\nIn Hardy-Weinberg equilibrium: {results['In Hardy-Weinberg equilibrium']}")
        
        # Generate plots if requested
        if args.plots:
            plot_genotype_distribution(results_list, args.output)
        
    except Exception as e:
        print(f"Error analyzing the dataset: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 