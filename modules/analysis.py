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

def calculate_hwe(df, mutation_column, column_name, tests=None):
    """
    Calculate Hardy-Weinberg equilibrium for a given mutation column.
    
    Args:
        df: DataFrame containing the genotype data
        mutation_column: Column name containing genotype data
        column_name: Human-readable name for the mutation
        tests: List of tests to perform ('chi_square', 'exact', 'logistic', 'bayesian')
        
    Returns:
        Dictionary with HWE analysis results
    """
    if tests is None:
        tests = ['chi_square']  # Default to chi-square test only
        
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
            "p-value (chi-square)": np.nan,
            "p-value (exact)": np.nan,
            "p-value (logistic)": np.nan,
            "Bayesian posterior": np.nan,
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
    
    # Initialize test results with NaN
    chi2 = np.nan
    chi2_p_value = np.nan
    exact_p_value = np.nan
    logistic_p_value = np.nan
    bayesian_posterior = np.nan
    df_freedom = 0
    is_in_hwe = "Cannot determine (insufficient data)"
    
    # Observed and expected values for tests
    observed = np.array([normal_count, heterozygote_count, mutant_count])
    expected = np.array([expected_normal, expected_heterozygote, expected_mutant])
    
    # Perform Chi-square test if requested
    if 'chi_square' in tests:
        # Filter out zeros to avoid division by zero in chi-square test
        valid_indices = expected > 0
        
        if sum(valid_indices) <= 1:
            # Not enough valid categories for chi-square test
            chi2 = np.nan
            chi2_p_value = np.nan
            df_freedom = 0
        else:
            # Calculate chi-square and p-value using scipy.stats.chisquare
            from scipy import stats
            chi2, chi2_p_value = stats.chisquare(
                observed[valid_indices], 
                expected[valid_indices]
            )
            
            # Degrees of freedom: number of categories - 1 (allele frequency) - 1 = #categories - 2
            df_freedom = sum(valid_indices) - 1 - 1
            df_freedom = max(1, df_freedom)  # Ensure at least 1 degree of freedom
    
    # Perform Exact test (Fisher's exact test) if requested
    if 'exact' in tests:
        try:
            from scipy import stats
            
            # Create a 2x3 table for exact test
            # Rows: alleles (normal, mutant)
            # Columns: genotypes (normal, heterozygote, mutant)
            
            # Calculate allele counts in each genotype category
            normal_alleles_in_normal = 2 * normal_count
            normal_alleles_in_heterozygote = heterozygote_count
            normal_alleles_in_mutant = 0
            
            mutant_alleles_in_normal = 0
            mutant_alleles_in_heterozygote = heterozygote_count
            mutant_alleles_in_mutant = 2 * mutant_count
            
            # Create the table
            table = np.array([
                [normal_alleles_in_normal, normal_alleles_in_heterozygote, normal_alleles_in_mutant],
                [mutant_alleles_in_normal, mutant_alleles_in_heterozygote, mutant_alleles_in_mutant]
            ])
            
            # Perform Fisher's exact test if there's enough data
            if np.sum(table) > 0 and not np.any(np.isnan(table)):
                # Flatten the table for stats.fisher_exact - it only works with 2x2 tables directly
                # So we'll use a simulation approach for the 2x3 table
                if sum(observed) >= 5 and min(expected) >= 1:
                    _, exact_p_value = stats.chi2_contingency(table, simulate_p_value=True)
                else:
                    exact_p_value = np.nan  # Too little data for accurate simulation
            else:
                exact_p_value = np.nan
        except Exception as e:
            exact_p_value = np.nan
            print(f"Error in exact test: {str(e)}")
            
    # Perform Logistic regression if requested
    if 'logistic' in tests and 'Diagnosis_Category' in df.columns:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import log_loss
            
            # We need to see if there's a disease association to test with logistic regression
            # Using the Diagnosis_Category column for disease status (binary: Liver Disease or not)
            if 'Diagnosis_Category' in df.columns:
                # Create binary outcome variable (1 = Liver Disease, 0 = Other)
                df_subset = df[[mutation_column, 'Diagnosis_Category']].copy()
                df_subset['is_liver_disease'] = df_subset['Diagnosis_Category'].apply(
                    lambda x: 1 if 'Liver Disease' in str(x) else 0
                )
                
                # Create numeric predictor (0=normal, 1=heterozygote, 2=mutant)
                df_subset['genotype_numeric'] = df_subset[mutation_column].map({
                    'normal': 0, 
                    'heterozygot': 1, 
                    'mutant': 2
                })
                
                # Drop rows with missing values
                df_subset = df_subset.dropna(subset=['genotype_numeric', 'is_liver_disease'])
                
                if len(df_subset) > 10:  # Enough data for regression
                    # Fit logistic regression
                    X = df_subset['genotype_numeric'].values.reshape(-1, 1)
                    y = df_subset['is_liver_disease'].values
                    
                    model = LogisticRegression(random_state=42)
                    model.fit(X, y)
                    
                    # Calculate p-value from log-likelihood ratio test
                    null_model = LogisticRegression(fit_intercept=True, random_state=42)
                    null_model.fit(np.ones((len(X), 1)), y)
                    
                    # Get log-likelihoods
                    y_pred_full = model.predict_proba(X)
                    y_pred_null = null_model.predict_proba(np.ones((len(X), 1)))
                    
                    # Calculate log-likelihood for both models
                    ll_full = -log_loss(y, y_pred_full) * len(X)
                    ll_null = -log_loss(y, y_pred_null) * len(X)
                    
                    # LR test statistic
                    lr_stat = 2 * (ll_full - ll_null)
                    
                    # P-value (chi-square distribution with 1 df)
                    from scipy.stats import chi2
                    logistic_p_value = 1 - chi2.cdf(lr_stat, 1)
                else:
                    logistic_p_value = np.nan  # Not enough data
            else:
                logistic_p_value = np.nan  # No disease information for logistic regression
        except Exception as e:
            logistic_p_value = np.nan
            print(f"Error in logistic regression: {str(e)}")
    
    # Perform Bayesian analysis if requested
    if 'bayesian' in tests:
        try:
            from scipy import stats
            
            # Bayesian approach using Beta distribution as prior
            # Beta(1,1) is a uniform prior - no previous knowledge
            # We update using the observed data to get the posterior
            
            # For HWE, we're interested in the parameter theta where:
            # theta measures the deviation from HWE
            # theta = 0 means perfect HWE
            
            # We'll use a simple Bayesian approach: compute the posterior probability
            # that the population is in HWE given the observed genotype counts
            
            # To simplify, we'll compute P(observed | HWE) using multinomial distribution
            # P(HWE) is our prior - we'll set to 0.5 (equal prior probability)
            
            # Calculate the probability of observing these genotypes under HWE
            # (Using Dirichlet-Multinomial conjugate model)
            
            # Prior concentration parameters (1,1,1) = uniform
            alpha_prior = np.array([1, 1, 1])
            
            # Posterior = prior + observed
            alpha_posterior = alpha_prior + observed
            
            # Simulate samples from posterior
            n_samples = 10000
            
            # Sample p from Dirichlet distribution
            p_samples = stats.dirichlet.rvs(alpha_posterior, size=n_samples)
            
            # For each p, calculate expected HWE frequencies
            # For HWE: p_AA = p_A^2, p_Aa = 2*p_A*p_a, p_aa = p_a^2
            hwe_samples = np.zeros((n_samples, 3))
            
            # Estimate p_A and p_a from each sample
            for i in range(n_samples):
                p_sample = p_samples[i]
                
                # Calculate allele frequencies from genotype frequencies
                p_A = (2*p_sample[0] + p_sample[1])/2  # p_AA + p_Aa/2
                p_a = (2*p_sample[2] + p_sample[1])/2  # p_aa + p_Aa/2
                
                # Calculate HWE genotype frequencies
                hwe_samples[i, 0] = p_A**2            # p_AA
                hwe_samples[i, 1] = 2 * p_A * p_a     # p_Aa
                hwe_samples[i, 2] = p_a**2            # p_aa
            
            # Calculate the proportion of samples that suggest HWE
            # by comparing the original and HWE-expected frequencies
            
            # Convert to counts
            original_frequency = observed / np.sum(observed)
            
            # Calculate Euclidean distance between observed and HWE frequencies
            distances = np.sqrt(np.sum((p_samples - hwe_samples)**2, axis=1))
            
            # Proportion of samples where distance is small (threshold: 0.1)
            bayesian_posterior = np.mean(distances < 0.1)
            
        except Exception as e:
            bayesian_posterior = np.nan
            print(f"Error in Bayesian analysis: {str(e)}")
            
    # Determine if in Hardy-Weinberg equilibrium based on available tests
    if 'chi_square' in tests and not np.isnan(chi2_p_value):
        is_in_hwe = "Yes" if chi2_p_value > 0.05 else "No"
    elif 'exact' in tests and not np.isnan(exact_p_value):
        is_in_hwe = "Yes" if exact_p_value > 0.05 else "No"
    elif 'bayesian' in tests and not np.isnan(bayesian_posterior):
        is_in_hwe = "Yes" if bayesian_posterior > 0.5 else "No"
    else:
        is_in_hwe = "Cannot determine (insufficient data)"
    
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
        "p-value (chi-square)": chi2_p_value,
        "p-value (exact)": exact_p_value,
        "p-value (logistic)": logistic_p_value,
        "Bayesian posterior": bayesian_posterior,
        "Degrees of freedom": df_freedom,
        "In Hardy-Weinberg equilibrium": is_in_hwe
    }
    
    return results

def analyze_hardy_weinberg(df, hfe_columns, hwe_tests=None):
    """
    Analyze Hardy-Weinberg equilibrium for HFE mutations in the dataset
    
    Args:
        df: DataFrame containing the genotype data
        hfe_columns: List of columns containing HFE genotype data
        hwe_tests: List of tests to perform ('chi_square', 'exact', 'logistic', 'bayesian')
    """
    if hwe_tests is None:
        hwe_tests = ['chi_square']  # Default to chi-square test only
        
    results = []
    results.append("HARDY-WEINBERG EQUILIBRIUM ANALYSIS")
    results.append(f"Tests selected: {', '.join(hwe_tests)}")
    
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
        hwe_results = calculate_hwe(df, column, column_name, hwe_tests)
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
        
        results.append("\nStatistical Tests:")
        
        # Report Chi-square results if requested
        if 'chi_square' in hwe_tests:
            if np.isnan(hwe_results['Chi-square']):
                results.append("Chi-square test could not be performed (insufficient data)")
            else:
                results.append(f"Chi-square statistic: {hwe_results['Chi-square']:.4f}")
                results.append(f"Chi-square p-value: {hwe_results['p-value (chi-square)']:.4f}")
                results.append(f"Degrees of freedom: {hwe_results['Degrees of freedom']}")
                chi_interp = "in HWE" if hwe_results['p-value (chi-square)'] > 0.05 else "not in HWE"
                results.append(f"Chi-square interpretation: Population is {chi_interp} (α=0.05)")
        
        # Report Exact test results if requested
        if 'exact' in hwe_tests:
            if np.isnan(hwe_results['p-value (exact)']):
                results.append("Exact test could not be performed (insufficient data)")
            else:
                results.append(f"Exact test p-value: {hwe_results['p-value (exact)']:.4f}")
                exact_interp = "in HWE" if hwe_results['p-value (exact)'] > 0.05 else "not in HWE"
                results.append(f"Exact test interpretation: Population is {exact_interp} (α=0.05)")
        
        # Report Logistic regression results if requested
        if 'logistic' in hwe_tests:
            if np.isnan(hwe_results['p-value (logistic)']):
                results.append("Logistic regression could not be performed (insufficient data or no diagnosis information)")
            else:
                results.append(f"Logistic regression p-value: {hwe_results['p-value (logistic)']:.4f}")
                logistic_interp = "significant" if hwe_results['p-value (logistic)'] < 0.05 else "not significant"
                results.append(f"Logistic regression interpretation: Association with liver disease is {logistic_interp} (α=0.05)")
        
        # Report Bayesian analysis results if requested
        if 'bayesian' in hwe_tests:
            if np.isnan(hwe_results['Bayesian posterior']):
                results.append("Bayesian analysis could not be performed (insufficient data)")
            else:
                results.append(f"Bayesian posterior probability of HWE: {hwe_results['Bayesian posterior']:.4f}")
                bayesian_interp = "supports HWE" if hwe_results['Bayesian posterior'] > 0.5 else "does not support HWE"
                results.append(f"Bayesian interpretation: Evidence {bayesian_interp}")
        
        results.append(f"\nOverall conclusion: {hwe_results['In Hardy-Weinberg equilibrium']}")
    
    return df, results, hwe_results_list 