#!/usr/bin/env python3
# Shiny GUI application for HFE mutation analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from datetime import datetime
import io
import base64
from scipy.stats import chi2_contingency

# Import Shiny for Python
from shiny import App, ui, render, reactive
import shinyswatch

# Define reusable functions from the existing scripts
def clean_dataset(df, remove_blank_validovany=True, remove_blank_diagnoza=True, remove_blank_hfe=True):
    """
    Clean the dataset by fixing ID format and removing rows with blank values in specified columns
    """
    results = []
    results.append("Starting dataset cleaning...")
    results.append(f"Initial rows: {len(df)}")
    
    # Find ID column (likely first column)
    id_col = df.columns[0] if 'id' not in df.columns.str.lower() else df.columns[df.columns.str.lower() == 'id'][0]
    results.append(f"Using '{id_col}' as the ID column")
    
    # Fix ID formatting (basic operations)
    if df[id_col].dtype in [np.int64, np.float64]:
        df[id_col] = df[id_col].astype('Int64')
        df[id_col] = df[id_col].astype(str)
        df[id_col] = df[id_col].replace(['nan', '<NA>'], None)
        results.append("Converted ID column to string format")
    
    # Find validovany vysledok column
    validovany_col = None
    for col in df.columns:
        if 'validovany' in str(col).lower() and 'vysledok' in str(col).lower():
            validovany_col = col
            break
    
    # Remove rows with blank validovany vysledok
    if validovany_col and remove_blank_validovany:
        initial_rows = len(df)
        df = df[df[validovany_col].notna() & (df[validovany_col] != '')]
        removed_rows = initial_rows - len(df)
        results.append(f"Removed {removed_rows} rows with blank values in '{validovany_col}' column")
    
    # Find diagnoza column
    diagnoza_col = None
    for col in df.columns:
        if 'diagnoza' in str(col).lower() or 'mkch' in str(col).lower():
            diagnoza_col = col
            break
    
    # Remove rows with blank diagnoza
    if diagnoza_col and remove_blank_diagnoza:
        initial_rows = len(df)
        df = df[df[diagnoza_col].notna() & (df[diagnoza_col] != '')]
        removed_rows = initial_rows - len(df)
        results.append(f"Removed {removed_rows} rows with blank values in '{diagnoza_col}' column")
    
    # Find HFE columns
    hfe_columns = []
    for col in df.columns:
        if 'hfe' in str(col).lower():
            hfe_columns.append(col)
    
    # Remove rows with blank HFE values
    if hfe_columns and remove_blank_hfe:
        initial_rows = len(df)
        for hfe_col in hfe_columns:
            df = df[df[hfe_col].notna() & (df[hfe_col] != '')]
        removed_rows = initial_rows - len(df)
        results.append(f"Removed {removed_rows} rows with blank values in HFE columns")
    
    results.append(f"Final rows after cleaning: {len(df)}")
    return df, results

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
    k760_in_mutation = df[df['Any_HFE_Mutation'] == 'Mutation Present']['Has_K760'].mean() * 100
    k760_in_no_mutation = df[df['Any_HFE_Mutation'] == 'No Mutation']['Has_K760'].mean() * 100
    
    k759_in_mutation = df[df['Any_HFE_Mutation'] == 'Mutation Present']['Has_K759'].mean() * 100
    k759_in_no_mutation = df[df['Any_HFE_Mutation'] == 'No Mutation']['Has_K759'].mean() * 100
    
    results.append(f"\nK76.0 (Fatty liver) prevalence:")
    results.append(f"- In patients with HFE mutations: {k760_in_mutation:.2f}%")
    results.append(f"- In patients without HFE mutations: {k760_in_no_mutation:.2f}%")
    
    results.append(f"\nK75.9 (Inflammatory liver disease) prevalence:")
    results.append(f"- In patients with HFE mutations: {k759_in_mutation:.2f}%")
    results.append(f"- In patients without HFE mutations: {k759_in_no_mutation:.2f}%")
    
    # Create a 2x2 contingency table for each specific disease
    k760_table = pd.crosstab(df['Any_HFE_Mutation'], df['Has_K760'])
    k759_table = pd.crosstab(df['Any_HFE_Mutation'], df['Has_K759'])
    
    # Perform chi-square tests on the 2x2 tables if possible
    try:
        k760_chi2, k760_p, k760_dof, k760_expected = chi2_contingency(k760_table)
        results.append(f"\nChi-Square Test for K76.0: chi2={k760_chi2:.2f}, p={k760_p:.4f}")
        if k760_p < 0.05:
            results.append("There is a significant association between HFE mutations and K76.0 (Fatty liver).")
        else:
            results.append("No significant association found between HFE mutations and K76.0 (Fatty liver).")
    except:
        results.append("Could not perform chi-square test for K76.0 - may have insufficient data.")
    
    try:
        k759_chi2, k759_p, k759_dof, k759_expected = chi2_contingency(k759_table)
        results.append(f"\nChi-Square Test for K75.9: chi2={k759_chi2:.2f}, p={k759_p:.4f}")
        if k759_p < 0.05:
            results.append("There is a significant association between HFE mutations and K75.9 (Inflammatory liver disease).")
        else:
            results.append("No significant association found between HFE mutations and K75.9 (Inflammatory liver disease).")
    except:
        results.append("Could not perform chi-square test for K75.9 - may have insufficient data.")
    
    return df, results

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

# Define the Shiny UI
app_ui = ui.page_fluid(
    ui.h2("HFE Mutation Analysis"),
    
    ui.layout_sidebar(
        ui.sidebar(
            ui.h3("Upload Dataset"),
            ui.input_file("file1", "Choose Excel File", accept=[".xls", ".xlsx"]),
            ui.hr(),
            
            ui.h3("Data Cleaning Options"),
            ui.input_checkbox("clean_validovany", "Remove rows with blank validovany vysledok", True),
            ui.input_checkbox("clean_diagnoza", "Remove rows with blank diagnoza", True),
            ui.input_checkbox("clean_hfe", "Remove rows with blank HFE values", True),
            ui.input_action_button("btn_clean", "Clean Dataset", class_="btn-primary"),
            ui.output_ui("formatted_download_button"),
            ui.hr(),
            
            ui.h3("Analysis Options"),
            ui.input_radio_buttons(
                "analysis_type",
                "Analysis Type",
                {
                    "basic": "Basic Dataset Analysis",
                    "hh_risk": "HH Risk Analysis",
                    "diagnosis": "Diagnosis Association Analysis",
                    "all": "Complete Analysis"
                },
                selected="all"
            ),
            ui.input_action_button("btn_analyze", "Run Analysis", class_="btn-success"),
            ui.hr(),
            
            ui.output_ui("data_download_button"),
        ),
        
        ui.navset_tab(
            ui.nav_panel("Dataset Overview",
                ui.h3("Dataset Information"),
                ui.output_text_verbatim("dataset_info"),
                ui.h3("Data Preview"),
                ui.output_data_frame("data_preview")
            ),
            ui.nav_panel("Analysis Results",
                ui.h3("Analysis Output"),
                ui.output_text_verbatim("analysis_results")
            ),
            ui.nav_panel("Visualizations",
                ui.h3("Genotype Distribution"),
                ui.output_ui("genotype_plots"),
                ui.h3("Risk Distribution"),
                ui.output_ui("risk_plot"),
                ui.h3("Diagnosis Associations"),
                ui.output_ui("diagnosis_plot"),
                ui.h3("Liver Disease by Risk Category"),
                ui.output_ui("liver_disease_plot")
            )
        )
    ),
    title="HFE Mutation Analysis",
    theme=shinyswatch.theme.superhero
)

# Define the Shiny server
def server(input, output, session):
    # Reactive value to store the dataset
    data = reactive.Value(None)
    data_cleaned = reactive.Value(False)
    analysis_run = reactive.Value(False)
    analysis_results_text = reactive.Value([])
    data_info = reactive.Value([])
    plots_genotype = reactive.Value([])
    plot_risk = reactive.Value(None)
    plot_diagnosis = reactive.Value(None)
    plot_liver = reactive.Value(None)
    
    @reactive.Effect
    @reactive.event(input.file1)
    def _():
        if input.file1() is None:
            return
        
        file_info = input.file1()
        file_path = file_info[0]["datapath"]
        
        try:
            df = pd.read_excel(file_path)
            data.set(df)
            data_cleaned.set(False)
            analysis_run.set(False)
            
            # Basic dataset info
            info = []
            info.append(f"File: {file_info[0]['name']}")
            info.append(f"Rows: {len(df)}")
            info.append(f"Columns: {len(df.columns)}")
            
            # HFE columns
            hfe_columns = []
            for col in df.columns:
                if 'hfe' in str(col).lower():
                    hfe_columns.append(col)
            
            if hfe_columns:
                info.append(f"Found {len(hfe_columns)} HFE mutation columns:")
                for col in hfe_columns:
                    info.append(f"- {col}")
            else:
                info.append("No HFE mutation columns found in the dataset")
            
            data_info.set(info)
        except Exception as e:
            ui.notification_show(f"Error loading file: {str(e)}", type="error", duration=None)
    
    @reactive.Effect
    @reactive.event(input.btn_clean)
    def _():
        if data() is None:
            ui.notification_show("Please upload a dataset first", type="warning")
            return
        
        try:
            df, results = clean_dataset(
                data(), 
                remove_blank_validovany=input.clean_validovany(),
                remove_blank_diagnoza=input.clean_diagnoza(),
                remove_blank_hfe=input.clean_hfe()
            )
            data.set(df)
            data_cleaned.set(True)
            analysis_run.set(False)
            data_info.set(results)
            ui.notification_show("Dataset cleaned successfully", type="success")
        except Exception as e:
            ui.notification_show(f"Error cleaning dataset: {str(e)}", type="error", duration=None)
    
    @reactive.Effect
    @reactive.event(input.btn_analyze)
    def _():
        if data() is None:
            ui.notification_show("Please upload a dataset first", type="warning")
            return
        
        if not data_cleaned():
            ui.notification_show("It's recommended to clean the dataset before analysis", type="info")
        
        try:
            df = data()
            results = []
            
            # Find HFE columns
            hfe_columns = []
            for col in df.columns:
                if 'hfe' in str(col).lower():
                    hfe_columns.append(col)
            
            if not hfe_columns:
                ui.notification_show("No HFE mutation columns found in the dataset", type="warning")
                return
            
            # Run selected analysis
            analysis_type = input.analysis_type()
            
            if analysis_type in ["basic", "all"]:
                basic_results = analyze_dataset(df)
                results.extend(basic_results)
            
            if analysis_type in ["hh_risk", "all"]:
                df, hh_results = analyze_hh_risk(df, hfe_columns)
                results.extend(hh_results)
            
            if analysis_type in ["diagnosis", "all"]:
                df, diag_results = analyze_diagnosis_associations(df, hfe_columns)
                results.extend(diag_results)
            
            # Generate plots
            plots_genotype.set(generate_genotype_distribution_plot(df, hfe_columns))
            
            if analysis_type in ["hh_risk", "all"]:
                plot_risk.set(generate_risk_distribution_plot(df))
            
            if analysis_type in ["diagnosis", "all"]:
                plot_diagnosis.set(generate_diagnosis_association_plot(df))
                plot_liver.set(generate_liver_disease_plot(df))
            
            # Update data with analysis results
            data.set(df)
            analysis_results_text.set(results)
            analysis_run.set(True)
            ui.notification_show("Analysis completed", type="success")
        
        except Exception as e:
            ui.notification_show(f"Error during analysis: {str(e)}", type="error", duration=None)
    
    @output
    @render.text
    def dataset_info():
        if data() is None:
            return "No dataset loaded"
        return "\n".join(data_info())
    
    @output
    @render.data_frame
    def data_preview():
        if data() is None:
            return None
        return render.DataGrid(data().head(10), width="100%")
    
    @output
    @render.text
    def analysis_results():
        if not analysis_run():
            return "Run analysis to see results"
        return "\n".join(analysis_results_text())
    
    @output
    @render.ui
    def genotype_plots():
        if not plots_genotype():
            return ui.p("No genotype plots available. Run analysis first.")
        
        plot_htmls = []
        for name, img_str in plots_genotype():
            plot_htmls.append(ui.tags.div(
                ui.tags.h4(name),
                ui.tags.img(src=f"data:image/png;base64,{img_str}", width="100%", style="max-width: 800px;"),
                ui.br(), ui.br()
            ))
        
        return ui.tags.div(*plot_htmls)
    
    @output
    @render.ui
    def risk_plot():
        if not plot_risk():
            return ui.p("No risk distribution plot available. Run HH Risk analysis first.")
        
        return ui.tags.div(
            ui.tags.img(src=f"data:image/png;base64,{plot_risk()}", width="100%", style="max-width: 800px;")
        )
    
    @output
    @render.ui
    def diagnosis_plot():
        if not plot_diagnosis():
            return ui.p("No diagnosis association plot available. Run Diagnosis Association analysis first.")
        
        return ui.tags.div(
            ui.tags.img(src=f"data:image/png;base64,{plot_diagnosis()}", width="100%", style="max-width: 800px;")
        )
    
    @output
    @render.ui
    def liver_disease_plot():
        if not plot_liver():
            return ui.p("No liver disease plot available. Run Diagnosis Association analysis first.")
        
        return ui.tags.div(
            ui.tags.img(src=f"data:image/png;base64,{plot_liver()}", width="100%", style="max-width: 800px;")
        )
    
    @reactive.Effect
    @reactive.event(data_cleaned)
    def _():
        # This effect will run whenever data_cleaned changes, but we'll manage
        # the button states through the disabled attribute in the UI
        pass
        
    @reactive.Effect
    @reactive.event(data)
    def _():
        # This effect will run whenever data changes, but we'll manage
        # the button states through the disabled attribute in the UI
        pass
        
    # Helper functions to determine button states
    @reactive.calc
    def formatted_button_disabled():
        return not (data() is not None and data_cleaned())
    
    @reactive.calc
    def download_button_disabled():
        return data() is None
    
    @output
    @render.ui
    def formatted_download_button():
        return ui.download_button(
            "download_formatted", 
            "Save Formatted Dataset", 
            class_="btn-outline-secondary mt-2", 
            disabled=formatted_button_disabled()
        )
    
    @output
    @render.ui
    def data_download_button():
        return ui.download_button(
            "download_data", 
            "Download Processed Data", 
            class_="btn-info", 
            disabled=download_button_disabled()
        )
    
    @output
    @render.download(filename="processed_dataset.xlsx")
    def download_data():
        # If no data, return None
        if data() is None:
            return None
            
        # Create temporary file name
        import os
        import tempfile
        temp_path = os.path.join(tempfile.gettempdir(), "processed_dataset.xlsx")
        
        # Save to Excel
        data().to_excel(temp_path, index=False)
        
        # Return the path as a string
        return temp_path
    
    @output
    @render.download(filename="formatted_dataset.xlsx")
    def download_formatted():
        # If no data, return None
        if data() is None:
            return None
            
        # Create temporary file name
        import os
        import tempfile
        from openpyxl.styles import Font
        
        temp_path = os.path.join(tempfile.gettempdir(), "formatted_dataset.xlsx")
        
        # Get data and format ID column
        df = data().copy()
        id_col = df.columns[0] if 'id' not in df.columns.str.lower() else df.columns[df.columns.str.lower() == 'id'][0]
        if id_col in df.columns and df[id_col].dtype != object:
            df[id_col] = df[id_col].astype(str)
        
        # Write to Excel with formatting
        with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
            
            # Apply formatting to Excel worksheet
            worksheet = writer.sheets['Sheet1']
            
            # Create a bold font object directly
            bold_font = Font(bold=True)
            
            # Format headers in bold
            for col_num, column_title in enumerate(df.columns, 1):
                cell = worksheet.cell(row=1, column=col_num)
                cell.font = bold_font
            
            # Set ID column to text format (to preserve leading zeros)
            id_col_index = list(df.columns).index(id_col) + 1  # +1 because Excel is 1-indexed
            for cell in worksheet.iter_cols(min_col=id_col_index, max_col=id_col_index, min_row=2):
                for x in cell:
                    x.number_format = '@'
            
            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = (max_length + 2)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        # Return the path as a string
        return temp_path

# Create the Shiny app
app = App(app_ui, server)

# Run the app
if __name__ == "__main__":
    print("Starting app on http://127.0.0.1:8095")
    # Use only supported parameters
    app.run(host="127.0.0.1", port=8095) 