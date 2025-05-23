#!/usr/bin/env python3
# UI module for HFE mutation analysis Shiny app

from shiny import ui
import shinyswatch

def create_ui():
    """Create the UI for the Shiny application"""
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
                ui.input_checkbox("remove_second_col", "Remove second column (usually empty)", True),
                ui.hr(),
                
                ui.h3("Age Filtering"),
                ui.input_checkbox("filter_by_age", "Filter by age", False),
                ui.input_slider("min_age", "Minimum Age", min=0, max=100, value=0),
                ui.input_slider("max_age", "Maximum Age", min=0, max=250, value=100),
                ui.hr(),
                
                ui.input_action_button("btn_clean", "Clean Dataset", class_="btn-primary"),
                ui.output_ui("formatted_download_button"),
                ui.hr(),
                
                ui.h3("Analysis Options"),
                ui.input_checkbox_group(
                    "analysis_type",
                    "Analysis Type",
                    {
                        "basic": "Basic Dataset Analysis",
                        "hh_risk": "HH Risk Analysis",
                        "diagnosis": "Diagnosis Association Analysis",
                        "hardy_weinberg": "Hardy-Weinberg Equilibrium Analysis",
                        "diagnosis_trends": "Diagnosis Trends Analysis"
                    },
                    selected=["basic"]
                ),
                
                ui.h4("Hardy-Weinberg Tests"),
                ui.input_checkbox_group(
                    "hwe_tests",
                    "Select Statistical Tests:",
                    {
                        "chi_square": "Chi-Square Test (χ²)",
                        "exact": "Exact Test (Fisher)",
                        "bayesian": "Bayesian Analysis"
                    },
                    selected=["chi_square"]
                ),
                
                ui.div(
                    {"id": "diagnosis_trends_options", "style": "margin-top: 15px; padding: 10px; border: 1px solid #ccc; border-radius: 5px;"},
                    ui.h4("Diagnosis Trends Options"),
                    ui.input_text(
                        "date_column",
                        "Date Column for Trends Analysis",
                        value="",
                        placeholder="Enter column name exactly as it appears in the dataset"
                    ),
                    ui.input_switch(
                        "auto_detect_date",
                        "Auto-detect date column",
                        value=True
                    )
                ),
                
                ui.input_action_button("btn_analyze", "Run Analysis", class_="btn-success"),
                ui.hr(),
                
                ui.output_ui("data_download_button"),
                ui.output_ui("word_download_button"),
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
                    
                    ui.h3("Genotypes by Age"),
                    ui.output_ui("genotype_age_plots"),
                    
                    ui.h3("Genotypes by Gender"),
                    ui.output_ui("genotype_gender_plots"),
                    
                    ui.h3("Genotypes by Diagnosis and Gender"),
                    ui.output_ui("genotype_diagnosis_gender_plots"),
                    
                    ui.h3("Risk Distribution"),
                    ui.output_ui("risk_plot"),
                    ui.h3("Diagnosis Associations"),
                    ui.output_ui("diagnosis_plot"),
                    ui.h3("Liver Disease by Risk Category"),
                    ui.output_ui("liver_disease_plot"),
                    ui.h3("Hardy-Weinberg Equilibrium"),
                    ui.output_ui("hardy_weinberg_plots"),
                    ui.h3("Diagnosis Trends"),
                    ui.output_ui("diagnosis_trends_plots")
                ),
                ui.nav_panel("Diagnosis Trends",
                    ui.h3("Diagnosis Trends Analysis"),
                    ui.output_text_verbatim("diagnosis_trends_results"),
                    ui.h3("Diagnosis Code Validation"),
                    ui.output_text_verbatim("diagnosis_validation_results")
                )
            )
        ),
        title="HFE Mutation Analysis",
        theme=shinyswatch.theme.superhero
    )
    
    return app_ui 