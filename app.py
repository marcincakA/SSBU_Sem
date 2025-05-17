#!/usr/bin/env python3
# Main application file for HFE mutation analysis

import pandas as pd
import numpy as np
import os
import tempfile
from openpyxl.styles import Font
from pathlib import Path

# Import Shiny for Python
from shiny import App, ui, render, reactive

# Import modules
from modules.data_processing import clean_dataset, analyze_age_column, find_hfe_columns
from modules.analysis import (
    analyze_dataset, analyze_hh_risk, analyze_diagnosis_associations, 
    analyze_hardy_weinberg
)
from modules.visualization import (
    generate_genotype_distribution_plot, generate_risk_distribution_plot,
    generate_diagnosis_association_plot, generate_liver_disease_plot,
    generate_hardy_weinberg_plots, generate_genotype_by_age_plot,
    generate_genotype_by_gender_plot, generate_genotype_diagnosis_gender_plot
)
from modules.ui import create_ui

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
    plots_hardy_weinberg = reactive.Value([])
    
    # Reactive values for the visualizations
    plots_age = reactive.Value([])
    plots_gender = reactive.Value([])
    plots_diagnosis_gender = reactive.Value([])
    
    # Diagnostic reactive values
    diagnostics_age = reactive.Value([])
    diagnostics_gender = reactive.Value([])
    diagnostics_diagnosis_gender = reactive.Value([])
    
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
            hfe_columns = find_hfe_columns(df)
            
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
                remove_blank_hfe=input.clean_hfe(),
                remove_second_column=input.remove_second_col(),
                min_age=input.min_age(),
                max_age=input.max_age(),
                filter_by_age=input.filter_by_age()
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
            hfe_columns = find_hfe_columns(df)
            
            if not hfe_columns:
                ui.notification_show("No HFE mutation columns found in the dataset", type="warning")
                return
            
            # Add information about dataset
            results.append("Dataset columns:")
            for i, col in enumerate(df.columns):
                results.append(f"  {i}: {col}")
            
            # Run selected analysis
            analysis_type = input.analysis_type()
            
            if analysis_type in ["basic", "all"]:
                basic_results = analyze_dataset(df)
                results.extend(basic_results)
                
                # Add specific age column analysis
                age_results = analyze_age_column(df)
                results.extend(age_results)
            
            if analysis_type in ["hh_risk", "all"]:
                df, hh_results = analyze_hh_risk(df, hfe_columns)
                results.extend(hh_results)
            
            if analysis_type in ["diagnosis", "all"]:
                df, diag_results = analyze_diagnosis_associations(df, hfe_columns)
                results.extend(diag_results)
            
            if analysis_type in ["hardy_weinberg", "all"]:
                # Get selected Hardy-Weinberg tests
                selected_hwe_tests = input.hwe_tests() if input.hwe_tests() else ["chi_square"]
                
                # Run Hardy-Weinberg analysis with selected tests
                df, hwe_results, hwe_results_list = analyze_hardy_weinberg(df, hfe_columns, selected_hwe_tests)
                results.extend(hwe_results)
                plots_hardy_weinberg.set(generate_hardy_weinberg_plots(hwe_results_list))
            
            # Generate plots
            plots_genotype.set(generate_genotype_distribution_plot(df, hfe_columns))
            
            # Generate new plots for age, gender, and diagnosis-gender relationships
            age_plots, age_diagnostics = generate_genotype_by_age_plot(df, hfe_columns)
            plots_age.set(age_plots)
            diagnostics_age.set(age_diagnostics)
            
            gender_plots, gender_diagnostics = generate_genotype_by_gender_plot(df, hfe_columns)
            plots_gender.set(gender_plots)
            diagnostics_gender.set(gender_diagnostics)
            
            diag_gender_plots, diag_gender_diagnostics = generate_genotype_diagnosis_gender_plot(df, hfe_columns)
            plots_diagnosis_gender.set(diag_gender_plots)
            diagnostics_diagnosis_gender.set(diag_gender_diagnostics)
            
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
            import traceback
            error_traceback = traceback.format_exc()
            analysis_results_text.set([f"Error during analysis: {str(e)}", "", "Traceback:", error_traceback])
            analysis_run.set(True)
    
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
    def genotype_age_plots():
        if not plots_age() or len(plots_age()) == 0:
            diagnostics = diagnostics_age()
            if diagnostics:
                # Show the most relevant diagnostic information
                key_messages = [msg for msg in diagnostics if any(term in msg for term in 
                              ['column', 'found', 'Sample', 'Age range', 'Error'])]
                
                # Limit to 5 most important messages
                if len(key_messages) > 5:
                    key_messages = key_messages[:5]
                    
                return ui.tags.div(
                    ui.p("No age plots available. Diagnostic information:"),
                    ui.tags.ul(*[ui.tags.li(msg) for msg in key_messages]),
                    ui.p("See Analysis Results tab for complete diagnostics.")
                )
            else:
                return ui.p("No age plots available. This could be because age data is missing or there aren't enough samples to create meaningful visualizations.")
        
        plot_htmls = []
        for name, img_str in plots_age():
            plot_htmls.append(ui.tags.div(
                ui.tags.h4(name),
                ui.tags.img(src=f"data:image/png;base64,{img_str}", width="100%", style="max-width: 800px;"),
                ui.br(), ui.br()
            ))
        
        return ui.tags.div(*plot_htmls)
    
    @output
    @render.ui
    def genotype_gender_plots():
        if not plots_gender() or len(plots_gender()) == 0:
            diagnostics = diagnostics_gender()
            if diagnostics:
                # Show the most relevant diagnostic information
                key_messages = [msg for msg in diagnostics if any(term in msg for term in 
                              ['column', 'found', 'Sample', 'distribution', 'Error'])]
                
                # Limit to 5 most important messages
                if len(key_messages) > 5:
                    key_messages = key_messages[:5]
                    
                return ui.tags.div(
                    ui.p("No gender plots available. Diagnostic information:"),
                    ui.tags.ul(*[ui.tags.li(msg) for msg in key_messages]),
                    ui.p("See Analysis Results tab for complete diagnostics.")
                )
            else:
                return ui.p("No gender plots available. This could be because gender data is missing or there aren't enough samples to create meaningful visualizations.")
        
        plot_htmls = []
        for name, img_str in plots_gender():
            plot_htmls.append(ui.tags.div(
                ui.tags.h4(name),
                ui.tags.img(src=f"data:image/png;base64,{img_str}", width="100%", style="max-width: 800px;"),
                ui.br(), ui.br()
            ))
        
        return ui.tags.div(*plot_htmls)
    
    @output
    @render.ui
    def genotype_diagnosis_gender_plots():
        if not plots_diagnosis_gender() or len(plots_diagnosis_gender()) == 0:
            diagnostics = diagnostics_diagnosis_gender()
            if diagnostics:
                # Show the most relevant diagnostic information
                key_messages = [msg for msg in diagnostics if any(term in msg for term in 
                              ['column', 'found', 'Sample', 'distribution', 'Error'])]
                
                # Limit to 5 most important messages
                if len(key_messages) > 5:
                    key_messages = key_messages[:5]
                    
                return ui.tags.div(
                    ui.p("No diagnosis-gender plots available. Diagnostic information:"),
                    ui.tags.ul(*[ui.tags.li(msg) for msg in key_messages]),
                    ui.p("See Analysis Results tab for complete diagnostics.")
                )
            else:
                return ui.p("No diagnosis-gender plots available. This could be because diagnosis or gender data is missing or there aren't enough samples to create meaningful visualizations.")
        
        plot_htmls = []
        for name, img_str in plots_diagnosis_gender():
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
    
    @output
    @render.ui
    def hardy_weinberg_plots():
        if not plots_hardy_weinberg():
            return ui.p("No Hardy-Weinberg plots available. Run Hardy-Weinberg analysis first.")
        
        plot_htmls = []
        for mutation, img_str in plots_hardy_weinberg():
            plot_htmls.append(ui.tags.div(
                ui.tags.h4(mutation),
                ui.tags.img(src=f"data:image/png;base64,{img_str}", width="100%", style="max-width: 800px;"),
                ui.br(), ui.br()
            ))
        
        return ui.tags.div(*plot_htmls)
    
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
        temp_path = os.path.join(tempfile.gettempdir(), "processed_dataset.xlsx")
        
        # Get data and format ID column
        df = data().copy()
        
        # Find ID column (likely first column)
        if 'id' in df.columns:
            id_col = 'id'
        elif any(col.lower() == 'id' for col in df.columns):
            id_col = next(col for col in df.columns if col.lower() == 'id')
        else:
            # Assuming the first column is the ID
            id_col = df.columns[0]
            
        # Fix ID formatting if needed
        if df[id_col].dtype in [np.int64, np.float64]:
            df[id_col] = df[id_col].astype('Int64')
            df[id_col] = df[id_col].astype(str)
            df[id_col] = df[id_col].replace(['nan', '<NA>'], None)
        
        # Apply zfill to preserve leading zeros
        mask = df[id_col].notna() & (df[id_col] != '')
        if any(mask):
            # Check for ID format
            sample_ids = df[id_col].dropna().head(10).tolist()
            id_lengths = [len(str(id_val)) for id_val in sample_ids if str(id_val) != 'nan' and str(id_val) != '']
            zfill_length = 10 if not id_lengths or max(id_lengths) > 9 else 9
            # Apply consistent formatting
            df.loc[mask, id_col] = df.loc[mask, id_col].str.zfill(zfill_length)
        
        # Write to Excel with formatting for ID column
        with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
            
            # Apply text format to ID column to preserve leading zeros
            worksheet = writer.sheets['Sheet1']
            id_col_index = list(df.columns).index(id_col) + 1  # +1 because Excel is 1-indexed
            for cell in worksheet.iter_cols(min_col=id_col_index, max_col=id_col_index, min_row=2):
                for x in cell:
                    x.number_format = '@'
        
        # Return the path as a string
        return temp_path
    
    @output
    @render.download(filename="formatted_dataset.xlsx")
    def download_formatted():
        # If no data, return None
        if data() is None:
            return None
            
        # Create temporary file name
        temp_path = os.path.join(tempfile.gettempdir(), "formatted_dataset.xlsx")
        
        # Get data and format ID column
        df = data().copy()
        
        # Find ID column (likely first column)
        if 'id' in df.columns:
            id_col = 'id'
        elif any(col.lower() == 'id' for col in df.columns):
            id_col = next(col for col in df.columns if col.lower() == 'id')
        else:
            # Assuming the first column is the ID
            id_col = df.columns[0]
            
        # Fix ID formatting - convert to string
        if df[id_col].dtype in [np.int64, np.float64]:
            # For numeric columns, only convert non-NaN values to string
            df[id_col] = df[id_col].astype('Int64')  # nullable integer type
            df[id_col] = df[id_col].astype(str)
            # Replace 'nan' or '<NA>' strings with None
            df[id_col] = df[id_col].replace(['nan', '<NA>'], None)
        
        # Apply zfill to non-null values to preserve leading zeros
        mask = df[id_col].notna() & (df[id_col] != '')
        if any(mask):
            # Determine ID format based on sample
            sample_ids = df[id_col].dropna().head(10).tolist()
            id_lengths = [len(str(id_val)) for id_val in sample_ids if str(id_val) != 'nan' and str(id_val) != '']
            zfill_length = 10 if not id_lengths or max(id_lengths) > 9 else 9
            # Apply zfill for consistent length
            df.loc[mask, id_col] = df.loc[mask, id_col].str.zfill(zfill_length)
        
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
app = App(create_ui(), server)

# Run the app if this script is executed directly
if __name__ == "__main__":
    print("Starting app on http://127.0.0.1:8095")
    app.run(host="127.0.0.1", port=8095) 