# HFE Mutation Analysis

A Python Shiny application for analyzing HFE gene mutations in patients and their relationship to various health conditions, particularly hereditary hemochromatosis.

## Features

- Data cleaning and preprocessing
- Basic dataset analysis 
- HFE mutation distribution visualization
- Hardy-Weinberg equilibrium analysis
- Diagnosis association analysis
- Demographic analysis (age, gender)
- Risk categorization
- Excel export with proper formatting

## Project Structure

The application is organized into the following components:

- `app.py` - Main application entry point
- `modules/` - Core application modules:
  - `data_processing.py` - Dataset cleaning and preparation 
  - `analysis.py` - Statistical analysis functions
  - `visualization.py` - Plot generation functions
  - `ui.py` - Shiny UI components

## Requirements

- Python 3.7+
- Required packages: see `requirements.txt`

## Installation

1. Clone the repository
2. Create a virtual environment (recommended)
3. Install dependencies:

```
pip install -r requirements.txt
```

## Running the Application

To start the application:

```
python app.py
```

The application will be available at http://127.0.0.1:8095 in your web browser.

## Usage

1. Upload an Excel file containing patient data with HFE mutation information
2. Configure cleaning options if needed
3. Clean the dataset
4. Select analysis type and run analysis
5. View results and visualizations
6. Download processed data if needed

## License

This project is licensed under the MIT License - see the LICENSE file for details. 