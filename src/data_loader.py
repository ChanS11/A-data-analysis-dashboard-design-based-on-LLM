import pandas as pd
import os
import glob
from src.config import PR_PATH, SI_PATH

def load_company_data(company_path, company_name):
    """
    Load and organize data from a company directory into structured dictionaries.
    """
    data = {}
    
    # Load data types (Setting, Stop, Worktime)
    for data_type in ['Setting', 'Stop', 'Worktime']:
        # Use glob to find files matching the pattern
        matching_files = glob.glob(os.path.join(company_path, f'*_{data_type}.csv'))
        
        if not matching_files:
            print(f"Warning: {data_type} file not found for {company_name}")
            continue
        
        file_path = matching_files[0]  # Use the first match
        try:
            df = pd.read_csv(file_path, low_memory=False, encoding='ISO-8859-1')
            df['source_file'] = os.path.basename(file_path)
            data[data_type.lower()] = df
        except Exception as e:
            print(f"Error loading {data_type} file for {company_name}: {e}")
    
    return data

def load_all_data():
    # Load data for both companies
    pr_data = load_company_data(PR_PATH, "XXX")
    si_data = load_company_data(SI_PATH, "XXX")

    # Create structured data dictionary
    data_by_company = {
        'pr': pr_data,
        'si': si_data
    }
    
    return data_by_company
