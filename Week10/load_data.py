import pandas as pd
import os

def load_data():
    """
    Load all competition data files into pandas DataFrames.
    Returns a dictionary containing all DataFrames.
    """
    # Define the data directory (go up one level to find the Data directory)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data')
    
    # Dictionary to store all dataframes
    dfs = {}
    
    # List of files to load
    files_to_load = {
        'train': 'train.csv.zip',
        'test': 'test.csv.zip',
        'sample_submission': 'sample_submission.csv.zip',
        'product_descriptions': 'product_descriptions.csv.zip',
        'attributes': 'attributes.csv.zip'
    }
    
    # Load each file
    for name, file in files_to_load.items():
        file_path = os.path.join(data_dir, file)
        print(f"Loading {name} data...")
        
        try:
            # Read the zip file with latin1 encoding (can handle any byte sequence)
            dfs[name] = pd.read_csv(file_path, compression='zip', encoding='latin1')
            print(f"Successfully loaded {name}")
            print(f"Shape: {dfs[name].shape}")
            print(f"Columns: {list(dfs[name].columns)}\n")
        except Exception as e:
            print(f"Error loading {name}: {str(e)}")
            continue
    
    return dfs 