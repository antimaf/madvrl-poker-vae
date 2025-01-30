import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
import pandas as pd
import numpy as np
import json

def analyze_parquet(file_path):
    """
    Analyze a Parquet file and provide detailed information
    """
    print(f"\n--- Analysis of {file_path} ---")
    
    # Read Parquet file
    df = pd.read_parquet(file_path)
    
    # Basic information
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print("\nColumns:")
    for col in df.columns:
        print(f"- {col}: {df[col].dtype}")
    
    # Descriptive statistics
    print("\nDescriptive Statistics:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_cols].describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

def analyze_npz(file_path):
    """
    Analyze an NPZ file and provide detailed information
    """
    print(f"\n--- Analysis of {file_path} ---")
    
    # Load NPZ file
    data = np.load(file_path)
    
    # Print keys and basic information
    print("NPZ File Contents:")
    for key in data.files:
        arr = data[key]
        print(f"- {key}: {arr.shape}, dtype: {arr.dtype}")
        
        # Basic statistics if numeric
        if np.issubdtype(arr.dtype, np.number):
            print(f"  Stats: min={arr.min()}, max={arr.max()}, mean={arr.mean()}")

def main():
    # Parquet files
    parquet_files = [
        'poker_game_metrics.parquet',
        'poker_game_metrics_full.parquet',
        'poker_game_metrics_vae.parquet'
    ]
    
    # NPZ files
    npz_files = [
        'processed_poker_data.npz'
    ]
    
    # Analyze Parquet files
    for file in parquet_files:
        try:
            analyze_parquet(file)
        except Exception as e:
            print(f"Error analyzing {file}: {e}")
    
    # Analyze NPZ files
    for file in npz_files:
        try:
            analyze_npz(file)
        except Exception as e:
            print(f"Error analyzing {file}: {e}")

if __name__ == "__main__":
    main()
