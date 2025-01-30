import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pyarrow.parquet as pq
import pyarrow as pa

# Determine project root directory
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

class PokerVAEDataset(Dataset):
    """
    PyTorch Dataset for Poker VAE preprocessing
    Handles large Parquet files with efficient memory loading
    """
    def __init__(
        self, 
        file_path, 
        features=None, 
        transform=None, 
        chunk_size=100000,
        device='cpu'
    ):
        """
        Initialize dataset with Parquet file
        
        Args:
            file_path (str): Path to Parquet file
            features (list, optional): List of features to use
            transform (callable, optional): Optional transform to be applied
            chunk_size (int): Number of rows to load at a time
            device (str): Device to load tensors on
        """
        # If a relative path is provided, join with data directory
        if not os.path.isabs(file_path):
            file_path = os.path.join(DATA_DIR, file_path)
        
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.device = torch.device(device)
        
        # Default features if not specified
        if features is None:
            features = [
                'hand_strength', 'pot_size_bb', 'stack_bb', 
                'position', 'action_fold_prob', 'action_call_prob', 
                'action_raise_prob', 'bet_amount_bb', 'pot_odds', 
                'stack_to_pot'
            ]
        self.features = features
        
        # Read entire dataset into memory (for smaller datasets)
        self.data = pd.read_parquet(file_path, columns=self.features)
        self.total_rows = len(self.data)
        
        # Preprocessing transforms
        self.transform = transform or self._default_transform()
        
        # Fit transformers on entire dataset
        self._fit_transformers()
    
    def _default_transform(self):
        """
        Create default preprocessing transform
        
        Returns:
            dict: Preprocessing transformers
        """
        return {
            'imputer': SimpleImputer(strategy='median'),
            'scaler': StandardScaler()
        }
    
    def _fit_transformers(self):
        """
        Fit preprocessing transformers on entire dataset
        """
        # Impute missing values
        X = self.data[self.features].values
        self.transform['imputer'].fit(X)
        X_imputed = self.transform['imputer'].transform(X)
        
        # Scale features
        self.transform['scaler'].fit(X_imputed)
    
    def __len__(self):
        """
        Get total number of samples
        
        Returns:
            int: Total number of samples
        """
        return self.total_rows
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset
        
        Args:
            idx (int): Index of sample
        
        Returns:
            torch.Tensor: Preprocessed sample
        """
        # Get sample
        sample = self.data.iloc[idx][self.features].values
        
        # Preprocess sample
        sample_imputed = self.transform['imputer'].transform(sample.reshape(1, -1)).flatten()
        sample_scaled = self.transform['scaler'].transform(sample_imputed.reshape(1, -1)).flatten()
        
        return torch.tensor(sample_scaled, dtype=torch.float32)

def create_vae_dataloader(
    file_path, 
    batch_size=256, 
    shuffle=True, 
    device='cpu'
):
    """
    Create DataLoader for VAE training
    
    Args:
        file_path (str): Path to Parquet file
        batch_size (int): Number of samples per batch
        shuffle (bool): Whether to shuffle data
        device (str): Device to load tensors on
    
    Returns:
        torch.utils.data.DataLoader: Prepared DataLoader
    """
    # Create dataset
    dataset = PokerVAEDataset(file_path, device=device)
    
    # Create DataLoader
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle
    )

def main():
    """
    Main function to demonstrate dataset usage
    """
    # Example usage
    dataloader = create_vae_dataloader('poker_game_metrics_full.parquet')
    
    # Iterate through first batch
    for batch in dataloader:
        print("Batch shape:", batch.shape)
        break

if __name__ == "__main__":
    main()
