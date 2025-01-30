import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pyarrow.parquet as pq
import pyarrow as pa

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
        
        # Apply transformations
        sample = self._preprocess_sample(sample)
        
        return sample
    
    def _preprocess_sample(self, sample):
        """
        Preprocess a single sample
        
        Args:
            sample (np.ndarray): Raw sample
        
        Returns:
            torch.Tensor: Preprocessed sample
        """
        # Impute missing values
        sample = self.transform['imputer'].transform(
            sample.reshape(1, -1)
        ).flatten()
        
        # Scale features
        sample = self.transform['scaler'].transform(
            sample.reshape(1, -1)
        ).flatten()
        
        # Convert to tensor
        return torch.tensor(sample, dtype=torch.float32).to(self.device)
    
    def get_feature_statistics(self):
        """
        Compute feature-wise statistics
        
        Returns:
            dict: Feature statistics
        """
        try:
            stats = {}
            for feature in self.features:
                stats[feature] = {
                    'mean': self.data[feature].mean(),
                    'std': self.data[feature].std(),
                    'min': self.data[feature].min(),
                    'max': self.data[feature].max()
                }
            
            return stats
        except Exception as e:
            print(f"Error computing feature statistics: {e}")
            raise

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
    try:
        dataset = PokerVAEDataset(
            file_path, 
            device=device
        )
        
        # Print feature statistics
        stats = dataset.get_feature_statistics()
        print("Feature Statistics:")
        for feature, stat in stats.items():
            print(f"{feature}:")
            for k, v in stat.items():
                print(f"  {k}: {v}")
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=0  # Disable multiprocessing
        )
        
        return dataloader
    except Exception as e:
        print(f"Error creating DataLoader: {e}")
        raise

def main():
    """
    Main function to demonstrate dataset usage
    """
    try:
        # Path to full dataset
        file_path = 'poker_game_metrics_full.parquet'
        
        # Create DataLoader
        dataloader = create_vae_dataloader(file_path)
        
        # Demonstrate iteration
        print("\nDataLoader Iteration:")
        for batch_idx, batch in enumerate(dataloader):
            print(f"Batch {batch_idx}:")
            print("  Shape:", batch.shape)
            print("  First sample:", batch[0])
            
            # Stop after first few batches
            if batch_idx >= 2:
                break
    except Exception as e:
        print(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()
