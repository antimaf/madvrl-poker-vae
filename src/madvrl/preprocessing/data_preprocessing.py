import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class PokerTensorDataset(Dataset):
    """PyTorch Dataset for Poker Game Metrics"""
    def __init__(self, features, targets=None):
        """
        Initialize dataset with features and optional targets
        
        Args:
            features (torch.Tensor): Input features
            targets (torch.Tensor, optional): Target values
        """
        self.features = features
        self.targets = targets
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx]

class PokerDataPreprocessor:
    def __init__(self, dataset_path='poker_game_metrics.parquet'):
        """
        Initialize the data preprocessor
        
        Args:
            dataset_path (str): Path to the Parquet file
        """
        self.dataset_path = dataset_path
        self.df = None
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
    
    def load_data(self):
        """
        Load the Parquet dataset
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        self.df = pd.read_parquet(self.dataset_path)
        print(f"Dataset loaded. Shape: {self.df.shape}")
        return self.df
    
    def preprocess_features(self, target_column='hand_strength'):
        """
        Preprocess features for VAE
        
        Args:
            target_column (str): Column to use as target variable
        
        Returns:
            tuple: Processed features and optional target
        """
        if self.df is None:
            self.load_data()
        
        # Select relevant features for VAE
        feature_columns = [
            'bet_amount', 'pot_size', 'stack_to_pot_ratio', 
            'big_blinds_remaining', 'hand_strength', 
            'position_relative_to_button'
        ]
        
        # Create DataFrame with selected features
        df_features = self.df[feature_columns].copy()
        
        # Handle missing values
        df_features = pd.DataFrame(
            self.imputer.fit_transform(df_features), 
            columns=df_features.columns
        )
        
        # Separate target if specified
        if target_column in df_features.columns:
            y = df_features[target_column].values
            X = df_features.drop(columns=[target_column])
        else:
            X = df_features
            y = None
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32) if y is not None else None
        
        self.X = X_tensor
        self.y = y_tensor
        
        return X_tensor, y_tensor
    
    def create_dataloader(
        self, 
        batch_size=64, 
        shuffle=True, 
        target_column='hand_strength'
    ):
        """
        Create PyTorch DataLoader
        
        Args:
            batch_size (int): Number of samples per batch
            shuffle (bool): Whether to shuffle the data
            target_column (str): Column to use as target
        
        Returns:
            torch.utils.data.DataLoader: Prepared DataLoader
        """
        # Preprocess features if not already done
        if self.X is None:
            self.preprocess_features(target_column)
        
        # Create dataset
        if self.y is not None:
            dataset = PokerTensorDataset(self.X, self.y)
        else:
            dataset = PokerTensorDataset(self.X)
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle
        )
        
        return dataloader
    
    def save_preprocessed_data(self, output_path='preprocessed_poker_data.pt'):
        """
        Save preprocessed data as PyTorch file
        
        Args:
            output_path (str): Path to save processed data
        """
        if self.X is None:
            self.preprocess_features()
        
        # Prepare data dictionary
        preprocessed_data = {
            'features': self.X,
            'targets': self.y,
            'scaler_mean': torch.tensor(self.scaler.mean_),
            'scaler_scale': torch.tensor(self.scaler.scale_)
        }
        
        # Save using torch
        torch.save(preprocessed_data, output_path)
        print(f"Preprocessed data saved to {output_path}")

def main():
    # Example usage
    preprocessor = PokerDataPreprocessor()
    
    # Preprocess features
    X, y = preprocessor.preprocess_features()
    print("Processed Features Shape:", X.shape)
    print("Targets Shape:", y.shape if y is not None else "No targets")
    
    # Create DataLoader
    dataloader = preprocessor.create_dataloader()
    
    # Save preprocessed data
    preprocessor.save_preprocessed_data()
    
    # Demonstrate batch iteration
    for batch in dataloader:
        if isinstance(batch, tuple):
            features, targets = batch
            print("Batch Features Shape:", features.shape)
            print("Batch Targets Shape:", targets.shape)
        else:
            print("Batch Features Shape:", batch.shape)
        break

if __name__ == "__main__":
    main()
