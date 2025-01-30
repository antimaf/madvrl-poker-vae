import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.distributions import Normal, kl_divergence
from typing import Tuple, Dict, Optional
from torch.utils.data import Dataset, DataLoader

from madvrl.preprocessing.vae_preprocessing import create_vae_dataloader

class PokerDataset(Dataset):
    """PyTorch Dataset for Poker Game Metrics"""
    def __init__(self, parquet_file):
        # Load dataset
        self.data = pd.read_parquet(parquet_file)
        
        # Convert to tensor
        self.features = torch.tensor(self.data.drop(columns=['game_id', 'player_id']).values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.features[idx]

class PokerInferenceVAE(nn.Module):
    """
    Enhanced Variational Autoencoder for Poker State Inference
    
    Learns probabilistic representations of poker game states
    """
    def __init__(
        self,
        input_dim: int,          # Number of input features
        obs_dim: int = 256,      # Dimension of observations
        hidden_dim: int = 512,   # LSTM hidden dimension
        latent_dim: int = 128,   # Latent space dimension
        num_layers: int = 2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Preprocessing layer
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, obs_dim),
            nn.ReLU()
        )
        
        # Encoder: q_φ(z|o_{1:t})
        self.encoder_lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Latent space projections
        self.mu_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.logvar_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x):
        """Encode input to latent distribution"""
        # Preprocess input
        x_proj = self.input_projection(x)
        
        # LSTM encoding
        lstm_out, _ = self.encoder_lstm(x_proj.unsqueeze(0))
        
        # Take the last bidirectional output
        encoded = lstm_out.squeeze(0)[-1]
        
        # Get mu and log variance
        mu = self.mu_encoder(encoded)
        logvar = self.logvar_encoder(encoded)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent representation"""
        return self.decoder(z)
    
    def forward(self, x):
        """Full VAE forward pass"""
        # Encode
        mu, logvar = self.encode(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decode(z)
        
        return x_recon, mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        """
        Compute VAE loss:
        - Reconstruction loss (MSE)
        - KL Divergence to encourage latent space regularization
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x)
        
        # KL Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kl_loss
    
    def predict_chip_dynamics(self, x):
        """
        Predict future chip dynamics based on current game state
        
        Args:
            x (torch.Tensor): Current game state features
        
        Returns:
            Predicted chip changes and betting probabilities
        """
        # Encode to latent space
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        # Predict chip changes
        chip_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Predict chip change, bet size, all-in probability
        )
        
        return chip_predictor(z)

class PokerHiddenStateVAE(nn.Module):
    """
    Variational Autoencoder for Poker Hidden State Inference
    
    Learns probabilistic representations of poker game states
    Captures uncertainty in opponent beliefs and hidden states
    """
    def __init__(
        self,
        input_dim: int = 10,          # Number of input features
        hidden_state_dim: int = 5,    # Dimension of hidden state
        hidden_dims: list = [64, 128],# Hidden layer dimensions
        latent_dim: int = 32,         # Dimension of latent space
        dropout_rate: float = 0.2,    # Dropout rate for regularization
        beta: float = 1.0,            # KL divergence weight
        alpha: float = 1.0            # Hidden state supervision weight
    ):
        """
        Initialize Poker Hidden State VAE
        
        Args:
            input_dim (int): Number of input features
            hidden_state_dim (int): Dimension of hidden state to predict
            hidden_dims (list): Dimensions of hidden layers
            latent_dim (int): Dimension of latent space
            dropout_rate (float): Dropout rate for regularization
            beta (float): KL divergence weight
            alpha (float): Hidden state supervision weight
        """
        super().__init__()
        
        # Hyperparameters
        self.input_dim = input_dim
        self.hidden_state_dim = hidden_state_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.beta = beta
        self.alpha = alpha
        
        # Encoder Network
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent Space Projection
        self.fc_mu = nn.Linear(prev_dim, latent_dim)      # Mean projection
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)  # Log variance projection
        
        # Hidden State Predictor
        hidden_state_layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            hidden_state_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        hidden_state_layers.append(nn.Linear(prev_dim, hidden_state_dim))
        self.hidden_state_predictor = nn.Sequential(*hidden_state_layers)
        
        # Decoder Network (for input reconstruction)
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters
        
        Args:
            x (torch.Tensor): Input features
        
        Returns:
            Tuple of latent mean and log variance
        """
        # Encode input through hidden layers
        h = self.encoder(x)
        
        # Project to latent space
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution
        
        Args:
            mu (torch.Tensor): Mean of latent distribution
            log_var (torch.Tensor): Log variance of latent distribution
        
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def predict_hidden_state(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict hidden state from latent representation
        
        Args:
            z (torch.Tensor): Latent vector
        
        Returns:
            Predicted hidden state
        """
        return self.hidden_state_predictor(z)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstructed input
        
        Args:
            z (torch.Tensor): Latent vector
        
        Returns:
            Reconstructed input features
        """
        return self.decoder(z)
    
    def forward(
        self, 
        x: torch.Tensor, 
        hidden_state: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through VAE
        
        Args:
            x (torch.Tensor): Input features
            hidden_state (torch.Tensor, optional): True hidden state for supervision
        
        Returns:
            Dictionary of outputs including reconstructed features, 
            latent distribution parameters
        """
        # Encode input to latent distribution
        mu, log_var = self.encode(x)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, log_var)
        
        # Reconstruct input
        x_recon = self.decode(z)
        
        # Predict hidden state
        predicted_hidden_state = self.predict_hidden_state(z)
        
        return {
            'x_recon': x_recon,
            'mu': mu,
            'log_var': log_var,
            'z': z,
            'predicted_hidden_state': predicted_hidden_state,
            'true_hidden_state': hidden_state
        }
    
    def loss_function(
        self, 
        x: torch.Tensor, 
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss function with hidden state supervision
        
        Args:
            x (torch.Tensor): Original input
            outputs (dict): Model outputs from forward pass
        
        Returns:
            Dictionary of loss components
        """
        # Reconstruction loss (Mean Squared Error)
        recon_loss = F.mse_loss(outputs['x_recon'], x, reduction='mean')
        
        # KL Divergence loss
        mu, log_var = outputs['mu'], outputs['log_var']
        kl_div = -0.5 * torch.mean(
            1 + log_var - mu.pow(2) - log_var.exp()
        )
        
        # Hidden State Supervision Loss
        if outputs['true_hidden_state'] is not None:
            hidden_state_loss = F.mse_loss(
                outputs['predicted_hidden_state'], 
                outputs['true_hidden_state'], 
                reduction='mean'
            )
        else:
            hidden_state_loss = torch.tensor(0.0, device=x.device)
        
        # Total loss
        loss = (
            recon_loss + 
            self.beta * kl_div + 
            self.alpha * hidden_state_loss
        )
        
        return {
            'loss': loss,
            'reconstruction_loss': recon_loss,
            'kl_divergence': kl_div,
            'hidden_state_loss': hidden_state_loss
        }

def train_step(
    model: PokerInferenceVAE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    """
    Training step for the Poker Inference VAE
    
    Args:
        model: VAE model to train
        dataloader: Dataset loader
        optimizer: Optimiser
        device: Computation device
    """
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        batch = batch.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, mu, logvar = model(batch)
        
        # Compute loss
        loss = model.loss_function(recon_batch, batch, mu, logvar)
        
        # Backpropagate
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train_poker_vae(
    model: PokerHiddenStateVAE,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    epochs: int = 1000,
    log_interval: int = 50,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Dict[str, list]:
    """
    Train Poker Hidden State VAE
    
    Args:
        model (PokerHiddenStateVAE): VAE model to train
        dataloader (DataLoader): Training data loader
        optimizer (Optimizer): Optimizer for training
        device (torch.device): Computation device
        epochs (int): Number of training epochs
        log_interval (int): Interval for logging training progress
        scheduler (Optional[_LRScheduler]): Learning rate scheduler
    
    Returns:
        Dictionary of training metrics
    """
    # Move model to device
    model.to(device)
    model.train()
    
    # Tracking metrics
    metrics = {
        'total_loss': [],
        'reconstruction_loss': [],
        'kl_divergence': [],
        'hidden_state_loss': []
    }
    
    # Training loop
    for epoch in range(epochs):
        epoch_metrics = {
            'total_loss': 0,
            'reconstruction_loss': 0,
            'kl_divergence': 0,
            'hidden_state_loss': 0
        }
        
        for batch_idx, x in enumerate(dataloader):
            # Move data to device
            x = x.to(device)
            
            # Extract hidden state if available (optional)
            # For this example, we'll use a subset of features as hidden state
            hidden_state = x[:, :model.hidden_state_dim] if model.hidden_state_dim > 0 else None
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(x, hidden_state)
            
            # Compute loss
            loss_dict = model.loss_function(x, outputs)
            loss = loss_dict['loss']
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_metrics['total_loss'] += loss.item()
            epoch_metrics['reconstruction_loss'] += loss_dict['reconstruction_loss'].item()
            epoch_metrics['kl_divergence'] += loss_dict['kl_divergence'].item()
            epoch_metrics['hidden_state_loss'] += loss_dict['hidden_state_loss'].item()
        
        # Average epoch metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= len(dataloader)
            metrics[key].append(epoch_metrics[key])
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(epoch_metrics['total_loss'])
        
        # Log progress
        if epoch % log_interval == 0:
            print(f"\nEpoch {epoch}:")
            for key, value in epoch_metrics.items():
                print(f"  {key}: {value:.6f}")
    
    return metrics

def main():
    """
    Example usage of Poker Hidden State VAE with extended training
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import sys
    from madvrl.preprocessing.vae_preprocessing import create_vae_dataloader
    
    # Ensure full output
    np.set_printoptions(threshold=sys.maxsize)
    
    # Hyperparameters
    input_dim = 10
    hidden_state_dim = 5  # Subset of features as hidden state
    hidden_dims = [64, 128]
    latent_dim = 32
    batch_size = 256
    learning_rate = 1e-3
    
    # Create DataLoader
    dataloader = create_vae_dataloader('poker_game_metrics_full.parquet')
    
    # Initialize model
    model = PokerHiddenStateVAE(
        input_dim=input_dim,
        hidden_state_dim=hidden_state_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        beta=0.1,  # Adjust KL divergence weight
        alpha=1.0  # Adjust hidden state supervision weight
    )
    
    # Optimizer with learning rate scheduling
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=50, 
        verbose=True
    )
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Train model for 1000 epochs
    epochs = 1000
    metrics = train_poker_vae(
        model, 
        dataloader, 
        optimizer, 
        device=device, 
        epochs=epochs, 
        log_interval=50,
        scheduler=scheduler
    )
    
    # Visualize training metrics
    plt.figure(figsize=(15, 10))
    
    # Total Loss
    plt.subplot(2, 2, 1)
    plt.plot(metrics['total_loss'], label='Total Loss')
    plt.title('Total Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    
    # Reconstruction Loss
    plt.subplot(2, 2, 2)
    plt.plot(metrics['reconstruction_loss'], label='Reconstruction Loss', color='green')
    plt.title('Reconstruction Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    
    # KL Divergence
    plt.subplot(2, 2, 3)
    plt.plot(metrics['kl_divergence'], label='KL Divergence', color='red')
    plt.title('KL Divergence over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.yscale('log')
    plt.legend()
    
    # Hidden State Loss
    plt.subplot(2, 2, 4)
    plt.plot(metrics['hidden_state_loss'], label='Hidden State Loss', color='purple')
    plt.title('Hidden State Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('poker_vae_training_metrics.png')
    plt.close()
    
    # Print final metrics with more detail
    print("\n" + "="*50)
    print("FINAL TRAINING METRICS")
    print("="*50)
    for key, value_list in metrics.items():
        print(f"\n{key.upper()}:")
        print(f"  Final Value:     {value_list[-1]:.6f}")
        print(f"  Minimum Value:   {min(value_list):.6f}")
        print(f"  Maximum Value:   {max(value_list):.6f}")
        print(f"  Mean Value:      {np.mean(value_list):.6f}")
        print(f"  Standard Dev:    {np.std(value_list):.6f}")
    
    # Optional: Save model
    torch.save(model.state_dict(), 'poker_hidden_state_vae.pth')
    
    return model, metrics

if __name__ == "__main__":
    main()