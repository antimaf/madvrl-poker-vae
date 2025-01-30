# MADVRL: Multi-Agent Deep Variational Reinforcement Learning

## Poker Game State Inference Project

### Overview
This project is part of the MADVRL (Multi-Agent Deep Variational Reinforcement Learning) research initiative, focusing on advanced probabilistic modeling of poker game states using Variational Autoencoders (VAEs).

### Project Structure
- `vae.py`: Core VAE implementation for hidden state inference
- `vae_preprocessing.py`: Data preprocessing and loading utilities
- `poker_game_metrics_full.parquet`: Full poker game metrics dataset

### Key Components
- Variational Autoencoder for hidden state representation
- Probabilistic modeling of poker game dynamics
- Advanced feature extraction and preprocessing

### Installation
```bash
git clone https://github.com/yourusername/madvrl-poker-vae.git
cd madvrl-poker-vae

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Training
Refer to `train_poker_vae.ipynb` for detailed training instructions.

### Research Objectives
- Learn latent representations of poker game states
- Capture uncertainty in opponent beliefs
- Enable more sophisticated multi-agent reinforcement learning strategies

### Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

### License
This project is licensed under the MIT License - see the LICENSE file for details.
