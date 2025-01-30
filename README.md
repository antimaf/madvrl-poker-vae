# MADVRL: Multi-Agent Deep Variational Reinforcement Learning

## Poker Game State Inference Project

### Overview
This project is part of the MADVRL (Multi-Agent Deep Variational Reinforcement Learning) research project, focusing on advanced probabilistic modeling of poker game states using Variational Autoencoders (VAEs).

### Amortised Inference in Variational Autoencoders

#### Inference as a Learned Optimisation

In traditional inference, we would compute the posterior $p(z|x)$ for each new observation $x$ by solving an optimization problem. Amortised inference fundamentally changes this approach by learning a *single* inference network $q_\phi(z|x)$ that can rapidly approximate the posterior for *any* input.

Mathematically, we transform the inference problem from:

$$z^* = \arg\max_{z} p(z|x)$$

To a learned mapping:

$$q_\phi(z|x) \approx p(z|x)$$

#### Key Mathematical Characteristics

1. **Inference Network Mapping**:
   $$z = f_\phi(x) + \epsilon$$
   Where $f_\phi$ is a neural network and $\epsilon \sim \mathcal{N}(0,1)$

2. **Posterior Approximation**:
   $$q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi(x))$$

3. **Marginal Likelihood Objective**:
   $$\max_\phi \mathbb{E}_{q_\phi(z|x)}[\log p(x|z)] - KL(q_\phi(z|x) \| p(z))$$

#### Inference Discard Strategy

After training, we discard the decoder and retain only the encoder $q_\phi(z|x)$, which serves as a probabilistic feature extractor. This approach allows:
- Rapid posterior approximation
- Compact representation of complex distributions
- Transfer of learned representations across different downstream tasks

The key insight is transforming inference from a per-instance optimisation to a learned, generalisable mapping.

### Project Structure
- `vae.py`: Core VAE implementation for hidden state inference
- `vae_preprocessing.py`: Data preprocessing and loading utilities
- `poker_game_metrics_full.parquet`: Full poker game metrics dataset
- `train_poker_vae.ipynb`: Training notebook

### Key Components
- Variational Autoencoder for hidden state representation
- Probabilistic modeling of poker game dynamics
- Advanced feature extraction and preprocessing

### Key Research Objectives
- Learn latent representations of poker game states
- Capture uncertainty in opponent beliefs
- Enable sophisticated multi-agent reinforcement learning strategies

### Installation

```bash
git clone https://github.com/antimaf/madvrl-poker-vae.git
cd madvrl-poker-vae

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Training
Open the notebook in Colab or Kaggle using the badges above.

### Dependencies
See `requirements.txt` for full dependency list.

