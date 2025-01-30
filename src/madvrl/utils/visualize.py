import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")
sns.set_palette("husl")

# Create images directory if it doesn't exist
Path("images").mkdir(exist_ok=True)

# Load the VAE dataset
print("Loading dataset...")
df = pd.read_parquet("poker_game_metrics_vae.parquet")

# 1. Hand Strength Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='hand_strength', bins=50)
plt.title('Distribution of Hand Strengths')
plt.xlabel('Hand Strength')
plt.ylabel('Count')
plt.savefig('images/hand_strength_dist.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Pot Size Evolution
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='position', y='pot_size_bb')
plt.title('Pot Size Distribution by Position')
plt.xlabel('Position (0=dealer, 1=SB)')
plt.ylabel('Pot Size (BB)')
plt.savefig('images/pot_size_by_position.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Action Probabilities
action_probs = df[['action_fold_prob', 'action_call_prob', 'action_raise_prob']].mean()
plt.figure(figsize=(8, 8))
plt.pie(action_probs, labels=['Fold', 'Call', 'Raise'], autopct='%1.1f%%')
plt.title('Average Action Probabilities')
plt.savefig('images/action_probabilities.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Stack Size vs Pot Odds
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df.sample(10000), x='stack_bb', y='pot_odds', alpha=0.5)
plt.title('Stack Size vs Pot Odds')
plt.xlabel('Stack Size (BB)')
plt.ylabel('Pot Odds')
plt.savefig('images/stack_vs_pot_odds.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Betting Patterns
plt.figure(figsize=(10, 6))
sns.histplot(data=df[df['bet_amount_bb'] > 0], x='bet_amount_bb', bins=50)
plt.title('Distribution of Bet Sizes')
plt.xlabel('Bet Size (BB)')
plt.ylabel('Count')
plt.savefig('images/bet_size_dist.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Position Impact on Hand Strength
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='position', y='hand_strength')
plt.title('Hand Strength Distribution by Position')
plt.xlabel('Position (0=dealer, 1=SB)')
plt.ylabel('Hand Strength')
plt.savefig('images/hand_strength_by_position.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Stack-to-Pot Ratio vs Action Probabilities
plt.figure(figsize=(12, 6))
df_melted = pd.melt(df, 
                    value_vars=['action_fold_prob', 'action_call_prob', 'action_raise_prob'],
                    id_vars=['stack_to_pot'])
sns.boxplot(data=df_melted, x='variable', y='value', hue='variable')
plt.title('Action Probabilities vs Stack-to-Pot Ratio')
plt.xlabel('Action Type')
plt.ylabel('Probability')
plt.savefig('images/action_probs_by_spr.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. Correlation Matrix
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.savefig('images/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Print summary statistics
print("\nDataset Summary:")
print("-" * 50)
print(f"Total number of game states: {len(df):,}")
print("\nAverage metrics:")
print(f"Hand Strength: {df['hand_strength'].mean():.3f}")
print(f"Pot Size (BB): {df['pot_size_bb'].mean():.1f}")
print(f"Stack Size (BB): {df['stack_bb'].mean():.1f}")
print("\nAction Probabilities:")
print(f"Fold: {df['action_fold_prob'].mean():.1%}")
print(f"Call: {df['action_call_prob'].mean():.1%}")
print(f"Raise: {df['action_raise_prob'].mean():.1%}")
print("\nBetting Patterns:")
print(f"Average Bet Size (when betting): {df[df['bet_amount_bb'] > 0]['bet_amount_bb'].mean():.1f} BB")
print(f"Frequency of Betting: {(df['bet_amount_bb'] > 0).mean():.1%}")
