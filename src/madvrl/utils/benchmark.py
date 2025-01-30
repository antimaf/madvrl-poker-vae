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

# Load the dataset
print("Loading dataset...")
df = pd.read_parquet("poker_game_metrics_vae.parquet")

# Professional poker benchmarks
BENCHMARKS = {
    'Aggression': {'min': 0.50, 'max': 0.60, 'description': 'Aggression Frequency'},
    'Fold_Freq': {'min': 0.15, 'max': 0.25, 'description': 'Fold Frequency'},
    'Raise_Freq': {'min': 0.25, 'max': 0.35, 'description': 'Raise Frequency'},
    'Position_Impact': {'min': 0.10, 'max': 0.15, 'description': 'Position Win Rate Delta'},
    'Bet_Sizing': {'min': 2.5, 'max': 3.5, 'description': 'Average Bet Size (BB)'}
}

# Calculate metrics from our dataset
our_metrics = {
    'Aggression': df['action_raise_prob'].mean(),
    'Fold_Freq': df['action_fold_prob'].mean(),
    'Raise_Freq': (df['bet_amount_bb'] > 0).mean(),
    'Position_Impact': df.groupby('position')['hand_strength'].mean().diff().mean(),
    'Bet_Sizing': df[df['bet_amount_bb'] > 0]['bet_amount_bb'].mean() / 10  # Normalize to typical sizing
}

# Create comparison visualization
plt.figure(figsize=(12, 6))
metrics = list(BENCHMARKS.keys())
x = np.arange(len(metrics))
width = 0.35

# Plot benchmark ranges
plt.bar(x - width/2, [BENCHMARKS[m]['max'] for m in metrics], width,
        label='Professional Max', alpha=0.3, color='green')
plt.bar(x - width/2, [BENCHMARKS[m]['min'] for m in metrics], width,
        label='Professional Min', alpha=0.3, color='blue')

# Plot our metrics
plt.bar(x + width/2, [our_metrics[m] for m in metrics], width,
        label='Our Model', color='red', alpha=0.6)

plt.xlabel('Metrics')
plt.ylabel('Frequency/Value')
plt.title('Poker Metrics Comparison: Our Model vs Professional Standards')
plt.xticks(x, metrics, rotation=45)
plt.legend()

plt.savefig('images/benchmark_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Calculate and print detailed comparison
print("\nDetailed Metric Comparison:")
print("-" * 50)
for metric in BENCHMARKS:
    benchmark = BENCHMARKS[metric]
    our_value = our_metrics[metric]
    within_range = benchmark['min'] <= our_value <= benchmark['max']
    
    print(f"\n{metric} ({benchmark['description']}):")
    print(f"Professional Range: {benchmark['min']:.3f} - {benchmark['max']:.3f}")
    print(f"Our Model: {our_value:.3f}")
    print(f"Status: {'✓ Within Range' if within_range else '✗ Out of Range'}")
    
    if not within_range:
        if our_value < benchmark['min']:
            print(f"Too low by {(benchmark['min'] - our_value):.3f}")
        else:
            print(f"Too high by {(our_value - benchmark['max']):.3f}")

# Additional analysis
print("\nDetailed Statistics:")
print("-" * 50)

# Position analysis
print("\nPosition Impact on Hand Strength:")
position_stats = df.groupby('position')['hand_strength'].agg(['mean', 'std'])
print(position_stats)

# Betting patterns by position
print("\nBetting Patterns by Position:")
position_bets = df[df['bet_amount_bb'] > 0].groupby('position')['bet_amount_bb'].agg(['mean', 'median', 'std'])
print(position_bets)

# Stack depth impact on aggression
print("\nStack Depth Impact on Aggression:")
stack_ranges = pd.qcut(df['stack_bb'], q=4)
stack_analysis = df.groupby(stack_ranges)['action_raise_prob'].agg(['mean', 'std'])
print(stack_analysis)

# Pot odds analysis
print("\nPot Odds Analysis:")
print(df['pot_odds'].describe())

# Save benchmark results
benchmark_results = pd.DataFrame({
    'Metric': metrics,
    'Our_Value': [our_metrics[m] for m in metrics],
    'Pro_Min': [BENCHMARKS[m]['min'] for m in metrics],
    'Pro_Max': [BENCHMARKS[m]['max'] for m in metrics],
    'Description': [BENCHMARKS[m]['description'] for m in metrics]
})
benchmark_results.to_csv('benchmark_results.csv', index=False)

# Additional visualizations
# 1. Stack size impact on betting
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[df['bet_amount_bb'] > 0], x=pd.qcut(df[df['bet_amount_bb'] > 0]['stack_bb'], q=4), y='bet_amount_bb')
plt.title('Stack Size Impact on Bet Sizing')
plt.xlabel('Stack Size Quartiles')
plt.ylabel('Bet Size (BB)')
plt.xticks(rotation=45)
plt.savefig('images/stack_impact.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Position impact on action probabilities
plt.figure(figsize=(10, 6))
action_probs = df.groupby('position')[['action_fold_prob', 'action_call_prob', 'action_raise_prob']].mean()
action_probs.plot(kind='bar', stacked=True)
plt.title('Action Probabilities by Position')
plt.xlabel('Position')
plt.ylabel('Probability')
plt.legend(title='Action Type')
plt.tight_layout()
plt.savefig('images/position_actions.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Hand strength distribution by position
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='position', y='hand_strength')
plt.title('Hand Strength Distribution by Position')
plt.xlabel('Position')
plt.ylabel('Hand Strength')
plt.savefig('images/position_strength.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Correlation heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('images/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
