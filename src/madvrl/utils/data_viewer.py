import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class PokerDatasetAnalyzer:
    def __init__(self, dataset_path='poker_game_metrics.parquet'):
        """
        Initialize the dataset analyzer with the Parquet file path
        
        Args:
            dataset_path (str): Path to the Parquet file containing poker game data
        """
        self.dataset_path = dataset_path
        self.df = None
    
    def load_dataset(self):
        """
        Load the Parquet dataset into a pandas DataFrame
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            self.df = pd.read_parquet(self.dataset_path)
            print(f"Dataset loaded successfully. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def describe_dataset(self):
        """
        Generate a comprehensive statistical description of the dataset
        
        Returns:
            pd.DataFrame: Descriptive statistics of the dataset
        """
        if self.df is None:
            self.load_dataset()
        
        print("\n--- Dataset Overview ---")
        print(self.df.info())
        
        print("\n--- Descriptive Statistics ---")
        return self.df.describe()
    
    def visualize_feature_distributions(self, features=None):
        """
        Create distribution plots for specified features
        
        Args:
            features (list): List of feature names to plot. If None, uses all numeric columns.
        """
        if self.df is None:
            self.load_dataset()
        
        if features is None:
            features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        plt.figure(figsize=(15, 5 * ((len(features) - 1) // 3 + 1)))
        
        for i, feature in enumerate(features, 1):
            plt.subplot(((len(features) - 1) // 3 + 1), 3, i)
            sns.histplot(self.df[feature], kde=True)
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_position_impact(self):
        """
        Analyze the impact of player position on various metrics
        
        Returns:
            pd.DataFrame: Position-based analysis results
        """
        if self.df is None:
            self.load_dataset()
        
        position_analysis = {}
        
        # Aggregate metrics by position
        position_metrics = {
            'mean_hand_strength': 'hand_strength',
            'mean_pot_size': 'pot_size',
            'mean_bet_amount': 'bet_amount',
            'mean_stack_to_pot_ratio': 'stack_to_pot_ratio'
        }
        
        for metric_name, column in position_metrics.items():
            position_analysis[metric_name] = self.df.groupby('position')[column].mean()
        
        return pd.DataFrame(position_analysis)
    
    def export_summary(self, output_path='poker_dataset_summary.txt'):
        """
        Export dataset summary to a text file
        
        Args:
            output_path (str): Path to save the summary file
        """
        with open(output_path, 'w') as f:
            f.write("Poker Dataset Summary\n")
            f.write("=" * 30 + "\n\n")
            
            # Dataset Overview
            f.write("Dataset Overview:\n")
            f.write(f"Total Samples: {len(self.df)}\n")
            f.write(f"Features: {', '.join(self.df.columns)}\n\n")
            
            # Descriptive Statistics
            f.write("Descriptive Statistics:\n")
            desc_stats = self.df.describe().to_string()
            f.write(desc_stats + "\n\n")
            
            # Position Impact Analysis
            f.write("Position Impact Analysis:\n")
            pos_impact = self.analyze_position_impact().to_string()
            f.write(pos_impact)
        
        print(f"Summary exported to {output_path}")

def main():
    # Example usage
    analyzer = PokerDatasetAnalyzer()
    
    # Load the dataset
    df = analyzer.load_dataset()
    
    # Print descriptive statistics
    print(analyzer.describe_dataset())
    
    # Visualize feature distributions
    analyzer.visualize_feature_distributions()
    
    # Analyze position impact
    print("\nPosition Impact Analysis:")
    print(analyzer.analyze_position_impact())
    
    # Export summary
    analyzer.export_summary()

if __name__ == "__main__":
    main()
