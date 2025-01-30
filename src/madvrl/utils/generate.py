import numpy as np
import pandas as pd
import torch
import typing
from dataclasses import dataclass, asdict
import random
import itertools
from typing import List, Tuple, Dict
from collections import defaultdict

class PokerCard:
    """Represents a playing card"""
    SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
    
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
        self.rank_value = self.RANKS.index(rank)
    
    def __repr__(self):
        return f"{self.rank} of {self.suit}"
    
    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit
    
    def __hash__(self):
        return hash((self.rank, self.suit))

class PokerHandEvaluator:
    """Advanced Poker Hand Evaluator"""
    @staticmethod
    def evaluate_hand(hand: List[PokerCard], community_cards: List[PokerCard] = None) -> Tuple[str, float]:
        """Evaluate hand strength including community cards"""
        if community_cards is None:
            community_cards = []
        
        all_cards = hand + community_cards
        if len(all_cards) < 5:
            # For pre-flop, evaluate based on hole cards only
            return PokerHandEvaluator._evaluate_hole_cards(hand)
        
        return PokerHandEvaluator._evaluate_complete_hand(all_cards)
    
    @staticmethod
    def _evaluate_hole_cards(hand: List[PokerCard]) -> Tuple[str, float]:
        """Evaluate pre-flop hand strength"""
        if len(hand) != 2:
            raise ValueError("Hole cards must be exactly 2 cards")
        
        # Sort by rank value
        sorted_hand = sorted(hand, key=lambda x: x.rank_value, reverse=True)
        
        # Pocket pairs
        if hand[0].rank == hand[1].rank:
            rank_value = hand[0].rank_value
            strength = 0.5 + (rank_value / len(PokerCard.RANKS))
            return "Pocket Pair", strength
        
        # Suited cards
        suited = hand[0].suit == hand[1].suit
        
        # Calculate gap between cards
        gap = abs(hand[0].rank_value - hand[1].rank_value)
        
        # Base strength calculation
        high_card_value = max(card.rank_value for card in hand)
        base_strength = 0.3 + (high_card_value / len(PokerCard.RANKS))
        
        # Adjust for suitedness and connectedness
        if suited:
            base_strength += 0.1
        if gap == 1:  # Connected cards
            base_strength += 0.1
        elif gap == 2:  # One gap
            base_strength += 0.05
        
        return "High Card", min(0.99, base_strength)
    
    @staticmethod
    def _evaluate_complete_hand(cards: List[PokerCard]) -> Tuple[str, float]:
        """Evaluate complete 5+ card hand"""
        # Get best 5-card combination
        best_hand = max(itertools.combinations(cards, 5), 
                       key=lambda h: PokerHandEvaluator._hand_value(h))
        
        return PokerHandEvaluator._hand_value(best_hand)
    
    @staticmethod
    def _hand_value(hand: List[PokerCard]) -> Tuple[str, float]:
        """Calculate hand value for 5 cards"""
        sorted_hand = sorted(hand, key=lambda x: x.rank_value, reverse=True)
        is_flush = len(set(card.suit for card in hand)) == 1
        
        # Check for straight
        values = [card.rank_value for card in sorted_hand]
        is_straight = (max(values) - min(values) == 4 and len(set(values)) == 5)
        
        # Special case for Ace-low straight
        if not is_straight and set(values) == {12, 0, 1, 2, 3}:
            is_straight = True
        
        # Count rank frequencies
        rank_counts = defaultdict(int)
        for card in hand:
            rank_counts[card.rank] += 1
        
        # Determine hand type and base strength
        if is_flush and is_straight:
            if values[0] == 12:  # Ace-high
                return "Royal Flush", 1.0
            return "Straight Flush", 0.95 + (values[0] / 100)
        
        if 4 in rank_counts.values():
            rank = max((r for r, c in rank_counts.items() if c == 4),
                      key=lambda x: PokerCard.RANKS.index(x))
            return "Four of a Kind", 0.9 + (PokerCard.RANKS.index(rank) / 100)
        
        if 3 in rank_counts.values() and 2 in rank_counts.values():
            three_rank = max((r for r, c in rank_counts.items() if c == 3),
                           key=lambda x: PokerCard.RANKS.index(x))
            return "Full House", 0.85 + (PokerCard.RANKS.index(three_rank) / 100)
        
        if is_flush:
            return "Flush", 0.8 + (values[0] / 100)
        
        if is_straight:
            return "Straight", 0.75 + (values[0] / 100)
        
        if 3 in rank_counts.values():
            rank = max((r for r, c in rank_counts.items() if c == 3),
                      key=lambda x: PokerCard.RANKS.index(x))
            return "Three of a Kind", 0.7 + (PokerCard.RANKS.index(rank) / 100)
        
        if list(rank_counts.values()).count(2) == 2:
            pairs = sorted([r for r, c in rank_counts.items() if c == 2],
                         key=lambda x: PokerCard.RANKS.index(x), reverse=True)
            return "Two Pair", 0.6 + (PokerCard.RANKS.index(pairs[0]) / 100)
        
        if 2 in rank_counts.values():
            rank = max((r for r, c in rank_counts.items() if c == 2),
                      key=lambda x: PokerCard.RANKS.index(x))
            return "One Pair", 0.5 + (PokerCard.RANKS.index(rank) / 100)
        
        return "High Card", 0.4 + (values[0] / 100)

class CFRTrainer:
    """CFR strategy trainer for poker"""
    def __init__(self, num_actions=3):  # fold, call, raise
        self.num_actions = num_actions
        self.regret_sum = defaultdict(lambda: np.zeros(num_actions))
        self.strategy_sum = defaultdict(lambda: np.zeros(num_actions))
        self.num_iterations = 0
        
        # Position-based strategy adjustments (increased position impact)
        self.position_weights = {
            'dealer': 2.0,     # Extremely aggressive in position
            'small_blind': 1.2, # Moderately aggressive
            'big_blind': 0.6    # Conservative out of position
        }
        
        # Stack-depth strategy adjustments (increased overall aggression)
        self.stack_depth_weights = {
            'deep': 2.0,     # Extremely aggressive with deep stacks
            'medium': 1.5,   # Very aggressive
            'short': 1.0     # Standard aggression with short stacks
        }
    
    def _get_weights(self, info_set: str) -> Tuple[float, float]:
        """Extract position and stack depth weights from info set"""
        try:
            parts = info_set.split('|')
            position = parts[5]
            stack_to_pot = float(parts[6])
            
            position_weight = self.position_weights.get(position, 1.0)
            
            if stack_to_pot > 50:
                stack_weight = self.stack_depth_weights['deep']
            elif stack_to_pot > 20:
                stack_weight = self.stack_depth_weights['medium']
            else:
                stack_weight = self.stack_depth_weights['short']
            
            return position_weight, stack_weight
        except (IndexError, ValueError):
            return 1.5, 1.5  # Default to aggressive
    
    def get_strategy(self, info_set: str) -> np.ndarray:
        """Get current strategy for an information set"""
        regrets = self.regret_sum[info_set]
        positive_regrets = np.maximum(regrets, 0)
        
        # Apply position and stack depth adjustments
        position_weight, stack_weight = self._get_weights(info_set)
        total_weight = position_weight * stack_weight
        
        # Increase aggression for raise action (index 2)
        positive_regrets[2] *= total_weight * 2.0  # Doubled aggression multiplier
        
        sum_positive_regret = np.sum(positive_regrets)
        
        if sum_positive_regret > 0:
            strategy = positive_regrets / sum_positive_regret
        else:
            # More aggressive default strategy
            strategy = np.array([0.1, 0.3, 0.6])  # [fold, call, raise]
            strategy = strategy * np.array([1.0, 1.0, total_weight])
            strategy = strategy / np.sum(strategy)
        
        return strategy
    
    def update(self, info_set: str, strategy: np.ndarray, regret: np.ndarray):
        """Update regret and strategy sums with increased learning rate"""
        # Increase learning rate for faster adaptation
        learning_rate = 1.2
        self.regret_sum[info_set] += regret * learning_rate
        self.strategy_sum[info_set] += strategy
        self.num_iterations += 1
    
    def get_average_strategy(self, info_set: str) -> np.ndarray:
        """Get average strategy across all iterations"""
        strategy_sum = self.strategy_sum[info_set]
        total = np.sum(strategy_sum)
        if total > 0:
            return strategy_sum / total
        return np.array([0.2, 0.3, 0.5])  # Default to aggressive strategy

    def _calculate_bet_size(self, state: GameState, player_id: int, position: str) -> float:
        """Calculate appropriate bet size based on position, pot, and stack depth"""
        pot = state.pot
        stack = state.player_chips[player_id]
        stack_to_pot = stack / pot if pot > 0 else 100
        
        # Base bet sizing as percentage of pot (increased sizes)
        if state.round == 0:  # Preflop
            if position == 'dealer':
                base_size = 4.0  # 4x BB from dealer
            else:
                base_size = 3.5  # 3.5x BB from other positions
        else:
            if stack_to_pot > 50:  # Deep stack
                base_size = 1.0   # Full pot
            elif stack_to_pot > 20:  # Medium stack
                base_size = 0.85  # 85% pot
            else:  # Short stack
                base_size = 0.75  # 75% pot
        
        # Calculate actual bet size
        bet_size = pot * base_size
        
        # Adjust for position (increased multipliers)
        position_multiplier = {
            'dealer': 2.0,    # Extremely aggressive in position
            'small_blind': 1.5,  # Very aggressive
            'big_blind': 1.2     # Moderately aggressive out of position
        }.get(position, 1.0)
        
        bet_size *= position_multiplier
        
        # Ensure bet is within valid range
        min_bet = state.current_bet + state.min_raise
        max_bet = min(stack, state.pot * 5)  # Cap at 5x pot
        
        return max(min_bet, min(bet_size, max_bet))

    def _get_legal_actions(self, state: GameState, player_id: int) -> List[str]:
        """Get legal actions for current state"""
        actions = ['fold']
        
        # Check/call decision
        if state.current_bet == 0:
            actions.append('check')
        else:
            # Only allow call if player has enough chips
            if state.player_chips[player_id] >= state.current_bet:
                actions.append('call')
        
        # Raise decision - consider position and stack depth
        if state.player_chips[player_id] >= state.current_bet + state.min_raise:
            position = ['dealer', 'small_blind', 'big_blind'][player_id % 3]
            stack_to_pot = state.player_chips[player_id] / state.pot if state.pot > 0 else 100
            
            # Further lowered raise thresholds
            raise_threshold = {
                'dealer': 5,       # Extremely aggressive in position
                'small_blind': 10, # Very aggressive
                'big_blind': 15    # Moderately aggressive out of position
            }.get(position, 10)
            
            if stack_to_pot > raise_threshold:
                actions.append('raise')
        
        return actions

@dataclass
class GameState:
    """Current state of the poker game"""
    pot: float
    current_player: int
    round: int  # 0: pre-flop, 1: flop, 2: turn, 3: river
    community_cards: List[PokerCard]
    player_hands: List[List[PokerCard]]
    player_chips: List[float]
    current_bet: float
    min_raise: float
    last_action: str
    last_bet: float
    big_blind: float

@dataclass
class PokerGameMetrics:
    """Comprehensive metrics for a poker game state"""
    game_id: int
    hand_id: int
    round_id: int
    player_id: int
    position: str  # dealer, small_blind, big_blind
    
    # Hand information
    hole_cards: str
    community_cards: str
    hand_type: str
    hand_strength: float
    
    # Action and betting
    action_taken: str
    bet_amount: float
    pot_size: float
    
    # Position and stack metrics
    position_relative_to_button: int
    stack_to_pot_ratio: float
    big_blinds_remaining: float
    
    # Strategy metrics
    cfr_strategy: List[float]
    action_probabilities: List[float]
    realized_equity: float
    
    def to_dict(self):
        d = asdict(self)
        # Convert lists to strings for proper serialization
        d['cfr_strategy'] = str(self.cfr_strategy)
        d['action_probabilities'] = str(self.action_probabilities)
        return d

class PokerGameSimulator:
    """Advanced Poker Game Simulator with CFR"""
    def __init__(self, num_players=2, initial_stack=1000, small_blind=5, big_blind=10):
        self.num_players = num_players
        self.initial_stack = initial_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.cfr_trainer = CFRTrainer()
    
    def _create_info_set(self, state: GameState, player_id: int) -> str:
        """Create information set string for CFR"""
        hand = state.player_hands[player_id]
        hand_str = ','.join(str(card) for card in sorted(hand, key=lambda x: (x.rank_value, x.suit)))
        community_str = ','.join(str(card) for card in state.community_cards)
        position = ['dealer', 'small_blind', 'big_blind'][player_id % 3]
        stack_to_pot = state.player_chips[player_id] / state.pot if state.pot > 0 else 100
        return f"{hand_str}|{community_str}|{state.pot}|{state.current_bet}|{state.round}|{position}|{stack_to_pot:.1f}"

    def _get_legal_actions(self, state: GameState, player_id: int) -> List[str]:
        """Get legal actions for current state"""
        actions = ['fold']
        
        # Check/call decision
        if state.current_bet == 0:
            actions.append('check')
        else:
            # Only allow call if player has enough chips
            if state.player_chips[player_id] >= state.current_bet:
                actions.append('call')
        
        # Raise decision - consider position and stack depth
        if state.player_chips[player_id] >= state.current_bet + state.min_raise:
            position = ['dealer', 'small_blind', 'big_blind'][player_id % 3]
            stack_to_pot = state.player_chips[player_id] / state.pot if state.pot > 0 else 100
            
            # Lower raise thresholds to encourage more raising
            raise_threshold = {
                'dealer': 10,      # Very aggressive in position
                'small_blind': 15, # Moderately aggressive
                'big_blind': 20    # Still somewhat aggressive out of position
            }.get(position, 15)
            
            if stack_to_pot > raise_threshold:
                actions.append('raise')
        
        return actions

    def _apply_action(self, state: GameState, action: str, amount: float = 0) -> GameState:
        """Apply action to game state"""
        player = state.current_player
        position = ['dealer', 'small_blind', 'big_blind'][player % 3]
        stack_to_pot = state.player_chips[player] / state.pot if state.pot > 0 else 100
        
        if action == 'fold':
            state.player_chips[player] = -1  # Mark as folded
        elif action in ['call', 'check']:
            bet_amount = state.current_bet - amount if amount > 0 else state.current_bet
            state.player_chips[player] -= bet_amount
            state.pot += bet_amount
        elif action == 'raise':
            # Calculate raise amount with position and stack depth considerations
            if amount == 0:
                amount = self._calculate_bet_size(state, player, position)
            state.player_chips[player] -= amount
            state.pot += amount
            state.current_bet = amount
            state.min_raise = max(amount - state.last_bet, state.big_blind)
            state.last_bet = amount
        
        state.last_action = action
        return state

    def _calculate_bet_size(self, state: GameState, player_id: int, position: str) -> float:
        """Calculate appropriate bet size based on position, pot, and stack depth"""
        pot = state.pot
        stack = state.player_chips[player_id]
        stack_to_pot = stack / pot if pot > 0 else 100
        
        # Base bet sizing as percentage of pot
        if state.round == 0:  # Preflop
            if position == 'dealer':
                base_size = 3.5  # 3.5x BB from dealer
            else:
                base_size = 3.0  # 3x BB from other positions
        else:
            if stack_to_pot > 50:  # Deep stack
                base_size = 0.85  # 85% pot
            elif stack_to_pot > 20:  # Medium stack
                base_size = 0.75  # 75% pot
            else:  # Short stack
                base_size = 0.6  # 60% pot
        
        # Calculate actual bet size
        bet_size = pot * base_size
        
        # Adjust for position
        position_multiplier = {
            'dealer': 1.5,    # Much more aggressive in position
            'small_blind': 1.2,  # Moderately aggressive
            'big_blind': 1.0     # Standard sizing out of position
        }.get(position, 1.0)
        
        bet_size *= position_multiplier
        
        # Ensure bet is within valid range
        min_bet = state.current_bet + state.min_raise
        max_bet = min(stack, state.pot * 4)  # Cap at 4x pot
        
        return max(min_bet, min(bet_size, max_bet))

    def generate_dataset(self, num_games=1000000):
        """Generate comprehensive poker dataset"""
        # Pre-allocate lists for better memory efficiency
        metrics_list = []
        
        print(f"Generating {num_games} poker games...")
        
        for game_id in range(num_games):
            if game_id % 10000 == 0:
                print(f"Progress: {game_id}/{num_games} games")
            
            # Initialize deck
            deck = list(itertools.product(PokerCard.RANKS, PokerCard.SUITS))
            random.shuffle(deck)
            
            # Deal hole cards
            player_hands = [
                [PokerCard(r, s) for r, s in deck[i:i+2]]
                for i in range(0, self.num_players * 2, 2)
            ]
            
            # Initialize game state
            state = GameState(
                pot=self.small_blind + self.big_blind,
                current_player=0,
                round=0,
                community_cards=[],
                player_hands=player_hands,
                player_chips=[self.initial_stack] * self.num_players,
                current_bet=self.big_blind,
                min_raise=self.big_blind,
                last_action='',
                last_bet=self.big_blind,
                big_blind=self.big_blind
            )
            
            # Play each round
            for round_id in range(4):  # pre-flop, flop, turn, river
                if round_id > 0:
                    # Deal community cards
                    new_cards = [
                        PokerCard(r, s) 
                        for r, s in deck[self.num_players*2 + (round_id-1)*3:
                                      self.num_players*2 + round_id*3]
                    ]
                    state.community_cards.extend(new_cards)
                
                # Each player acts
                for player_id in range(self.num_players):
                    if state.player_chips[player_id] < 0:  # Skip folded players
                        continue
                    
                    # Get player's position
                    position = ['dealer', 'small_blind', 'big_blind'][player_id % 3]
                    
                    # Create information set
                    info_set = self._create_info_set(state, player_id)
                    
                    # Get CFR strategy
                    strategy = self.cfr_trainer.get_strategy(info_set)
                    
                    # Get legal actions and their probabilities
                    legal_actions = self._get_legal_actions(state, player_id)
                    action_probs = strategy[:len(legal_actions)]
                    action_probs = action_probs / np.sum(action_probs)
                    
                    # Choose action
                    action = np.random.choice(legal_actions, p=action_probs)
                    
                    # Calculate bet amount if raising
                    bet_amount = 0
                    if action == 'raise':
                        bet_amount = self._calculate_bet_size(state, player_id, position)
                    
                    # Apply action
                    state = self._apply_action(state, action, bet_amount)
                    
                    # Calculate hand strength
                    hand_type, hand_strength = PokerHandEvaluator.evaluate_hand(
                        state.player_hands[player_id],
                        state.community_cards
                    )
                    
                    # Record VAE-relevant metrics
                    metrics = {
                        'game_id': game_id,
                        'player_id': player_id,
                        'round_id': round_id,
                        'hand_strength': hand_strength,
                        'pot_size_bb': state.pot / self.big_blind,
                        'stack_bb': state.player_chips[player_id] / self.big_blind,
                        'position': player_id,  # 0=dealer, 1=small_blind, 2=big_blind
                        'action_fold_prob': action_probs[0] if 'fold' in legal_actions else 0,
                        'action_call_prob': action_probs[1] if 'call' in legal_actions else 0,
                        'action_raise_prob': action_probs[2] if 'raise' in legal_actions else 0,
                        'bet_amount_bb': bet_amount / self.big_blind if bet_amount > 0 else 0,
                        'pot_odds': state.current_bet / state.pot if state.pot > 0 else 0,
                        'stack_to_pot': state.player_chips[player_id] / state.pot if state.pot > 0 else 100,
                    }
                    
                    metrics_list.append(metrics)
                    
                    # Update CFR
                    reward = state.pot if action != 'fold' else 0
                    regret = np.zeros(self.cfr_trainer.num_actions)
                    action_idx = legal_actions.index(action)
                    regret[action_idx] = reward
                    self.cfr_trainer.update(info_set, strategy, regret)
                    
                    # Early termination if all but one player folded
                    active_players = sum(1 for chips in state.player_chips if chips >= 0)
                    if active_players <= 1:
                        break
                
                if active_players <= 1:
                    break
        
        # Convert to DataFrame
        print("Converting to DataFrame...")
        df = pd.DataFrame(metrics_list)
        
        # Save both full and compact versions
        print("Saving datasets...")
        df.to_parquet("poker_game_metrics_full.parquet", index=False)
        
        # Create compact version with only essential features for VAE
        essential_features = [
            'hand_strength', 'pot_size_bb', 'stack_bb', 'position',
            'action_fold_prob', 'action_call_prob', 'action_raise_prob',
            'bet_amount_bb', 'pot_odds', 'stack_to_pot'
        ]
        df_compact = df[essential_features]
        df_compact.to_parquet("poker_game_metrics_vae.parquet", index=False)
        
        print("\nDataset Statistics:")
        print("-" * 50)
        print(f"Total number of game states: {len(df)}")
        print("\nFeature Statistics:")
        print(df_compact.describe())
        
        return df

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate dataset with 1M games
    generator = PokerGameSimulator(num_players=2)
    dataset = generator.generate_dataset(num_games=1000000)