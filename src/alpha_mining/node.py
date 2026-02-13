from __future__ import annotations
from typing import List, Optional, Dict
import math
import numpy as np

class AlphaNode:
    def __init__(self, formula: str, parent: Optional[AlphaNode] = None, c_puct: float = 1.0):
        self.formula = formula
        self.parent = parent
        self.children: List[AlphaNode] = []
        self.c_puct = c_puct

        # MCTS Statistics
        self.visits = 0
        self.value = 0.0  # Q-value (moving average)
        self.max_value = 0.0 # Track max reward in subtree
        self.value_sum = 0.0  # Sum for mean calculation

        # Evaluation Metrics
        self.metrics: Dict[str, float] = {}

        # Alpha Info
        self.name = ""
        self.description = ""
        
    def add_child(self, child: AlphaNode):
        self.children.append(child)
        
    def update(self, reward: float, use_moving_avg: bool = True):
        self.visits += 1
        self.value_sum += reward
        self.max_value = max(self.max_value, reward)

        if use_moving_avg:
            # Use exponential moving average for better exploration
            alpha = 0.1
            self.value = (1 - alpha) * self.value + alpha * reward
        else:
            # Use max (as in original AlphaZero)
            self.value = max(self.value, reward)
        
    def get_uct_score(self, c: float = None) -> float:
        """
        Calculate UCT score for selection.
        UCT = Q(s, a) + c * sqrt(ln(N_parent) / N_child)
        """
        if self.visits == 0:
            return float('inf')

        c_value = c if c is not None else self.c_puct
        parent_visits = self.parent.visits if self.parent else 1
        exploitation = self.value
        exploration = c_value * math.sqrt(math.log(parent_visits) / self.visits)

        return exploitation + exploration

    @property
    def mean_value(self) -> float:
        """Return mean value across all visits"""
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def __repr__(self):
        return f"<AlphaNode {self.formula[:20]}... V={self.value:.3f} N={self.visits}>"
