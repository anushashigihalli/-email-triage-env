"""Reward function with partial signals for Email Triage Environment."""

from typing import Dict, Any, List, Optional


class RewardCalculator:
    """Calculates rewards with partial credit and penalties."""
    
    def __init__(self):
        self.step_rewards: List[float] = []
        self.previous_action: Optional[Dict[str, Any]] = None
        
    def calculate_step_reward(
        self,
        action: Dict[str, Any],
        ground_truth: Dict[str, Any],
        task_id: str
    ) -> Dict[str, Any]:
        """
        Calculate reward for a single step with partial signals.
        
        Args:
            action: Agent's action
            ground_truth: Ground truth for comparison
            task_id: Current task identifier
            
        Returns:
            Dict with 'value', 'breakdown', and 'message'
        """
        breakdown = {}
        message = ""
        
        # Check for empty/no-op action
        if not action or all(v is None for v in action.values()):
            penalty = -0.1
            breakdown['noop_penalty'] = penalty
            message = "No-op action penalty"
            self.step_rewards.append(max(0.0, penalty))
            self.previous_action = action
            return {
                'value': max(0.0, penalty),
                'breakdown': breakdown,
                'message': message
            }
        
        # Check for repeated action penalty
        repeat_penalty = 0.0
        if self.previous_action and action.get('email_id') == self.previous_action.get('email_id'):
            if (action.get('classification') == self.previous_action.get('classification') and
                action.get('action') == self.previous_action.get('action')):
                repeat_penalty = -0.05
                breakdown['repeat_penalty'] = repeat_penalty
                message = "Repeated action penalty"
        
        # Positive rewards for correct actions
        positive_reward = 0.0
        
        # Classification reward
        if 'classification' in action and action['classification']:
            predicted_class = action['classification'].lower()
            true_class = ground_truth.get('true_category', '').lower()
            if predicted_class == true_class:
                positive_reward += 0.1
                breakdown['classification_correct'] = 0.1
        
        # Action reward
        if 'action' in action and action['action']:
            predicted_action = action['action'].lower()
            true_action = ground_truth.get('true_action', '').lower()
            if predicted_action == true_action:
                positive_reward += 0.1
                breakdown['action_correct'] = 0.1
        
        # Priority ranking reward (for medium task)
        if 'priority_ranking' in action and action['priority_ranking']:
            # Will be graded separately by MediumGrader
            breakdown['ranking_submitted'] = 0.0
        
        # Calculate final step reward
        step_reward = positive_reward + repeat_penalty
        
        # Ensure non-negative
        step_reward = max(0.0, step_reward)
        
        if not message and positive_reward > 0:
            message = f"Correct: +{positive_reward:.2f}"
        
        self.step_rewards.append(step_reward)
        self.previous_action = action.copy()
        
        return {
            'value': step_reward,
            'breakdown': breakdown,
            'message': message
        }
    
    def calculate_final_reward(self, max_possible_reward: float) -> float:
        """
        Calculate normalized final reward for the episode.
        
        Args:
            max_possible_reward: Maximum possible reward for normalization
            
        Returns:
            Normalized reward in [0.0, 1.0]
        """
        if not self.step_rewards:
            return 0.0
        
        total_reward = sum(self.step_rewards)
        
        # Normalize to [0.0, 1.0]
        if max_possible_reward > 0:
            normalized = total_reward / max_possible_reward
        else:
            normalized = 0.0
        
        return max(0.0, min(1.0, normalized))
    
    def reset(self):
        """Reset the reward calculator for a new episode."""
        self.step_rewards = []
        self.previous_action = None
