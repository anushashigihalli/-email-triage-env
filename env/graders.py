"""Deterministic graders for Email Triage Environment."""

from typing import List, Dict, Any, Optional
from scipy.stats import kendalltau
import numpy as np


class EasyGrader:
    """Grader for single label classification task."""
    
    def grade(self, action: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        """
        Grade the agent's classification action.
        
        Args:
            action: Agent's action with 'classification' field
            ground_truth: Contains 'true_category' field
            
        Returns:
            Score in [0.0, 1.0] - 1.0 for exact match, 0.0 otherwise
        """
        if not action or 'classification' not in action:
            return 0.0
        
        predicted = action['classification'].lower() if action['classification'] else ""
        actual = ground_truth.get('true_category', '').lower()
        
        return 1.0 if predicted == actual else 0.0


class MediumGrader:
    """Grader for priority sorting task using Kendall's tau."""
    
    def grade(self, action: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        """
        Grade the agent's priority ranking using Kendall's tau correlation.
        
        Args:
            action: Agent's action with 'priority_ranking' field (list of email IDs)
            ground_truth: Contains email IDs with their true priorities
            
        Returns:
            Score in [0.0, 1.0] normalized from Kendall's tau [-1.0, 1.0]
        """
        if not action or 'priority_ranking' not in action:
            return 0.0
        
        ranking = action['priority_ranking']
        if not ranking or len(ranking) == 0:
            return 0.0
        
        # Get ground truth priorities
        email_priorities = ground_truth.get('email_priorities', {})
        
        # Filter to only emails that exist in ground truth
        valid_ranking = [eid for eid in ranking if eid in email_priorities]
        
        if len(valid_ranking) < 2:
            return 0.0
        
        # Create predicted ranking (position in list = priority rank)
        predicted_ranks = {email_id: idx for idx, email_id in enumerate(valid_ranking)}
        
        # Get true ranks
        true_ranks = {eid: email_priorities[eid] for eid in valid_ranking}
        
        # Align by email ID
        email_ids = list(valid_ranking)
        predicted_values = [predicted_ranks[eid] for eid in email_ids]
        true_values = [true_ranks[eid] for eid in email_ids]
        
        # Calculate Kendall's tau
        if len(email_ids) < 2:
            return 0.0
        
        try:
            tau, _ = kendalltau(true_values, predicted_values)
            # Normalize from [-1, 1] to [0, 1]
            normalized_score = (tau + 1.0) / 2.0
            return max(0.0, min(1.0, normalized_score))
        except:
            return 0.0


class HardGrader:
    """Grader for triage and respond task with weighted sub-scores."""
    
    def grade(self, action: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        """
        Grade the agent's triage action with multiple components.
        
        Args:
            action: Agent's action with 'classification', 'action', and optionally 'reply_text'
            ground_truth: Contains 'true_category', 'true_action', and 'reply_keywords'
            
        Returns:
            Score in [0.0, 1.0] with weighted components:
            - Classification accuracy: 0.4
            - Action correctness: 0.4
            - Reply quality: 0.2
        """
        if not action:
            return 0.0
        
        # Classification score (0.0 - 0.4)
        class_score = 0.0
        if 'classification' in action and action['classification']:
            predicted_class = action['classification'].lower()
            true_class = ground_truth.get('true_category', '').lower()
            if predicted_class == true_class:
                class_score = 0.4
        
        # Action score (0.0 - 0.4)
        action_score = 0.0
        if 'action' in action and action['action']:
            predicted_action = action['action'].lower()
            true_action = ground_truth.get('true_action', '').lower()
            if predicted_action == true_action:
                action_score = 0.4
        
        # Reply quality score (0.0 - 0.2)
        reply_score = 0.0
        if 'reply_text' in action and action['reply_text']:
            reply_text = action['reply_text'].lower()
            reply_keywords = ground_truth.get('reply_keywords', [])
            
            if reply_keywords and len(reply_keywords) > 0:
                # Check how many keywords appear in the reply
                matched_keywords = sum(1 for kw in reply_keywords if kw.lower() in reply_text)
                keyword_ratio = matched_keywords / len(reply_keywords)
                reply_score = 0.2 * keyword_ratio
            elif not reply_keywords:
                # If no keywords expected (e.g., for spam/newsletter), giving reply is neutral
                reply_score = 0.1  # Partial credit for attempting
        
        total_score = class_score + action_score + reply_score
        return max(0.0, min(1.0, total_score))


def get_grader(task_id: str):
    """Factory function to get the appropriate grader for a task."""
    graders = {
        'single_label_classification': EasyGrader(),
        'priority_sort': MediumGrader(),
        'triage_and_respond': HardGrader()
    }
    return graders.get(task_id)
