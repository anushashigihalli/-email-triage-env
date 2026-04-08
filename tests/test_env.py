"""Tests for Email Triage Environment."""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import EmailTriageEnv
from env.models import Action, Observation, Email
from env.graders import EasyGrader, MediumGrader, HardGrader
from env.reward import RewardCalculator
from env.tasks import TaskManager


class TestEasyGrader:
    """Test easy task grader."""
    
    def test_correct_classification(self):
        grader = EasyGrader()
        action = {"classification": "spam"}
        ground_truth = {"true_category": "spam"}
        assert grader.grade(action, ground_truth) == 1.0
    
    def test_incorrect_classification(self):
        grader = EasyGrader()
        action = {"classification": "spam"}
        ground_truth = {"true_category": "urgent"}
        assert grader.grade(action, ground_truth) == 0.0
    
    def test_empty_action(self):
        grader = EasyGrader()
        action = {}
        ground_truth = {"true_category": "spam"}
        assert grader.grade(action, ground_truth) == 0.0
    
    def test_case_insensitive(self):
        grader = EasyGrader()
        action = {"classification": "SPAM"}
        ground_truth = {"true_category": "spam"}
        assert grader.grade(action, ground_truth) == 1.0


class TestMediumGrader:
    """Test medium task grader."""
    
    def test_perfect_ranking(self):
        grader = MediumGrader()
        action = {"priority_ranking": ["email_1", "email_2", "email_3"]}
        ground_truth = {
            "email_priorities": {
                "email_1": 1,
                "email_2": 2,
                "email_3": 3
            }
        }
        score = grader.grade(action, ground_truth)
        assert score == 1.0
    
    def test_reverse_ranking(self):
        grader = MediumGrader()
        action = {"priority_ranking": ["email_3", "email_2", "email_1"]}
        ground_truth = {
            "email_priorities": {
                "email_1": 1,
                "email_2": 2,
                "email_3": 3
            }
        }
        score = grader.grade(action, ground_truth)
        assert score == 0.0
    
    def test_empty_ranking(self):
        grader = MediumGrader()
        action = {"priority_ranking": []}
        ground_truth = {"email_priorities": {"email_1": 1}}
        assert grader.grade(action, ground_truth) == 0.0


class TestHardGrader:
    """Test hard task grader."""
    
    def test_perfect_triage(self):
        grader = HardGrader()
        action = {
            "classification": "urgent",
            "action": "escalate",
            "reply_text": "We're checking the server issue now"
        }
        ground_truth = {
            "true_category": "urgent",
            "true_action": "escalate",
            "reply_keywords": ["server", "checking"]
        }
        score = grader.grade(action, ground_truth)
        assert score == 1.0
    
    def test_partial_triage(self):
        grader = HardGrader()
        action = {
            "classification": "urgent",
            "action": "archive",
            "reply_text": None
        }
        ground_truth = {
            "true_category": "urgent",
            "true_action": "escalate",
            "reply_keywords": ["urgent"]
        }
        score = grader.grade(action, ground_truth)
        assert 0.3 <= score <= 0.5  # Only classification correct
    
    def test_empty_action(self):
        grader = HardGrader()
        action = {}
        ground_truth = {"true_category": "spam", "true_action": "delete"}
        assert grader.grade(action, ground_truth) == 0.0


class TestRewardCalculator:
    """Test reward calculation."""
    
    def test_correct_action_reward(self):
        calc = RewardCalculator()
        action = {"classification": "spam", "action": "delete"}
        ground_truth = {"true_category": "spam", "true_action": "delete"}
        
        result = calc.calculate_step_reward(action, ground_truth, "triage_and_respond")
        assert result['value'] == 0.2  # 0.1 for classification + 0.1 for action
    
    def test_noop_penalty(self):
        calc = RewardCalculator()
        action = {}
        ground_truth = {"true_category": "spam"}
        
        result = calc.calculate_step_reward(action, ground_truth, "easy")
        assert result['value'] == 0.0  # Penalized but clamped to 0
    
    def test_final_reward_normalization(self):
        calc = RewardCalculator()
        calc.step_rewards = [0.2, 0.2, 0.1]
        
        final = calc.calculate_final_reward(max_possible_reward=1.0)
        assert 0.0 <= final <= 1.0
    
    def test_reset(self):
        calc = RewardCalculator()
        calc.step_rewards = [0.1, 0.2]
        calc.reset()
        
        assert len(calc.step_rewards) == 0
        assert calc.previous_action is None


class TestTaskManager:
    """Test task management."""
    
    def test_load_tasks(self):
        manager = TaskManager()
        tasks = manager.get_all_tasks()
        assert len(tasks) == 3
    
    def test_sample_easy(self):
        manager = TaskManager()
        task_data = manager.sample_easy_task()
        
        assert 'emails' in task_data
        assert 'ground_truth' in task_data
        assert len(task_data['emails']) == 1
    
    def test_sample_medium(self):
        manager = TaskManager()
        task_data = manager.sample_medium_task()
        
        assert 'emails' in task_data
        assert len(task_data['emails']) == 5
    
    def test_sample_hard(self):
        manager = TaskManager()
        task_data = manager.sample_hard_task()
        
        assert 'emails' in task_data
        assert len(task_data['emails']) == 8


class TestEmailTriageEnv:
    """Test environment integration."""
    
    def test_reset_returns_observation(self):
        env = EmailTriageEnv()
        obs = env.reset(task_id="single_label_classification")
        
        assert isinstance(obs, Observation)
        assert len(obs.emails) == 1
        assert obs.task_id == "single_label_classification"
        assert obs.current_step == 0
    
    def test_step_returns_tuple(self):
        env = EmailTriageEnv()
        obs = env.reset(task_id="single_label_classification")
        
        action = Action(
            email_id=obs.emails[0].id,
            classification="spam",
            action="delete"
        )
        
        result = env.step(action)
        assert len(result) == 4
        
        next_obs, reward, done, info = result
        assert isinstance(next_obs, Observation)
        assert 0.0 <= reward.value <= 1.0
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    
    def test_state_returns_dict(self):
        env = EmailTriageEnv()
        env.reset(task_id="single_label_classification")
        
        state = env.state()
        assert isinstance(state, dict)
        assert 'task_id' in state
        assert 'current_step' in state
        assert 'total_reward' in state
    
    def test_episode_completes(self):
        env = EmailTriageEnv()
        obs = env.reset(task_id="single_label_classification")
        
        action = Action(
            email_id=obs.emails[0].id,
            classification="spam",
            action="delete"
        )
        
        next_obs, reward, done, info = env.step(action)
        assert done is True
    
    def test_all_tasks_runnable(self):
        env = EmailTriageEnv()
        
        for task_id in ['single_label_classification', 'priority_sort', 'triage_and_respond']:
            obs = env.reset(task_id=task_id)
            assert obs.task_id == task_id
            assert len(obs.emails) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
