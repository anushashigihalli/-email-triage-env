"""Core OpenEnv environment class for Email Triage."""

from typing import Tuple, Dict, Any, Optional, List
from env.models import Observation, Action, Reward, Email
from env.tasks import TaskManager
from env.graders import get_grader
from env.reward import RewardCalculator


class EmailTriageEnv:
    """
    Email Triage Environment implementing OpenEnv interface.
    
    Simulates a real-world email triage task where an AI agent reads incoming emails
    and must correctly classify, prioritize, and respond to them.
    """
    
    def __init__(self, task_id: Optional[str] = None):
        """
        Initialize the environment.
        
        Args:
            task_id: Specific task to run, or None for manual selection
        """
        self.task_manager = TaskManager()
        self.reward_calculator = RewardCalculator()
        self.current_task_id = task_id
        
        # State variables
        self.current_observation: Optional[Observation] = None
        self.current_step = 0
        self.total_reward = 0.0
        self.done = False
        self.emails_data: List[Dict[str, Any]] = []
        self.ground_truth: Dict[str, Any] = {}
        self.step_history: List[Dict[str, Any]] = []
        self.processed_emails: set = set()
    
    def reset(self, task_id: Optional[str] = None) -> Observation:
        """
        Reset the environment and return initial observation.
        
        Args:
            task_id: Task to run (overrides constructor task_id if provided)
            
        Returns:
            Initial Observation object
        """
        # Determine task
        tid = task_id or self.current_task_id or 'single_label_classification'
        self.current_task_id = tid
        
        # Reset state
        self.current_step = 0
        self.total_reward = 0.0
        self.done = False
        self.step_history = []
        self.processed_emails = set()
        self.reward_calculator.reset()
        
        # Sample task data
        task_data = self.task_manager.sample_task(tid)
        self.emails_data = task_data['emails']
        self.ground_truth = task_data['ground_truth']
        
        # Create observation
        emails = [Email(**{k: v for k, v in e.items() if k in Email.model_fields}) 
                  for e in self.emails_data]
        
        task_info = self.task_manager.get_task_info(tid)
        
        self.current_observation = Observation(
            emails=emails,
            current_step=0,
            max_steps=task_info.max_steps,
            task_id=tid,
            step_history=[]
        )
        
        return self.current_observation
    
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action object from the agent
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        
        # Increment step counter
        self.current_step += 1
        
        # Convert action to dict for processing
        action_dict = action.model_dump(exclude_none=True)
        
        # Get grader for current task
        grader = get_grader(self.current_task_id)
        
        # Calculate reward based on task type
        if self.current_task_id == 'single_label_classification':
            reward_value = self._grade_easy_task(action_dict)
        elif self.current_task_id == 'priority_sort':
            reward_value = self._grade_medium_task(action_dict)
        elif self.current_task_id == 'triage_and_respond':
            reward_value = self._grade_hard_task(action_dict)
        else:
            reward_value = 0.0
        
        # Create reward object
        reward = Reward(
            value=reward_value,
            breakdown={'step_reward': reward_value},
            message=f"Step {self.current_step}"
        )
        
        # Update total reward
        self.total_reward += reward_value
        
        # Track step history
        step_info = {
            'step': self.current_step,
            'action': action_dict,
            'reward': reward_value
        }
        self.step_history.append(step_info)
        
        # Check if episode is done
        task_info = self.task_manager.get_task_info(self.current_task_id)
        self.done = (self.current_step >= task_info.max_steps)
        
        # Create next observation
        next_obs = Observation(
            emails=self.current_observation.emails,
            current_step=self.current_step,
            max_steps=task_info.max_steps,
            task_id=self.current_task_id,
            step_history=self.step_history.copy()
        )
        
        self.current_observation = next_obs
        
        # Info dict
        info = {
            'task_id': self.current_task_id,
            'step': self.current_step,
            'total_reward': self.total_reward
        }
        
        return next_obs, reward, self.done, info
    
    def _grade_easy_task(self, action_dict: Dict[str, Any]) -> float:
        """Grade easy task (single label classification)."""
        grader = get_grader('single_label_classification')
        score = grader.grade(action_dict, self.ground_truth)
        return score
    
    def _grade_medium_task(self, action_dict: Dict[str, Any]) -> float:
        """Grade medium task (priority sort)."""
        grader = get_grader('priority_sort')
        score = grader.grade(action_dict, self.ground_truth)
        return score
    
    def _grade_hard_task(self, action_dict: Dict[str, Any]) -> float:
        """Grade hard task (triage and respond)."""
        email_id = action_dict.get('email_id')
        if email_id and email_id in self.ground_truth:
            gt = self.ground_truth[email_id]
            grader = get_grader('triage_and_respond')
            score = grader.grade(action_dict, gt)
            self.processed_emails.add(email_id)
            return score
        return 0.0
    
    def state(self) -> Dict[str, Any]:
        """
        Get current state of the environment.
        
        Returns:
            Dict with current state information
        """
        return {
            'task_id': self.current_task_id,
            'current_step': self.current_step,
            'total_reward': self.total_reward,
            'emails_processed': len(self.processed_emails),
            'done': self.done,
            'step_history': self.step_history
        }
    
    def get_task_info(self, task_id: str) -> Dict[str, Any]:
        """Get metadata for a specific task."""
        task = self.task_manager.get_task_info(task_id)
        return task.dict()
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get metadata for all tasks."""
        return [task.dict() for task in self.task_manager.get_all_tasks()]
