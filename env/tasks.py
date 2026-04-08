"""Task definitions for Email Triage Environment."""

import json
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
from env.models import Email, TaskInfo


class TaskManager:
    """Manages task definitions and email data loading."""
    
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            data_dir = Path(__file__).parent / "data"
        else:
            data_dir = Path(data_dir)
        
        self.data_dir = data_dir
        self.easy_emails = self._load_json("easy_emails.json")
        self.medium_emails = self._load_json("medium_emails.json")
        self.hard_emails = self._load_json("hard_emails.json")
        
        self.tasks = {
            'single_label_classification': TaskInfo(
                id='single_label_classification',
                description='Classify a single email into one of 5 categories',
                difficulty='easy',
                max_steps=1,
                reward_range=[0.0, 1.0]
            ),
            'priority_sort': TaskInfo(
                id='priority_sort',
                description='Rank 5 emails by priority',
                difficulty='medium',
                max_steps=5,
                reward_range=[0.0, 1.0]
            ),
            'triage_and_respond': TaskInfo(
                id='triage_and_respond',
                description='Classify, action, and reply to 8 mixed emails',
                difficulty='hard',
                max_steps=16,
                reward_range=[0.0, 1.0]
            )
        }
    
    def _load_json(self, filename: str) -> Any:
        """Load JSON data file."""
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_task_info(self, task_id: str) -> TaskInfo:
        """Get task metadata."""
        if task_id not in self.tasks:
            raise ValueError(f"Unknown task: {task_id}")
        return self.tasks[task_id]
    
    def get_all_tasks(self) -> List[TaskInfo]:
        """Get all task definitions."""
        return list(self.tasks.values())
    
    def sample_easy_task(self) -> Dict[str, Any]:
        """Sample a single email for easy task."""
        email = random.choice(self.easy_emails)
        return {
            'emails': [email],
            'ground_truth': {
                'true_category': email['true_category'],
                'true_priority': email['true_priority'],
                'true_action': email['true_action'],
                'reply_keywords': email.get('reply_keywords', [])
            }
        }
    
    def sample_medium_task(self) -> Dict[str, Any]:
        """Sample a set of 5 emails for medium task."""
        email_set = random.choice(self.medium_emails)
        emails = email_set['emails']
        
        # Build ground truth with priorities
        email_priorities = {email['id']: email['true_priority'] for email in emails}
        
        return {
            'emails': emails,
            'ground_truth': {
                'email_priorities': email_priorities,
                'emails': emails
            }
        }
    
    def sample_hard_task(self) -> Dict[str, Any]:
        """Sample a batch of 8 emails for hard task."""
        email_batch = random.choice(self.hard_emails)
        emails = email_batch['emails']
        
        # Build ground truth for each email
        ground_truth = {email['id']: {
            'true_category': email['true_category'],
            'true_priority': email['true_priority'],
            'true_action': email['true_action'],
            'reply_keywords': email.get('reply_keywords', [])
        } for email in emails}
        
        return {
            'emails': emails,
            'ground_truth': ground_truth
        }
    
    def sample_task(self, task_id: str) -> Dict[str, Any]:
        """Sample data for a specific task."""
        if task_id == 'single_label_classification':
            return self.sample_easy_task()
        elif task_id == 'priority_sort':
            return self.sample_medium_task()
        elif task_id == 'triage_and_respond':
            return self.sample_hard_task()
        else:
            raise ValueError(f"Unknown task: {task_id}")
