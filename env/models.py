"""Pydantic typed models for Email Triage Environment."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class EmailCategory(str, Enum):
    """Valid email classification categories."""
    SPAM = "spam"
    URGENT = "urgent"
    NEWSLETTER = "newsletter"
    SUPPORT = "support"
    INQUIRY = "inquiry"


class EmailAction(str, Enum):
    """Valid actions for email triage."""
    ARCHIVE = "archive"
    ESCALATE = "escalate"
    REPLY = "reply"
    DELETE = "delete"
    FORWARD = "forward"


class Email(BaseModel):
    """Represents a single email."""
    id: str
    subject: str
    body: str
    sender: str
    timestamp: str


class EmailWithGroundTruth(Email):
    """Email with ground truth labels for evaluation."""
    true_category: EmailCategory
    true_priority: int = Field(ge=1, le=5, description="Priority rank (1=highest)")
    true_action: EmailAction
    reply_keywords: Optional[List[str]] = None


class Observation(BaseModel):
    """Observation returned by the environment."""
    emails: List[Email]
    current_step: int
    max_steps: int
    task_id: str
    step_history: List[Dict[str, Any]] = Field(default_factory=list)


class Action(BaseModel):
    """Action taken by the agent."""
    email_id: Optional[str] = None
    classification: Optional[EmailCategory] = None
    action: Optional[EmailAction] = None
    reply_text: Optional[str] = None
    priority_ranking: Optional[List[str]] = None


class Reward(BaseModel):
    """Reward signal from the environment."""
    value: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)
    message: str = ""


class StepResult(BaseModel):
    """Result of a step: (observation, reward, done, info)."""
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class TaskInfo(BaseModel):
    """Metadata about a task."""
    id: str
    description: str
    difficulty: str
    max_steps: int
    reward_range: List[float]


class EnvironmentState(BaseModel):
    """Current state of the environment."""
    task_id: Optional[str] = None
    current_step: int = 0
    total_reward: float = 0.0
    emails_processed: int = 0
    done: bool = False
    step_history: List[Dict[str, Any]] = Field(default_factory=list)
