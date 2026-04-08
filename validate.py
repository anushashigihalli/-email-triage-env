"""Quick validation script to verify all components work."""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from env.environment import EmailTriageEnv
from env.models import Action
from env.graders import EasyGrader, MediumGrader, HardGrader


def test_environment():
    """Test basic environment functionality."""
    print("=" * 60)
    print("EMAIL TRIAGE ENVIRONMENT - VALIDATION")
    print("=" * 60)
    
    # Initialize environment
    env = EmailTriageEnv()
    
    # Test 1: Easy Task
    print("\n✓ Testing Task 1: Single Label Classification")
    obs = env.reset(task_id="single_label_classification")
    print(f"  - Emails loaded: {len(obs.emails)}")
    print(f"  - Task ID: {obs.task_id}")
    print(f"  - Max steps: {obs.max_steps}")
    
    # Make a test action
    action = Action(
        email_id=obs.emails[0].id,
        classification="spam",
        action="delete"
    )
    
    next_obs, reward, done, info = env.step(action)
    print(f"  - Step reward: {reward.value:.2f}")
    print(f"  - Episode done: {done}")
    assert done is True, "Easy task should complete in 1 step"
    print("  ✓ Task 1 PASSED")
    
    # Test 2: Medium Task
    print("\n✓ Testing Task 2: Priority Sort")
    obs = env.reset(task_id="priority_sort")
    print(f"  - Emails loaded: {len(obs.emails)}")
    print(f"  - Task ID: {obs.task_id}")
    
    # Make a test ranking action
    email_ids = [email.id for email in obs.emails]
    action = Action(
        priority_ranking=email_ids
    )
    
    next_obs, reward, done, info = env.step(action)
    print(f"  - Step reward: {reward.value:.2f}")
    print(f"  - Episode done: {done}")
    print(f"  - Current step: {next_obs.current_step}")
    print("  ✓ Task 2 PASSED")
    
    # Test 3: Hard Task
    print("\n✓ Testing Task 3: Triage and Respond")
    obs = env.reset(task_id="triage_and_respond")
    print(f"  - Emails loaded: {len(obs.emails)}")
    print(f"  - Task ID: {obs.task_id}")
    print(f"  - Max steps: {obs.max_steps}")
    
    # Make a test action for first email
    action = Action(
        email_id=obs.emails[0].id,
        classification="urgent",
        action="escalate",
        reply_text="Checking the issue now"
    )
    
    next_obs, reward, done, info = env.step(action)
    print(f"  - Step reward: {reward.value:.2f}")
    print(f"  - Episode done: {done}")
    print("  ✓ Task 3 PASSED")
    
    # Test 4: State
    print("\n✓ Testing State Retrieval")
    state = env.state()
    print(f"  - Task ID: {state['task_id']}")
    print(f"  - Current step: {state['current_step']}")
    print(f"  - Total reward: {state['total_reward']:.2f}")
    print("  ✓ State Retrieval PASSED")
    
    # Test 5: Graders
    print("\n✓ Testing Graders")
    
    # Easy grader
    easy_grader = EasyGrader()
    score = easy_grader.grade(
        {"classification": "spam"},
        {"true_category": "spam"}
    )
    assert score == 1.0, "Easy grader should give 1.0 for correct classification"
    print(f"  - Easy grader (correct): {score:.2f} ✓")
    
    score = easy_grader.grade(
        {"classification": "spam"},
        {"true_category": "urgent"}
    )
    assert score == 0.0, "Easy grader should give 0.0 for incorrect classification"
    print(f"  - Easy grader (incorrect): {score:.2f} ✓")
    
    # Medium grader
    medium_grader = MediumGrader()
    score = medium_grader.grade(
        {"priority_ranking": ["email_1", "email_2", "email_3"]},
        {"email_priorities": {"email_1": 1, "email_2": 2, "email_3": 3}}
    )
    assert score == 1.0, "Medium grader should give 1.0 for perfect ranking"
    print(f"  - Medium grader (perfect): {score:.2f} ✓")
    
    # Hard grader
    hard_grader = HardGrader()
    score = hard_grader.grade(
        {
            "classification": "urgent",
            "action": "escalate",
            "reply_text": "Checking server issue now"
        },
        {
            "true_category": "urgent",
            "true_action": "escalate",
            "reply_keywords": ["server", "checking"]
        }
    )
    assert score == 1.0, "Hard grader should give 1.0 for perfect triage"
    print(f"  - Hard grader (perfect): {score:.2f} ✓")
    
    print("  ✓ All Graders PASSED")
    
    # Test 6: Task Metadata
    print("\n✓ Testing Task Metadata")
    tasks = env.get_all_tasks()
    print(f"  - Total tasks: {len(tasks)}")
    for task in tasks:
        print(f"    • {task['id']} ({task['difficulty']}): {task['max_steps']} max steps")
    print("  ✓ Task Metadata PASSED")
    
    print("\n" + "=" * 60)
    print("ALL VALIDATION TESTS PASSED ✓")
    print("=" * 60)
    print("\nThe environment is ready for:")
    print("  1. Local testing: python inference.py")
    print("  2. Docker deployment: docker build -t email-triage-env .")
    print("  3. HF Spaces deployment: Upload all files")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_environment()
    except Exception as e:
        print(f"\n✗ VALIDATION FAILED: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
