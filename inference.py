"""Inference script for Email Triage Environment.

Runs all 3 tasks sequentially using an LLM agent and emits structured logs
in the exact format required by judges: [START]/[STEP]/[END].
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Optional
from openai import OpenAI

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import EmailTriageEnv
from env.models import Action, Observation, Email


# Configuration from environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)


def call_llm(prompt: str, max_tokens: int = 500) -> Optional[str]:
    """Call LLM with error handling."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] LLM call failed: {str(e)}", file=sys.stderr)
        return None


def parse_llm_response(response: str) -> Dict[str, Any]:
    """Parse LLM response into action dict."""
    try:
        # Try to extract JSON from response
        if '{' in response:
            start = response.index('{')
            end = response.rindex('}') + 1
            json_str = response[start:end]
            return json.loads(json_str)
    except:
        pass
    
    # Return empty action if parsing fails
    return {}


def run_easy_task(env: EmailTriageEnv) -> float:
    """Run Task 1: Single label classification."""
    task_id = "single_label_classification"
    episode = 1
    
    # Reset environment
    obs = env.reset(task_id=task_id)
    
    print(f'[START] {{"task_id": "{task_id}", "episode": {episode}}}')
    
    total_reward = 0.0
    done = False
    
    try:
        while not done:
            # Build prompt for LLM
            email = obs.emails[0]
            prompt = f"""Classify this email into ONE category: [spam, urgent, newsletter, support, inquiry]

Subject: {email.subject}
From: {email.sender}
Body: {email.body}

Respond with ONLY a JSON object like this:
{{"classification": "category_name"}}"""
            
            # Call LLM
            response = call_llm(prompt, max_tokens=100)
            
            if response:
                action_dict = parse_llm_response(response)
            else:
                action_dict = {"classification": "spam"}  # Fallback
            
            # Create action object
            action = Action(
                email_id=email.id,
                classification=action_dict.get("classification"),
                action=None,
                reply_text=None
            )
            
            # Step environment
            obs, reward, done, info = env.step(action)
            total_reward += reward.value
            
            # Print step log
            step_log = {
                "step": obs.current_step,
                "action": action_dict,
                "reward": round(reward.value, 2),
                "done": done,
                "obs": {
                    "emails_count": len(obs.emails),
                    "current_step": obs.current_step
                }
            }
            print(f'[STEP] {json.dumps(step_log)}')
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
    
    except Exception as e:
        print(f"[ERROR] Task failed: {str(e)}", file=sys.stderr)
        done = True
    
    success = total_reward >= 0.99
    end_log = {
        "task_id": task_id,
        "episode": episode,
        "total_reward": round(total_reward, 2),
        "success": success
    }
    print(f'[END] {json.dumps(end_log)}')
    
    return total_reward


def run_medium_task(env: EmailTriageEnv) -> float:
    """Run Task 2: Priority sort."""
    task_id = "priority_sort"
    episode = 1
    
    # Reset environment
    obs = env.reset(task_id=task_id)
    
    print(f'[START] {{"task_id": "{task_id}", "episode": {episode}}}')
    
    total_reward = 0.0
    done = False
    
    try:
        # Build prompt for all emails
        email_descriptions = []
        for i, email in enumerate(obs.emails, 1):
            email_descriptions.append(
                f"{i}. ID: {email.id}\n"
                f"   Subject: {email.subject}\n"
                f"   From: {email.sender}\n"
                f"   Body: {email.body[:200]}..."
            )
        
        prompt = f"""Rank these {len(obs.emails)} emails by priority (1=highest priority, 5=lowest).

Emails:
{chr(10).join(email_descriptions)}

Respond with ONLY a JSON object containing the email IDs in priority order:
{{"priority_ranking": ["email_id_1", "email_id_2", ...]}}

Where the first ID is the most urgent and the last is least urgent."""
        
        # Call LLM once for ranking
        response = call_llm(prompt, max_tokens=300)
        
        if response:
            action_dict = parse_llm_response(response)
        else:
            # Fallback: use original order
            action_dict = {
                "priority_ranking": [email.id for email in obs.emails]
            }
        
        # Submit ranking as single action
        action = Action(
            email_id=None,
            classification=None,
            action=None,
            reply_text=None,
            priority_ranking=action_dict.get("priority_ranking", [])
        )
        
        # Step environment
        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        
        # Print step log
        step_log = {
            "step": obs.current_step,
            "action": action_dict,
            "reward": round(reward.value, 2),
            "done": done,
            "obs": {
                "emails_count": len(obs.emails),
                "current_step": obs.current_step
            }
        }
        print(f'[STEP] {json.dumps(step_log)}')
        
        time.sleep(0.5)
    
    except Exception as e:
        print(f"[ERROR] Task failed: {str(e)}", file=sys.stderr)
        done = True
    
    success = total_reward >= 0.6
    end_log = {
        "task_id": task_id,
        "episode": episode,
        "total_reward": round(total_reward, 2),
        "success": success
    }
    print(f'[END] {json.dumps(end_log)}')
    
    return total_reward


def run_hard_task(env: EmailTriageEnv) -> float:
    """Run Task 3: Triage and respond."""
    task_id = "triage_and_respond"
    episode = 1
    
    # Reset environment
    obs = env.reset(task_id=task_id)
    
    print(f'[START] {{"task_id": "{task_id}", "episode": {episode}}}')
    
    total_reward = 0.0
    done = False
    step_count = 0
    
    try:
        while not done and step_count < obs.max_steps:
            # Process emails in pairs (classify + action per email)
            emails_remaining = len(obs.emails)
            email_idx = step_count // 2
            
            if email_idx >= emails_remaining:
                break
            
            email = obs.emails[email_idx]
            
            # Build prompt
            prompt = f"""For this email, provide:
1. Classification: [spam, urgent, newsletter, support, inquiry]
2. Action: [archive, escalate, reply, delete, forward]
3. If action is 'reply', draft a short response (≤50 words)

Subject: {email.subject}
From: {email.sender}
Body: {email.body}

Respond with ONLY a JSON object:
{{
  "classification": "category",
  "action": "action_type",
  "reply_text": "response text if action is reply, otherwise null"
}}"""
            
            # Call LLM
            response = call_llm(prompt, max_tokens=300)
            
            if response:
                action_dict = parse_llm_response(response)
            else:
                action_dict = {
                    "classification": "support",
                    "action": "archive",
                    "reply_text": None
                }
            
            # Ensure reply_text is None for non-reply actions
            if action_dict.get("action") != "reply":
                action_dict["reply_text"] = None
            
            # Create action object
            action = Action(
                email_id=email.id,
                classification=action_dict.get("classification"),
                action=action_dict.get("action"),
                reply_text=action_dict.get("reply_text")
            )
            
            # Step environment
            obs, reward, done, info = env.step(action)
            total_reward += reward.value
            step_count += 1
            
            # Print step log
            step_log = {
                "step": step_count,
                "action": action_dict,
                "reward": round(reward.value, 2),
                "done": done,
                "obs": {
                    "email_id": email.id,
                    "current_step": step_count
                }
            }
            print(f'[STEP] {json.dumps(step_log)}')
            
            # Small delay
            time.sleep(0.5)
            
            if done:
                break
    
    except Exception as e:
        print(f"[ERROR] Task failed: {str(e)}", file=sys.stderr)
        done = True
    
    success = total_reward >= 0.5
    end_log = {
        "task_id": task_id,
        "episode": episode,
        "total_reward": round(total_reward, 2),
        "success": success
    }
    print(f'[END] {json.dumps(end_log)}')
    
    return total_reward


def main():
    """Run all 3 tasks sequentially."""
    print("=" * 60)
    print("EMAIL TRIAGE ENVIRONMENT - INFERENCE")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"API Base: {API_BASE_URL}")
    print("=" * 60)
    print()
    
    # Initialize environment
    env = EmailTriageEnv()
    
    # Track total performance
    start_time = time.time()
    scores = {}
    
    try:
        # Task 1: Easy
        print("\n" + "=" * 60)
        print("TASK 1: Single Label Classification (Easy)")
        print("=" * 60)
        scores['single_label_classification'] = run_easy_task(env)
        
        time.sleep(1)
        
        # Task 2: Medium
        print("\n" + "=" * 60)
        print("TASK 2: Priority Sort (Medium)")
        print("=" * 60)
        scores['priority_sort'] = run_medium_task(env)
        
        time.sleep(1)
        
        # Task 3: Hard
        print("\n" + "=" * 60)
        print("TASK 3: Triage and Respond (Hard)")
        print("=" * 60)
        scores['triage_and_respond'] = run_hard_task(env)
        
    except Exception as e:
        print(f"\n[ERROR] Inference failed: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Scores: {json.dumps({k: round(v, 2) for k, v in scores.items()})}")
    print("=" * 60)


if __name__ == "__main__":
    main()
