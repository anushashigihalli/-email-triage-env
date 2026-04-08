---
title: Email Triage Environment
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# Email Triage Environment

A production-ready OpenEnv environment for the Scaler × Meta × PyTorch Hackathon (Round 1). This environment simulates a **real-world email triage task** where an AI agent reads incoming emails and must correctly classify, prioritize, and respond to them.

## 🎯 Real-World Motivation

In modern organizations, employees receive hundreds of emails daily that require:
- **Classification** (spam, urgent, newsletter, support, inquiry)
- **Prioritization** (what needs immediate attention vs. what can wait)
- **Action decisions** (archive, escalate, reply, delete, forward)
- **Response drafting** (for emails requiring replies)

This environment provides a realistic testbed for training and evaluating AI agents on this critical productivity task.

---

## 📋 Observation Space

The observation space is a structured object containing:

| Field | Type | Description |
|-------|------|-------------|
| `emails` | `List[Email]` | List of email objects to process |
| `current_step` | `int` | Current step number in the episode |
| `max_steps` | `int` | Maximum allowed steps for the task |
| `task_id` | `str` | Identifier for the current task |
| `step_history` | `List[Dict]` | History of actions and rewards |

### Email Object Structure

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique email identifier |
| `subject` | `str` | Email subject line |
| `body` | `str` | Full email body text |
| `sender` | `str` | Sender email address |
| `timestamp` | `str` | ISO 8601 timestamp |

---

## 🎮 Action Space

The action space allows the agent to perform structured actions:

| Field | Type | Valid Values | Description |
|-------|------|--------------|-------------|
| `email_id` | `str` | Any valid email ID | Target email for the action |
| `classification` | `str` | `spam`, `urgent`, `newsletter`, `support`, `inquiry` | Email category |
| `action` | `str` | `archive`, `escalate`, `reply`, `delete`, `forward` | Action to take |
| `reply_text` | `str` (optional) | Any text ≤50 words | Drafted response (if action=reply) |
| `priority_ranking` | `List[str]` | List of email IDs | Ordered by priority (for medium task) |

---

## 📚 Task Descriptions

### Task 1: Single Label Classification (Easy)

**Objective:** Classify a single email into one of 5 categories.

- **Input:** 1 email
- **Output:** Classification label
- **Max Steps:** 1
- **Scoring:** Exact match → 1.0, incorrect → 0.0

**Categories:**
- `spam` - Unsolicited or malicious emails
- `urgent` - Time-sensitive, critical issues
- `newsletter` - Informational updates, marketing
- `support` - Customer help requests
- `inquiry` - Business questions, partnership requests

**Expected Difficulty:** Easy  
**Baseline Score:** ~0.80

---

### Task 2: Priority Sort (Medium)

**Objective:** Rank 5 emails by priority (1=highest priority).

- **Input:** 5 emails
- **Output:** Ordered list of email IDs by priority
- **Max Steps:** 5
- **Scoring:** Kendall's tau rank correlation normalized to [0.0, 1.0]
- **Partial Credit:** Yes, for partially correct orderings

**Scoring Breakdown:**
- Perfect ranking: 1.0
- Random ranking: ~0.5
- Reverse ranking: 0.0

**Expected Difficulty:** Medium  
**Baseline Score:** ~0.65

---

### Task 3: Triage and Respond (Hard)

**Objective:** Classify, decide actions, and draft replies for 8 mixed emails.

- **Input:** 8 emails of various types
- **Output per email:**
  1. Classification (spam/urgent/newsletter/support/inquiry)
  2. Action (archive/escalate/reply/delete/forward)
  3. Reply text (if action=reply, ≤50 words)
- **Max Steps:** 16 (2 steps per email)
- **Scoring:** Weighted sum of sub-scores

**Scoring Breakdown:**

| Component | Weight | Description |
|-----------|--------|-------------|
| Classification Accuracy | 0.4 | Correct category for each email |
| Action Correctness | 0.4 | Correct action for each email |
| Reply Quality | 0.2 | Keyword matching in drafted replies |

**Final Score Calculation:**
```
score = (0.4 × class_accuracy) + (0.4 × action_accuracy) + (0.2 × reply_quality)
```

**Expected Difficulty:** Hard  
**Baseline Score:** ~0.45

---

## 📊 Baseline Scores

| Task | Difficulty | Baseline Score |
|------|------------|----------------|
| single_label_classification | Easy | ~0.80 |
| priority_sort | Medium | ~0.65 |
| triage_and_respond | Hard | ~0.45 |

---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.11+
- Hugging Face API token (for LLM inference)

### Local Setup

```bash
# Clone or navigate to the project directory
cd email_triage_env

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/test_env.py -v
```

### Docker Setup

```bash
# Build the Docker image
docker build -t email-triage-env .

# Run the container
docker run -p 7860:7860 email-triage-env

# The server will be available at http://localhost:7860
```

### Hugging Face Spaces Deployment

1. Create a new Space on Hugging Face
2. Select "Docker" as the SDK
3. Upload all project files
4. Set environment variables in Space settings:
   - `API_BASE_URL`: Your LLM API endpoint
   - `MODEL_NAME`: Model identifier
   - `HF_TOKEN`: Your Hugging Face token

---

## 🔧 Running Inference

### Environment Variables

Before running inference, set these environment variables:

```bash
# For Hugging Face Inference API
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_huggingface_token_here"

# For OpenAI-compatible endpoints
export API_BASE_URL="https://your-endpoint.com/v1"
export MODEL_NAME="your-model-name"
export HF_TOKEN="your_api_key_here"
```

### Run Inference Script

```bash
python inference.py
```

The script will:
1. Run all 3 tasks sequentially
2. Emit structured logs in `[START]`, `[STEP]`, `[END]` format
3. Print final scores and execution time

### Example Output

```
[START] {"task_id": "single_label_classification", "episode": 1}
[STEP] {"step": 1, "action": {"classification": "spam"}, "reward": 1.0, "done": true, "obs": {...}}
[END] {"task_id": "single_label_classification", "episode": 1, "total_reward": 1.0, "success": true}

[START] {"task_id": "priority_sort", "episode": 1}
[STEP] {"step": 1, "action": {...}, "reward": 0.72, "done": true, "obs": {...}}
[END] {"task_id": "priority_sort", "episode": 1, "total_reward": 0.72, "success": true}

[START] {"task_id": "triage_and_respond", "episode": 1}
...
[END] {"task_id": "triage_and_respond", "episode": 1, "total_reward": 0.55, "success": false}
```

---

## 🌐 API Endpoints

When running the FastAPI server, the following endpoints are available:

### Health Check
```bash
curl http://localhost:7860/health
```
**Response:** `{"status": "ok"}`

### List Tasks
```bash
curl http://localhost:7860/tasks
```

### Reset Environment
```bash
curl -X POST "http://localhost:7860/reset?task_id=single_label_classification"
```

### Step Environment
```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "email_id": "easy_001",
    "classification": "spam",
    "action": "delete"
  }'
```

### Get State
```bash
curl http://localhost:7860/state
```

---

## 📁 Project Structure

```
email_triage_env/
├── openenv.yaml              # OpenEnv configuration
├── Dockerfile                # Docker deployment config
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── inference.py              # LLM inference script
├── app.py                    # FastAPI server (HF Spaces entry point)
├── env/
│   ├── __init__.py           # Package initialization
│   ├── environment.py        # Core OpenEnv environment class
│   ├── models.py             # Pydantic typed models
│   ├── tasks.py              # Task definitions and data loading
│   ├── graders.py            # Deterministic graders for each task
│   ├── reward.py             # Reward function with partial signals
│   └── data/
│       ├── easy_emails.json      # 10 emails for easy task
│       ├── medium_emails.json    # 5 sets of 5 emails for medium task
│       └── hard_emails.json      # 3 batches of 8 emails for hard task
└── tests/
    └── test_env.py           # Comprehensive test suite
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/test_env.py -v

# Run specific test class
pytest tests/test_env.py::TestEasyGrader -v

# Run with coverage
pytest tests/test_env.py --cov=env --cov-report=html
```

---

## 🏗️ Architecture

### Environment Flow

```
1. reset(task_id) → Initial Observation
2. step(action) → (Observation, Reward, Done, Info)
3. Repeat step() until done=True
4. state() → Current environment state
```

### Reward System

The reward system provides **partial credit** at each step:

- **Correct classification:** +0.1
- **Correct action:** +0.1
- **Repeated action penalty:** -0.05
- **No-op/empty action penalty:** -0.1
- **Final reward:** Normalized to [0.0, 1.0]

---

## 📝 Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `API_BASE_URL` | Base URL for LLM API | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `meta-llama/Llama-3.1-8B-Instruct` |
| `HF_TOKEN` | API key/token for authentication | `hf_xxxxxxxxxxxxxxxxxxxx` |

**Note:** Never hardcode API keys in source code. Always use environment variables.

---

## ✅ Validation Checklist

Before submission, verify:

- [ ] `curl http://localhost:7860/health` → HTTP 200
- [ ] `POST /reset` → Valid Observation JSON
- [ ] `openenv validate openenv.yaml` → Passes
- [ ] `docker build .` → Exits 0
- [ ] `python inference.py` → Completes with [START]/[STEP]/[END] logs
- [ ] All grader scores in [0.0, 1.0]
- [ ] No hardcoded API keys
- [ ] All rewards normalized to [0.0, 1.0]

---



