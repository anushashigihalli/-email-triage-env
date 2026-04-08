"""FastAPI server for Email Triage Environment (HF Spaces entry point)."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import EmailTriageEnv
from env.models import Action, Observation

app = FastAPI(
    title="Email Triage Environment",
    description="Real-world email triage environment for AI agent training",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
env: Optional[EmailTriageEnv] = None


@app.on_event("startup")
async def startup_event():
    """Initialize environment on startup."""
    global env
    env = EmailTriageEnv()


@app.get("/health")
async def health_check():
    """Health check endpoint - must return HTTP 200."""
    return {"status": "ok"}


@app.get("/tasks")
async def list_tasks():
    """List all available tasks with metadata."""
    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")
    
    tasks = env.get_all_tasks()
    return {"tasks": tasks}


@app.post("/reset")
async def reset_environment(task_id: Optional[str] = None):
    """
    Reset the environment and return initial observation.
    
    Args:
        task_id: Optional task ID to reset to
    """
    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")
    
    try:
        observation = env.reset(task_id=task_id)
        return {
            "observation": observation.model_dump(),
            "task_id": env.current_task_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step")
async def step_environment(action: Action):
    """
    Execute one step in the environment.
    
    Args:
        action: Action object from the agent
    """
    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")
    
    try:
        observation, reward, done, info = env.step(action)
        
        return {
            "observation": observation.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.get("/state")
async def get_state():
    """Get current state of the environment."""
    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")
    
    state = env.state()
    return state


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
