"""FastAPI server for Email Triage Environment (HF Spaces entry point)."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
async def home():
    """Beautiful landing page for the Email Triage Environment."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Email Triage Environment - Scaler × Meta × PyTorch Hackathon</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; color: #333; }
            .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            .header { text-align: center; padding: 40px 0; color: white; }
            .header h1 { font-size: 3em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
            .header p { font-size: 1.2em; opacity: 0.9; }
            .card { background: white; border-radius: 15px; padding: 30px; margin: 20px 0; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
            .card h2 { color: #667eea; margin-bottom: 20px; font-size: 2em; }
            .endpoint { background: #f7f7f7; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #667eea; }
            .endpoint code { background: #667eea; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.9em; }
            .endpoint a { color: #667eea; text-decoration: none; font-weight: bold; }
            .endpoint a:hover { text-decoration: underline; }
            .badge { display: inline-block; background: #48bb78; color: white; padding: 5px 15px; border-radius: 20px; font-size: 0.9em; margin: 5px; }
            .task-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
            .task-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }
            .task-card h3 { font-size: 1.5em; margin-bottom: 10px; }
            .task-card .difficulty { display: inline-block; background: rgba(255,255,255,0.3); padding: 3px 10px; border-radius: 15px; font-size: 0.8em; margin: 5px 0; }
            .btn { display: inline-block; background: #667eea; color: white; padding: 12px 30px; border-radius: 8px; text-decoration: none; margin: 10px 5px; font-weight: bold; transition: transform 0.2s; }
            .btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); }
            .status-indicator { display: inline-block; width: 10px; height: 10px; background: #48bb78; border-radius: 50%; margin-right: 5px; animation: pulse 2s infinite; }
            @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
            .footer { text-align: center; padding: 30px 0; color: white; opacity: 0.8; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📧 Email Triage Environment</h1>
                <p>Production-ready OpenEnv for AI Agent Training</p>
                <p style="margin-top: 10px; font-size: 0.9em;">Scaler × Meta × PyTorch Hackathon 2026</p>
            </div>

            <div class="card">
                <h2><span class="status-indicator"></span>System Status</h2>
                <p style="font-size: 1.1em; margin: 15px 0;"><span class="badge">✅ Server Running</span><span class="badge">✅ Port 7860</span><span class="badge">✅ Docker Deployed</span></p>
                <div style="margin-top: 20px;">
                    <a href="/health" class="btn">🔍 Test Health Endpoint</a>
                    <a href="/docs" class="btn">📚 API Documentation</a>
                    <a href="/tasks" class="btn">📋 View Tasks</a>
                </div>
            </div>

            <div class="card">
                <h2>Available Tasks</h2>
                <div class="task-grid">
                    <div class="task-card">
                        <h3>🎯 Single Label Classification</h3>
                        <div class="difficulty">Easy</div>
                        <p style="margin-top: 10px;">Classify emails into 5 categories: spam, urgent, newsletter, support, inquiry</p>
                        <p style="margin-top: 10px; font-size: 0.9em; opacity: 0.9;">Max Steps: 1 | Scoring: Exact Match</p>
                    </div>
                    <div class="task-card">
                        <h3>📊 Priority Sort</h3>
                        <div class="difficulty">Medium</div>
                        <p style="margin-top: 10px;">Rank 5 emails by priority using Kendall's tau correlation</p>
                        <p style="margin-top: 10px; font-size: 0.9em; opacity: 0.9;">Max Steps: 5 | Scoring: Rank Correlation</p>
                    </div>
                    <div class="task-card">
                        <h3>🔥 Triage and Respond</h3>
                        <div class="difficulty">Hard</div>
                        <p style="margin-top: 10px;">Classify, action, and reply to 8 mixed emails with partial credit</p>
                        <p style="margin-top: 10px; font-size: 0.9em; opacity: 0.9;">Max Steps: 16 | Scoring: Weighted Sum</p>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>API Endpoints</h2>
                <div class="endpoint">
                    <code>GET /health</code><br>
                    <a href="/health">https://anushashigihalli-email-triage-env.hf.space/health</a>
                </div>
                <div class="endpoint">
                    <code>GET /tasks</code><br>
                    <a href="/tasks">https://anushashigihalli-email-triage-env.hf.space/tasks</a>
                </div>
                <div class="endpoint">
                    <code>POST /reset</code><br>
                    <span>Reset environment and get initial observation</span>
                </div>
                <div class="endpoint">
                    <code>POST /step</code><br>
                    <span>Execute action and get (observation, reward, done, info)</span>
                </div>
                <div class="endpoint">
                    <code>GET /state</code><br>
                    <span>Get current environment state</span>
                </div>
                <div class="endpoint">
                    <code>GET /docs</code><br>
                    <a href="/docs">Interactive Swagger UI Documentation</a>
                </div>
            </div>

            <div class="card">
                <h2>Quick Start</h2>
                <pre style="background: #2d3748; color: #f7f7f7; padding: 20px; border-radius: 8px; overflow-x: auto; margin: 15px 0;"><code># Test health endpoint
curl https://anushashigihalli-email-triage-env.hf.space/health

# List all tasks
curl https://anushashigihalli-email-triage-env.hf.space/tasks

# Reset environment
curl -X POST https://anushashigihalli-email-triage-env.hf.space/reset?task_id=single_label_classification

# Run inference locally
export HF_TOKEN=your_token
python inference.py</code></pre>
            </div>

            <div class="footer">
                <p>Built with ❤️ for Scaler × Meta × PyTorch Hackathon 2026</p>
                <p style="margin-top: 10px;">Powered by FastAPI, OpenEnv, and Pydantic</p>
            </div>
        </div>
    </body>
    </html>
    """


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
