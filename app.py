"""
AI Customer Support RL Environment — OpenEnv compliant FastAPI app.
"""
import os
import json
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.env import CustomerSupportEnv
from env.models import Action, TicketState, Reward
from env.grader import grade_trajectory
from env.tasks import TASKS

app = FastAPI(title="AI Customer Support RL Environment", version="1.0.0")

# CORS — required for HuggingFace Space iframe embeds
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Debug helper — surfaces 422 errors clearly
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    import sys
    print(f"Validation Error at {request.url.path}: {exc.errors()}", file=sys.stderr)
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory session store
envs: Dict[str, CustomerSupportEnv] = {}


class StepRequest(BaseModel):
    session_id: str
    action: Action


# ---------------------------------------------------------------------------
# Core Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "ok", "environment": "AI Customer Support RL Environment"}


@app.post("/reset")
async def reset(request: Request) -> Dict[str, Any]:
    """
    OpenEnv /reset endpoint.
    Accepts POST with no body, empty body, or JSON body {"task_id": "..."}.
    The OpenEnv checker sends a POST with NO body — we must handle that gracefully.
    """
    task_id = "easy_refund"
    try:
        body_bytes = await request.body()
        if body_bytes:
            body_json = json.loads(body_bytes)
            task_id = body_json.get("task_id", "easy_refund")
    except Exception:
        pass  # No body / invalid JSON → use default

    session_id = os.urandom(8).hex()
    env = CustomerSupportEnv(task_id=task_id)
    state = env.reset()
    envs[session_id] = env
    return {"session_id": session_id, "state": state.model_dump()}


@app.post("/step")
async def step(req: StepRequest) -> Dict[str, Any]:
    if req.session_id not in envs:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")

    env = envs[req.session_id]
    state, reward = env.step(req.action)

    score = None
    if reward.done:
        score = grade_trajectory(env.task_id, state.history)
        del envs[req.session_id]

    return {
        "state": state.model_dump(),
        "reward": reward.model_dump(),
        "score": score,
    }


@app.get("/state")
def get_state(session_id: str) -> Dict[str, Any]:
    if session_id not in envs:
        raise HTTPException(status_code=404, detail="Session not found.")
    return envs[session_id].get_state().model_dump()


@app.get("/tasks")
def get_tasks() -> Dict[str, Any]:
    return {
        task_id: {
            "ticket_text": data["ticket_text"],
            "customer_history": data["customer_history"],
            "difficulty": data["difficulty"],
            "expected_sequence": data["expected_sequence"],
        }
        for task_id, data in TASKS.items()
    }


@app.get("/validate")
def validate() -> Dict[str, Any]:
    """Quick sanity check used by openenv validate."""
    results = []
    for task_id in TASKS:
        env = CustomerSupportEnv(task_id=task_id)
        state = env.reset()
        seq = TASKS[task_id]["expected_sequence"]
        rewards = []
        for action_type in seq:
            payload = TASKS[task_id].get("expected_category", "") if action_type == "classify_issue" else None
            _, reward = env.step(Action(action_type=action_type, payload=payload))
            rewards.append(reward.value)
        score = grade_trajectory(task_id, env.state.history)
        results.append({"task_id": task_id, "score": score, "rewards": rewards})
    return {"status": "ok", "results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
