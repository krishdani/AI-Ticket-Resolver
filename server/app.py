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
    state, reward_val, done, info = env.step(req.action)

    score = None
    if done:
        score = grade_trajectory(env.task_id, state.history)
        # Keep env for a bit or delete? Usually keep for /state check unless it's a one-shot.
        # del envs[req.session_id] 

    return {
        "state": state.model_dump(),
        "reward": {"value": reward_val, "reason": info.get("reason", ""), "done": done},
        "done": done,
        "score": score,
    }


class RaiseTicketRequest(BaseModel):
    session_id: str
    subject: str
    body: str
    customer_history: str
    priority: str
    category: str


@app.post("/raise_ticket")
async def raise_ticket(req: RaiseTicketRequest) -> Dict[str, Any]:
    if req.session_id not in envs:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    env = envs[req.session_id]
    env.queue.append({
        "id": os.urandom(4).hex(),
        "text": f"{req.subject}: {req.body}",
        "history": req.customer_history,
        "priority": req.priority,
        "category": req.category
    })
    
    # Sort after adding to maintain priority order
    prio_map = {"high": 3, "medium": 2, "low": 1, "hard": 3, "easy": 1}
    env.queue.sort(key=lambda x: prio_map.get(x["priority"], 2), reverse=True)
    
    return {"status": "ok", "queue_size": len(env.queue)}


@app.get("/state")
def get_state(session_id: str) -> Dict[str, Any]:
    if session_id not in envs:
        raise HTTPException(status_code=404, detail="Session not found.")
    return envs[session_id].state().model_dump()


@app.get("/tasks")
def get_tasks() -> Dict[str, Any]:
    return {
        task_id: {
            "ticket_text": data[0]["ticket_text"],
            "customer_history": data[0]["customer_history"],
            "difficulty": task_id.split("_")[0],
            "expected_sequence": ["classify_issue", "offer_refund" if "refund" in task_id else "offer_replacement" if "replacement" in task_id else "request_more_info", "close_ticket"],
        }
        for task_id, data in TASKS.items()
    }


@app.get("/validate")
def validate() -> Dict[str, Any]:
    """Quick sanity check used by openenv validate."""
    results = []
    for task_id, data_list in TASKS.items():
        env = CustomerSupportEnv(task_id=task_id)
        state = env.reset()
        
        # Determine sequence based on first ticket's category
        category = data_list[0].get("category", "refund")
        action_seq = ["classify_issue"]
        if category == "refund":
            action_seq.append("offer_refund")
        elif category == "replacement":
            action_seq.append("offer_replacement")
        elif category == "shipping_delay":
            action_seq.append("request_more_info")
        action_seq.append("close_ticket")
        
        rewards = []
        for action_type in action_seq:
            payload = category if action_type == "classify_issue" else None
            _, r_val, done, info = env.step(Action(action_type=action_type, payload=payload))
            rewards.append(r_val)
        
        # We only grade the first ticket for internal validation
        from env.grader import grade_trajectory
        score = grade_trajectory(task_id, env.state().history)
        results.append({"task_id": task_id, "score": score, "rewards": rewards})
        
    return {"status": "ok", "results": results}


def main():
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
