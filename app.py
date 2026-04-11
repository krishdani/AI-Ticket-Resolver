import os
import uuid
import gradio as gr
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from env.env import CustomerSupportEnv
from env.models import Action
from env.tasks import TASKS
from env.grader import grade_trajectory

app = FastAPI(title="AI Customer Support resolver")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared state for sessions
sessions = {}

# --- OPENENV API ENDPOINTS ---

@app.post("/reset")
async def reset(request: Request):
    try:
        data = await request.json()
    except:
        data = {}
    
    task_id = data.get("task_id", "medium_refund")
    session_id = str(uuid.uuid4())
    env = CustomerSupportEnv(task_id=task_id)
    obs = env.reset()
    sessions[session_id] = env
    
    return {
        "session_id": session_id,
        "observation": obs.model_dump(),
        "task_name": TASKS[task_id]["name"]
    }

@app.post("/step")
async def step(request: Request):
    data = await request.json()
    session_id = data.get("session_id")
    action_data = data.get("action")
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    env = sessions[session_id]
    action = Action(**action_data)
    obs, reward, done, info = env.step(action)
    
    score = None
    if done:
        score = grade_trajectory(env.task_id, obs.history)
        
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
        "score": score
    }

@app.get("/state")
async def get_state(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id].state().model_dump()

@app.get("/health")
def health():
    return {"status": "ok"}

# --- GRADIO INTERFACE ---

def gradio_reset(task_name):
    task_id = next(tid for tid, t in TASKS.items() if t["name"] == task_name)
    session_id = "gradio_session_" + str(uuid.uuid4())[:8]
    env = CustomerSupportEnv(task_id=task_id)
    obs = env.reset()
    sessions[session_id] = env
    
    history_html = "<i>No actions taken yet.</i>"
    return session_id, obs.ticket_text, obs.customer_history, obs.status, history_html, 0.0, "Ready."

def gradio_step(session_id, action_type, payload):
    if not session_id or session_id not in sessions:
        return "Error", "", "", "", "", 0.0, "Please reset first."
    
    env = sessions[session_id]
    action = Action(action_type=action_type, payload=payload)
    obs, reward, done, info = env.step(action)
    
    # Format history
    history_html = "<ul style='list-style-type: none; padding: 0;'>"
    for h in obs.history:
        color = "#4CAF50" if "Correct" in info.get("reason", "") else "#2196F3"
        history_html += f"<li style='margin-bottom: 5px; border-left: 4px solid {color}; padding-left: 10px;'><b>Step {h['step']}:</b> {h['action']} ({h['payload'] or ''})</li>"
    history_html += "</ul>"
    
    status_label = f"{obs.status.upper()}"
    if done:
        score = grade_trajectory(env.task_id, obs.history)
        status_label += f" | FINAL SCORE: {score:.2f}"
    
    return session_id, obs.ticket_text, obs.customer_history, status_label, history_html, reward, info.get("reason", "")

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")) as demo:
    gr.Markdown("# 🎫 AI Customer Support Ticket Resolver")
    gr.Markdown("Interactive environment for training AI support agents. Select a task and try to resolve it step-by-step.")
    
    session_id_state = gr.State("")
    
    with gr.Row():
        with gr.Column(scale=1):
            task_dropdown = gr.Dropdown(choices=[t["name"] for t in TASKS.values()], value=TASKS["medium_refund"]["name"], label="Select Task")
            reset_btn = gr.Button("♻️ Initialize / Reset Environment", variant="primary")
            
            with gr.Group():
                gr.Markdown("### 🤖 Take Action")
                action_type = gr.Dropdown(
                    choices=["classify_issue", "request_more_info", "offer_refund", "offer_replacement", "escalate", "close_ticket"],
                    value="classify_issue",
                    label="Action Type"
                )
                payload = gr.Textbox(placeholder="e.g. refund, replacement, or message text...", label="Action Payload")
                step_btn = gr.Button("▶️ Submit Action", variant="secondary")

        with gr.Column(scale=2):
            with gr.Row():
                status_box = gr.Label(value="IDLE", label="Environment Status")
                reward_box = gr.Number(value=0.0, label="Last Reward")
            
            with gr.Tabs():
                with gr.TabItem("📋 Ticket Details"):
                    ticket_text = gr.Textbox(label="Customer Message", interactive=False)
                    history_context = gr.Textbox(label="Customer History", interactive=False)
                
                with gr.TabItem("📜 Interaction Logs"):
                    logs_html = gr.HTML("<i>No actions taken yet.</i>")
            
            feedback_box = gr.Textbox(label="Execution Feedback", interactive=False)

    reset_btn.click(
        gradio_reset, 
        inputs=[task_dropdown], 
        outputs=[session_id_state, ticket_text, history_context, status_box, logs_html, reward_box, feedback_box]
    )
    
    step_btn.click(
        gradio_step,
        inputs=[session_id_state, action_type, payload],
        outputs=[session_id_state, ticket_text, history_context, status_box, logs_html, reward_box, feedback_box]
    )

# Mount Gradio into FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
