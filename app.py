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

# --- GRADIO INTERFACE (UPGRADED) ---

def gradio_reset(task_name):
    task_id = next(tid for tid, t in TASKS.items() if t["name"] == task_name)
    session_id = "gs_" + str(uuid.uuid4())[:6]
    env = CustomerSupportEnv(task_id=task_id)
    obs = env.reset()
    sessions[session_id] = env
    
    history_html = "<div style='color: #666; font-style: italic;'>No actions taken. Waiting for input...</div>"
    return session_id, obs.ticket_text, obs.customer_history, obs.status.upper(), history_html, 0.0, "🔴 Environment Reset. Ready."

def gradio_step(session_id, action_type, payload):
    if not session_id or session_id not in sessions:
        return "", "", "", "IDLE", "", 0.0, "❌ Error: Session expired or invalid. Please Reset."
    
    env = sessions[session_id]
    action = Action(action_type=action_type, payload=payload)
    obs, reward, done, info = env.step(action)
    
    feedback = info.get("feedback", "")
    
    # Color-coded feedback
    fb_color = "#2ecc71" if reward > 0 else "#e74c3c" if reward < 0 else "#f1c40f"
    feedback_styled = f"<div style='background-color: {fb_color}; color: white; padding: 10px; border-radius: 5px; font-weight: bold;'>{feedback}</div>"
    
    # Build logs
    logs_html = "<div style='font-family: monospace;'>"
    for h in obs.history:
        logs_html += f"<div style='border-bottom: 1px solid #eee; padding: 5px 0;'>[Step {h['step']}] <b>{h['action']}</b> -> {h['payload'] or 'None'}</div>"
    logs_html += "</div>"
    
    status_label = f"{obs.status.upper()}"
    if done:
        score = grade_trajectory(env.task_id, obs.history)
        status_label += f" (TASK FINISHED)"
        feedback_styled += f"<div style='margin-top: 10px; font-size: 1.2em;'>🏁 <b>Final Grader Score: {score:.2f}</b></div>"
    
    return session_id, obs.ticket_text, obs.customer_history, status_label, logs_html, reward, feedback_styled

with gr.Blocks(theme=gr.themes.Default(primary_hue="indigo", secondary_hue="slate"), css=".feedback-area { min-height: 100px; }") as demo:
    gr.HTML("<div style='text-align: center; padding: 20px;'><h1>🎫 AI Support Agent Workspace</h1><p>High-fidelity interaction environment for customer support reasoning.</p></div>")
    
    session_id_state = gr.State("")
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### ⚙️ Configuration")
                task_dropdown = gr.Dropdown(choices=[t["name"] for t in TASKS.values()], value=TASKS["medium_refund"]["name"], label="Difficulty Task")
                reset_btn = gr.Button("♻️ Initialize Workspace", variant="primary")
            
            with gr.Group():
                gr.Markdown("### 🛠️ Agent Console")
                action_type = gr.Radio(
                    choices=["classify_issue", "request_more_info", "offer_refund", "offer_replacement", "escalate", "close_ticket"],
                    value="classify_issue",
                    label="Action Selection"
                )
                payload = gr.Textbox(placeholder="Enter category or message detail...", label="Action Payload")
                step_btn = gr.Button("🚀 Execute Action", variant="secondary")

        with gr.Column(scale=2):
            with gr.Row():
                status_box = gr.Textbox(label="Lifecycle Status", interactive=False)
                reward_box = gr.Number(label="Latest Step Reward", interactive=False)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📥 Ticket Details")
                    ticket_text = gr.TextArea(label="Subject Text", lines=3, interactive=False)
                    history_context = gr.TextArea(label="Contextual History", lines=2, interactive=False)
                
                with gr.Column():
                    gr.Markdown("#### 📝 Execution Logs")
                    logs_area = gr.HTML("<div style='color: #666;'>Log is empty.</div>")
            
            gr.Markdown("#### 🎯 Performance Feedback")
            feedback_area = gr.HTML(label="Feedback", elem_classes=["feedback-area"])

    reset_btn.click(
        gradio_reset, 
        inputs=[task_dropdown], 
        outputs=[session_id_state, ticket_text, history_context, status_box, logs_area, reward_box, feedback_area]
    )
    
    step_btn.click(
        gradio_step,
        inputs=[session_id_state, action_type, payload],
        outputs=[session_id_state, ticket_text, history_context, status_box, logs_area, reward_box, feedback_area]
    )

# Mount Gradio into FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
