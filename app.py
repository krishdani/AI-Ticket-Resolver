import os
import uuid
import gradio as gr
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from env.env import CustomerSupportEnv
from env.models import Action
from env.tasks import TASKS
from env.grader import grade_trajectory

# --- PROXY/API SETUP ---
app = FastAPI(title="AI Customer Support Resolver")

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
    try: data = await request.json()
    except: data = {}
    task_id = data.get("task_id", "medium_refund")
    session_id = str(uuid.uuid4())
    env = CustomerSupportEnv(task_id=task_id)
    obs = env.reset()
    sessions[session_id] = env
    return {"session_id": session_id, "observation": obs.model_dump(), "task_name": TASKS[task_id]["name"]}

@app.post("/step")
async def step(request: Request):
    data = await request.json()
    session_id = data.get("session_id")
    action_data = data.get("action")
    if session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    env = sessions[session_id]
    action = Action(**action_data)
    obs, reward, done, info = env.step(action)
    score = grade_trajectory(env.task_id, obs.history) if done else None
    return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info, "score": score}

@app.get("/state")
async def get_state(session_id: str):
    if session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id].state().model_dump()

@app.get("/health")
def health(): return {"status": "ok"}

# --- PREMIUM GRADIO UI ---

custom_css = """
.dashboard-container { background: #fdfdfd; border-radius: 20px; padding: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); }
.metric-card { background: white; border-radius: 15px; padding: 15px; border: 1px solid #eee; transition: transform 0.2s; }
.metric-card:hover { transform: translateY(-5px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
.log-entry { font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.9em; padding: 8px; border-bottom: 1px solid #f0f0f0; }
.log-success { color: #2ecc71; font-weight: bold; }
.log-fail { color: #e74c3c; font-weight: bold; }
.progress-bar-container { background: #eee; border-radius: 10px; height: 10px; margin: 15px 0; overflow: hidden; }
.progress-bar-fill { background: linear-gradient(90deg, #4f46e5, #818cf8); height: 100%; transition: width 0.5s ease-in-out; }
h1 { color: #1e293b; font-weight: 800 !important; letter-spacing: -1px; }
.gradio-container { max-width: 1100px !important; }
"""

def get_progress_html(current_stage, total_stages):
    percentage = (current_stage / total_stages) * 100 if total_stages > 0 else 0
    return f"""
    <div style="margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span style="font-weight: 600; font-size: 0.9em;">Workflow Progress</span>
            <span style="font-weight: 600; font-size: 0.9em;">{int(percentage)}%</span>
        </div>
        <div class="progress-bar-container">
            <div class="progress-bar-fill" style="width: {percentage}%"></div>
        </div>
    </div>
    """

def gradio_reset(task_name):
    task_id = next(tid for tid, t in TASKS.items() if t["name"] == task_name)
    session_id = "gs_" + str(uuid.uuid4())[:8]
    env = CustomerSupportEnv(task_id=task_id)
    obs = env.reset()
    sessions[session_id] = env
    
    total_stages = 1 if task_id == "easy_spam_filter" else 2 if task_id == "medium_refund" else 4
    prog_html = get_progress_html(0, total_stages)
    history_html = "<div style='color: #94a3b8; font-style: italic; text-align: center; padding: 40px;'>Waiting for the first action...</div>"
    
    return (
        session_id, 
        obs.ticket_text, 
        obs.customer_history, 
        f"STAGE 1: {task_name}", 
        history_html, 
        0.0, 
        "<div style='color: #6366f1; font-weight: 600;'>⚡ Workspace Initialized. System ready.</div>",
        prog_html
    )

def gradio_step(session_id, action_type, payload):
    if not session_id or session_id not in sessions:
        return "", "", "", "OFFLINE", "", 0.0, "❌ Session invalid.", ""
    
    env = sessions[session_id]
    action = Action(action_type=action_type, payload=payload)
    obs, reward, done, info = env.step(action)
    
    stage = info.get("current_stage", 0)
    task_id = env.task_id
    total_stages = 1 if task_id == "easy_spam_filter" else 2 if task_id == "medium_refund" else 4
    
    # Progress UI
    prog_html = get_progress_html(stage, total_stages)
    
    # Feedback UI
    fb_text = info.get("feedback", "")
    fb_color = "#10b981" if reward > 0 else "#ef4444" if reward < 0 else "#6366f1"
    fb_html = f"<div style='border-left: 5px solid {fb_color}; padding: 12px; background: {fb_color}11; border-radius: 8px;'><b style='color: {fb_color}'>{fb_text}</b></div>"
    
    # History Log UI
    logs_html = "<div>"
    for h in obs.history:
        status_point = "<span class='log-success'>✓</span>" if reward >= 0 else "<span class='log-fail'>×</span>"
        if h["step"] == len(obs.history) and done and reward > 0: status_point = "🏆 <span class='log-success'>SUCCESS</span>"
        logs_html += f"<div class='log-entry'><b>[{h['step']}]</b> {h['action']} &rarr; {status_point}</div>"
    logs_html += "</div>"
    
    status_label = f"STAGE {stage + 1} / {total_stages}"
    if done:
        score = grade_trajectory(env.task_id, obs.history)
        status_label = "✅ COMPLETED"
        fb_html += f"<div style='margin-top: 15px; font-size: 1.5em; color: #1e293b; background: white; padding: 15px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);'><b>Grader Score: {score:.2f}/1.00</b></div>"
        prog_html = get_progress_html(total_stages, total_stages)
        
    return session_id, obs.ticket_text, obs.customer_history, status_label, logs_html, reward, fb_html, prog_html

# --- UI BUILD ---

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", font=[gr.themes.GoogleFont("Outfit"), "sans-serif"]), css=custom_css) as demo:
    gr.HTML("<div style='text-align: center; margin-bottom: 30px; display: flex; align-items: center; justify-content: center; gap: 15px;'>"
            "<div style='font-size: 3em;'>🎯</div>"
            "<div><h1>OpenEnv: Customer Ticket Resolver</h1>"
            "<p style='color: #64748b; font-size: 1.1em;'>High-precision reinforcement learning playground for support agents</p></div></div>")
    
    session_id_state = gr.State("")
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            with gr.Group(elem_classes="dashboard-container"):
                gr.Markdown("### ⚙️ Workspace Config")
                task_dropdown = gr.Dropdown(choices=[t["name"] for t in TASKS.values()], value=TASKS["medium_refund"]["name"], label="Difficulty Task")
                reset_btn = gr.Button("🚀 Initialize Environment", variant="primary")
            
            with gr.Group(elem_classes="dashboard-container"):
                gr.Markdown("### 🤖 Agent Control")
                action_type = gr.Radio(choices=["classify_issue", "request_more_info", "offer_refund", "offer_replacement", "escalate", "close_ticket"], value="classify_issue", label="Action Select")
                payload = gr.Textbox(placeholder="Category string or metadata...", label="Action Payload")
                step_btn = gr.Button("⚡ Submit Step", variant="secondary")

        with gr.Column(scale=2):
            with gr.Group(elem_classes="dashboard-container"):
                progress_html_area = gr.HTML(get_progress_html(0, 3))
                with gr.Row():
                    status_box = gr.Textbox(label="Status Tracker", interactive=False)
                    reward_box = gr.Number(label="Step Return", interactive=False)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 📥 Live Ticket")
                        ticket_text = gr.TextArea(label="Current Customer Query", lines=3, interactive=False)
                        history_context = gr.TextArea(label="User Context/History", lines=2, interactive=False)
                    with gr.Column():
                        gr.Markdown("#### 📋 Action Trace")
                        logs_area = gr.HTML(elem_classes="metric-card")
                
                gr.Markdown("#### 💡 Intelligence Feedback")
                feedback_area = gr.HTML()

    # Interactions
    reset_btn.click(gradio_reset, inputs=[task_dropdown], outputs=[session_id_state, ticket_text, history_context, status_box, logs_area, reward_box, feedback_area, progress_html_area])
    step_btn.click(gradio_step, inputs=[session_id_state, action_type, payload], outputs=[session_id_state, ticket_text, history_context, status_box, logs_area, reward_box, feedback_area, progress_html_area])

# Mount & Launch
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
