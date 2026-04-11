import os
import sys
import json
from typing import List
from openai import OpenAI
from env.env import CustomerSupportEnv
from env.models import Action
from env.tasks import TASKS
from env.grader import grade_trajectory

# --- Configuration ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy_key"
BENCHMARK = "ai_customer_support_resolver"

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def get_agent_action(obs_json: str, task_id: str) -> Action:
    """Queries the LLM to decide the next action."""
    prompt = f"""
    You are an AI Customer Support Agent.
    Your objective is to resolve the ticket for task: {task_id}.
    
    Current Environment State:
    {obs_json}
    
    Available Actions:
    - classify_issue (payload: spam, refund, or replacement)
    - request_more_info (ask for more details)
    - offer_refund (give money back)
    - offer_replacement (send new item)
    - escalate (send to human)
    - close_ticket (finalize interaction)
    
    Rule 1: Always classify_issue first if status is 'open'.
    Rule 2: Read current text and history carefully to choose the right resolution.
    Rule 3: Only close_ticket after resolution.
    
    Respond ONLY with a JSON object: {{"action_type": "...", "payload": "..."}}
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        data = json.loads(response.choices[0].message.content)
        return Action(action_type=data["action_type"], payload=data.get("payload"))
    except Exception as e:
        # Fallback to a simple heuristic if LLM fails
        return Action(action_type="close_ticket", payload=str(e))

def run_inference():
    for task_id in TASKS.keys():
        env = CustomerSupportEnv(task_id=task_id)
        obs = env.reset()
        
        print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)
        
        done = False
        step_n = 0
        rewards_list = []
        
        while not done and step_n < 8:
            step_n += 1
            
            # 1. Get action from agent
            action = get_agent_action(obs.model_dump_json(), task_id)
            
            # 2. Step environment
            obs, reward, done, info = env.step(action)
            rewards_list.append(reward)
            
            # 3. Log step
            # [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
            action_str = f"{action.action_type}({action.payload or ''})"
            print(f"[STEP] step={step_n} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            
        # 4. End episode and log summary
        score = grade_trajectory(task_id, obs.history)
        success = score > 0.7
        rewards_str = ",".join([f"{r:.2f}" for r in rewards_list])
        
        # [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
        print(f"[END] success={str(success).lower()} steps={step_n} score={score:.2f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    run_inference()
