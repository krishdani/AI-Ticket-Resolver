#!/usr/bin/env python3
"""
OpenEnv-compliant inference script for AI Customer Support RL Environment.
Follows the mandatory [START], [STEP], [END] logging format.
"""
import os
import sys
import json
import asyncio
from typing import List, Dict, Any, Optional

from openai import OpenAI
from env.env import CustomerSupportEnv
from env.models import Action
from env.tasks import TASKS
from env.grader import grade_trajectory

# --- Configuration (from Hackathon Specs) ---
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy_key"
BENCHMARK = "ai_customer_support_resolver"

# --- OpenAI Client Setup ---
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def get_llm_action(task_id: str, state_json: str) -> Action:
    """Uses LLM to decide the next action."""
    system_prompt = (
        "You are an AI customer support agent. Your goal is to resolve the customer's issue efficiently.\n"
        "You can take the following actions:\n"
        "1. classify_issue (payload: category name e.g. refund, replacement, shipping_delay)\n"
        "2. request_more_info (ask for more details)\n"
        "3. offer_refund (give money back)\n"
        "4. offer_replacement (send a new item)\n"
        "5. escalate (send to a human)\n"
        "6. close_ticket (finalize once resolved)\n"
        "Respond ONLY with a JSON object like: {\"action_type\": \"...\", \"payload\": \"...\"}"
    )
    user_prompt = f"Current State: {state_json}\nTask: {task_id}\nWhat is your next action?"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        data = json.loads(response.choices[0].message.content)
        return Action(action_type=data["action_type"], payload=data.get("payload"))
    except Exception as e:
        # Fallback to a safe action if LLM fails or is unavailable for baseline
        if "refund" in state_json.lower(): return Action(action_type="classify_issue", payload="refund")
        return Action(action_type="close_ticket")

async def run_task(task_id: str):
    env = CustomerSupportEnv(task_id=task_id)
    obs = env.reset()
    
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    done = False
    step_count = 0
    max_steps = 8
    rewards = []
    
    while not done and step_count < max_steps:
        step_count += 1
        
        # In a real baseline, we might use heuritics for reproducibility or LLM
        # For this hackathon, we use the LLM client as required.
        action = get_llm_action(task_id, obs.model_dump_json())
        
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        
        # Mandatory [STEP] format
        # [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
        action_str = f"{action.action_type}({action.payload or ''})"
        print(
            f"[STEP] step={step_count} action={action_str} "
            f"reward={reward:.2f} done={str(done).lower()} error=null",
            flush=True
        )

    # Calculate final score
    score = grade_trajectory(task_id, obs.history)
    success = score > 0.7 # Define success threshold

    # Mandatory [END] format
    # [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(
        f"[END] success={str(success).lower()} steps={step_count} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True
    )

async def main():
    for task_id in TASKS.keys():
        await run_task(task_id)

if __name__ == "__main__":
    asyncio.run(main())
