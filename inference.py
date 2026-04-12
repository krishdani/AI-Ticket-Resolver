#!/usr/bin/env python3
"""
OpenEnv-compliant inference script for AI Customer Support RL Environment.
Strictly follows the [START], [STEP], [END] logging format defined in the 
Hackathon Submission Guidelines.
"""
import os
import sys
import json
import asyncio
from typing import List, Dict, Any, Optional
from openai import OpenAI

# --- 1. Environment Variables & Client Setup (Mandatory) ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# The challenge benchmark identifier
BENCHMARK = "ai_customer_support_resolver"

# Initialize OpenAI client as required
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# --- 2. Environment Dependencies ---
from env.env import CustomerSupportEnv
from env.models import Action
from env.tasks import TASKS
from env.grader import grade_trajectory

def get_llm_action(task_id: str, state_json: str) -> Action:
    """Uses the OpenAI client to decide the next action."""
    system_prompt = (
        "You are an AI customer support agent. Respond ONLY with a JSON object.\n"
        "Actions: classify_issue (payload: category), request_more_info, offer_refund, offer_replacement, escalate, close_ticket.\n"
        "JSON format: {\"action_type\": \"...\", \"payload\": \"...\"}"
    )
    user_prompt = f"State: {state_json}\nTask: {task_id}\nAction?"

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
    except Exception:
        # Fallback to avoid script failure
        return Action(action_type="close_ticket")

async def run_task(task_id: str):
    env = CustomerSupportEnv(task_id=task_id)
    obs = env.reset()
    
    # [START] task=<task_name> env=<benchmark> model=<model_name>
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    done = False
    step_count = 0
    max_steps = 10
    rewards = []
    
    try:
        while not done and step_count < max_steps:
            step_count += 1
            
            # Use mandatory LLM usage requirement
            action = get_llm_action(task_id, obs.model_dump_json())
            
            # Perform env step
            obs, reward, done, info = env.step(action)
            rewards.append(float(reward))
            
            # [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
            action_str = f"{action.action_type}"
            if action.payload:
                action_str += f"('{action.payload}')"
            
            print(
                f"[STEP] step={step_count} action={action_str} "
                f"reward={reward:.2f} done={str(done).lower()} error=null",
                flush=True
            )

        # Calculate success for the END log
        score = grade_trajectory(task_id, obs.history)
        success = score >= 0.7

    except Exception as e:
        # Log step with error if something fails internally
        print(f"[STEP] step={step_count+1} action=error reward=0.01 done=true error={str(e)}", flush=True)
        success = False
    finally:
        # [END] success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
        rewards_str = ",".join([f"{r:.2f}" for r in rewards])
        print(
            f"[END] success={str(success).lower()} steps={step_count} "
            f"score={score:.2f} rewards={rewards_str}",
            flush=True
        )

async def main():
    # Run against the 3 required tasks
    for task_id in TASKS.keys():
        await run_task(task_id)

if __name__ == "__main__":
    asyncio.run(main())
