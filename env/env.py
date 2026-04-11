import uuid
from typing import Tuple, Dict, Any, List
from openenv.core.env_server.interfaces import Environment
from .models import TicketState, Action
from .tasks import TASKS

class CustomerSupportEnv(Environment):
    def __init__(self, task_id: str = "easy_spam"):
        self.task_id = task_id
        if task_id not in TASKS:
             self.task_id = "easy_spam"
        self.task_data = TASKS[self.task_id]
        self.reset()

    def reset(self) -> TicketState:
        self.state = TicketState(
            ticket_id=str(uuid.uuid4())[:8],
            ticket_text=self.task_data["ticket_text"],
            customer_history=self.task_data["customer_history"],
            current_step=0,
            status="open",
            history=[]
        )
        return self.state

    def state(self) -> TicketState:
        return self.state

    def step(self, action: Action) -> Tuple[TicketState, float, bool, Dict[str, Any]]:
        self.state.current_step += 1
        reward_val = 0.0
        reason = ""
        done = False

        action_type = action.action_type
        payload = (action.payload or "").lower()
        
        # Record history
        self.state.history.append({"step": self.state.current_step, "action": action_type, "payload": action.payload})

        # --- REWARD LOGIC ---
        
        if action_type == "classify_issue":
            if self.state.status == "open":
                expected = self.task_data["expected_category"].lower()
                if payload == expected or (expected in payload and len(payload) > 2):
                    reward_val = 0.3 if self.task_id != "hard_damaged_replacement" else 0.2
                    reason = f"Correct classification: {payload}"
                    self.state.status = "classified"
                    self.state.issue_category = payload
                else:
                    reward_val = -0.3
                    reason = f"Incorrect classification: {payload} (Expected {expected})"
            else:
                reward_val = -0.1
                reason = "Already classified"

        elif action_type == "request_more_info":
            if self.task_id == "hard_damaged_replacement" and self.state.status == "classified":
                reward_val = 0.3
                reason = "Correct step: Requesting item details for replacement."
                self.state.status = "resolved" # Mocking progress
            else:
                reward_val = -0.2
                reason = "Unnecessary information request."

        elif action_type == "offer_refund":
            if self.task_id == "medium_refund" and self.state.status == "classified":
                reward_val = 0.5
                reason = "Correct resolution: Refund offered."
                self.state.status = "resolved"
            else:
                reward_val = -0.4
                reason = "Refund offered at wrong time or for wrong task."
        
        elif action_type == "offer_replacement":
            if self.task_id == "hard_damaged_replacement" and self.state.status == "resolved":
                reward_val = 0.4
                reason = "Correct resolution: Replacement offered after details verified."
                self.state.status = "resolved"
            else:
                reward_val = -0.4
                reason = "Replacement offered without proper verification or for wrong task."

        elif action_type == "close_ticket":
            # Easy task special case: can close spam immediately
            if self.task_id == "easy_spam":
                reward_val = 0.7
                reason = "Successfully handled spam."
                done = True
            elif self.state.status == "resolved":
                reward_val = 0.2
                reason = "Ticket closed after resolution."
                done = True
            else:
                reward_val = -0.5
                reason = "Closed ticket prematurely."
                done = True

        elif action_type == "escalate":
            reward_val = 0.0 # Neutral
            reason = "Escalated to human."
            done = True

        # --- PENALTIES ---
        if self.state.current_step > 5:
            reward_val -= 0.1
            reason += " (Step penalty)"
        
        if self.state.current_step >= 8:
            done = True
            reason += " (Max steps reached)"

        info = {"reason": reason, "task_id": self.task_id}
        return self.state, float(reward_val), done, info
