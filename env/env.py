import uuid
from typing import Tuple, Dict, Any, List
from openenv.core.env_server.interfaces import Environment
from .models import TicketState, Action
from .tasks import TASKS

class CustomerSupportEnv(Environment):
    def __init__(self, task_id: str = "medium_refund"):
        self.task_id = task_id
        if task_id not in TASKS:
            task_id = list(TASKS.keys())[0]
            self.task_id = task_id
            
        self.task_data = TASKS[self.task_id]
        self.reset()

    def reset(self, **kwargs) -> TicketState:
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
        # Record history for grading
        self.state.history.append({"step": self.state.current_step, "action": action_type, "payload": action.payload})

        # Logic for classifications
        if action_type == "classify_issue":
            if self.state.status == "open":
                expected = self.task_data["expected_category"].lower()
                provided = (action.payload or "").lower()
                if provided == expected or (expected in provided and len(provided) > 2):
                    reward_val = 0.3
                    reason = f"Correct classification: {action.payload}"
                    self.state.status = "classified"
                    self.state.issue_category = action.payload
                else:
                    reward_val = -0.2
                    reason = f"Incorrect classification: {action.payload}"
            else:
                reward_val = -0.1
                reason = "Already classified"

        # Logic for actions
        elif action_type == "offer_refund":
            if self.state.status in ["classified", "info_requested"] and self.task_id == "medium_refund":
                reward_val = 0.5
                reason = "Correct resolution: Refund offered"
                self.state.status = "resolved"
            else:
                reward_val = -0.3
                reason = "Refund was not the correct action or was premature"
        
        elif action_type == "offer_replacement":
            if self.task_id == "hard_complex_replacement":
                if self.state.status == "info_requested":
                    reward_val = 0.5
                    reason = "Correct resolution: Replacement offered after info gathered"
                    self.state.status = "resolved"
                else:
                    reward_val = -0.1
                    reason = "Must request more info before replacing for this case"
            else:
                reward_val = -0.3
                reason = "Replacement was not the correct action"

        elif action_type == "request_more_info":
            if self.task_id == "hard_complex_replacement" and self.state.status == "classified":
                reward_val = 0.3
                reason = "Correct step: Requesting critical info"
                self.state.status = "info_requested"
            else:
                reward_val = -0.1
                reason = "Unnecessary info request"

        elif action_type == "close_ticket":
            if self.state.status == "resolved":
                reward_val = 0.2
                reason = "Ticket closed correctly"
                self.state.status = "closed"
                done = True
            elif self.task_id == "easy_spam_case" and self.state.status == "open":
                reward_val = 1.0 # One-step success
                reason = "Closed spam ticket correctly"
                self.state.status = "closed"
                done = True
            else:
                reward_val = -0.5
                reason = "Closed ticket without resolution"
                done = True

        elif action_type == "escalate":
            reward_val = -0.2 
            reason = "Escalated to human supervisor"
            self.state.status = "escalated"
            done = True

        # Global Step Penalty (Efficiency)
        reward_val -= 0.05
        
        # Max steps boundary
        if self.state.current_step >= 8:
            done = True

        info = {"reason": reason, "task_id": self.task_id}
        return self.state, reward_val, done, info
