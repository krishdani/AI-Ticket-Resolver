import uuid
from typing import Tuple, Dict, Any
from .models import TicketState, Action, Reward
from .tasks import TASKS

class CustomerSupportEnv:
    def __init__(self, task_id: str = "easy_refund"):
        self.task_id = task_id
        self.task_data = TASKS.get(task_id, TASKS["easy_refund"])
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

    def get_state(self) -> TicketState:
        return self.state

    def step(self, action: Action) -> Tuple[TicketState, Reward]:
        self.state.current_step += 1
        reward_val = 0.0
        reason = ""
        done = False

        action_type = action.action_type
        self.state.history.append({"step": self.state.current_step, "action": action_type, "payload": action.payload})

        # Logic for classifications
        if action_type == "classify_issue":
            if self.state.status == "open":
                expected = self.task_data["expected_category"].lower()
                provided = action.payload.lower() if action.payload else ""
                if provided == expected or expected in provided:
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
            if self.task_id == "easy_refund":
                reward_val = 0.5
                reason = "Correct resolution: Refund offered"
                self.state.status = "resolved"
            else:
                reward_val = -0.3
                reason = "Refund was not the correct action"
        
        elif action_type == "offer_replacement":
            if self.task_id == "medium_replacement":
                reward_val = 0.5
                reason = "Correct resolution: Replacement offered"
                self.state.status = "resolved"
            else:
                reward_val = -0.3
                reason = "Replacement was not the correct action"

        elif action_type == "request_more_info":
            if self.task_id == "hard_delay_case":
                reward_val = 0.4
                reason = "Correct step: Requesting info for delay"
                # For hard case, we might need more steps
            else:
                reward_val = -0.1
                reason = "Unnecessary info request"

        elif action_type == "close_ticket":
            if self.state.status in ["resolved", "classified"] or self.task_id == "hard_delay_case":
                reward_val = 0.2
                reason = "Ticket closed correctly"
                self.state.status = "closed"
                done = True
            else:
                reward_val = -0.5
                reason = "Closed ticket without resolution"
                done = True

        elif action_type == "escalate":
            reward_val = -0.1 # Generic penalty for escalation unless needed
            reason = "Escalated to human supervisor"
            self.state.status = "escalated"
            done = True

        # Penalty for too many steps
        if self.state.current_step > 5:
            reward_val -= 0.1
            reason += " (Step penalty)"
            if self.state.current_step >= 8:
                done = True

        return self.state, Reward(value=reward_val, reason=reason, done=done)
