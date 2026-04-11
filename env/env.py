import uuid
from typing import Tuple, Dict, Any, List
from openenv.core.env_server.interfaces import Environment
from .models import TicketState, Action
from .tasks import TASKS

class CustomerSupportEnv(Environment):
    """
    High-fidelity state-based Customer Support environment.
    Enforces strict action sequencing (Stage 0 -> Stage 1 -> Stage 2).
    """
    def __init__(self, task_id: str = "medium_refund"):
        self.task_id = task_id if task_id in TASKS else "medium_refund"
        self.task_data = TASKS[self.task_id]
        self.reset()

    def reset(self, task_id: str = None) -> TicketState:
        if task_id and task_id in TASKS:
            self.task_id = task_id
            self.task_data = TASKS[self.task_id]
            
        self.current_stage = 0
        self.state_data = TicketState(
            ticket_id=str(uuid.uuid4())[:8],
            ticket_text=self.task_data["ticket_text"],
            customer_history=self.task_data["customer_history"],
            current_step=0,
            status="open",
            history=[]
        )
        return self.state_data

    def state(self) -> TicketState:
        return self.state_data

    def _get_required_action(self, stage: int) -> str:
        """Helper to get the expected action type for a given stage/task."""
        if stage == 0:
            return "classify_issue"
        
        if self.task_id == "easy_spam_filter":
            return "DONE" # Should have finished after stage 0
        
        if self.task_id == "medium_refund":
            if stage == 1: return "offer_refund"
            return "DONE"
            
        if self.task_id == "hard_damaged_replacement":
            if stage == 1: return "request_more_info"
            if stage == 2: return "offer_replacement"
            if stage == 3: return "close_ticket"
            return "DONE"
        
        return "DONE"

    def step(self, action: Action) -> Tuple[TicketState, float, bool, Dict[str, Any]]:
        self.state_data.current_step += 1
        reward_val = 0.0
        feedback = ""
        done = False

        action_type = action.action_type
        payload = (action.payload or "").lower()
        
        # 1. State/History Updates
        previous_actions = [h["action"] for h in self.state_data.history]
        is_repeat = action_type in previous_actions and action_type not in ["request_more_info"]
        self.state_data.history.append({
            "step": self.state_data.current_step, 
            "action": action_type, 
            "payload": action.payload,
            "stage": self.current_stage
        })

        # 2. Get Expected Action
        expected_type = self._get_required_action(self.current_stage)
        
        # 3. Sequence Validation Logic
        if action_type == expected_type:
            # Correct Action Type
            if action_type == "classify_issue":
                expected_cat = self.task_data["expected_category"].lower()
                if payload == expected_cat or (expected_cat in payload and len(payload) > 2):
                    reward_val = 0.3
                    feedback = f"Step {self.current_stage + 1} correct: classification successful"
                    self.current_stage += 1
                    self.state_data.status = "classified"
                    # EASY task ends here
                    if self.task_id == "easy_spam_filter":
                        reward_val += 0.4
                        feedback += ". Task completed successfully"
                        done = True
                else:
                    reward_val = -0.2
                    feedback = f"Wrong classification provided: {payload}"
            
            elif action_type == "request_more_info":
                reward_val = 0.3
                feedback = f"Step {self.current_stage + 1} correct: details requested"
                self.current_stage += 1
                self.state_data.status = "validated"
                
            elif action_type in ["offer_refund", "offer_replacement"]:
                reward_val = 0.4
                feedback = f"Step {self.current_stage + 1} correct: resolution offered"
                self.current_stage += 1
                self.state_data.status = "resolved"
                # MEDIUM task ends here
                if self.task_id == "medium_refund":
                    feedback = "Task completed successfully"
                    done = True
                    
            elif action_type == "close_ticket":
                reward_val = 0.4
                feedback = "Task completed successfully"
                self.state_data.status = "closed"
                done = True
        
        else:
            # Wrong Action Type
            if is_repeat:
                reward_val = -0.1
                feedback = "Repeated action detected"
            else:
                reward_val = -0.2
                if expected_type == "DONE":
                    feedback = "Task already completed, please close ticket"
                    done = True
                else:
                    feedback = f"Step {self.current_stage + 1} required: {expected_type} before resolution"

        # 4. Final Constraints
        if self.state_data.current_step >= 8:
            done = True

        info = {
            "feedback": feedback, 
            "current_stage": self.current_stage,
            "expected_next": self._get_required_action(self.current_stage)
        }
        
        return self.state_data, float(reward_val), done, info
