import uuid
from typing import Tuple, Dict, Any
from openenv.core.env_server.interfaces import Environment
from .models import TicketState, Action
from .tasks import TASKS
from .grader import safe_score

class CustomerSupportEnv(Environment):
    def __init__(self, task_id: str = "easy_refund"):
        self.task_id = task_id
        # Load all 5 tickets for this difficulty level
        batch = TASKS.get(task_id, TASKS["easy_refund"])
        self.queue = []
        
        for t in batch:
            self.queue.append({
                "id": str(uuid.uuid4())[:8],
                "text": t["ticket_text"],
                "history": t["customer_history"],
                "priority": t.get("priority", "medium"),
                "category": t["category"]
            })
        
        self.reset()

    def reset(self) -> TicketState:
        if not self.queue:
            # Fallback if queue somehow empty
            self.queue = [{"id": "err", "text": "No tickets", "history": "", "priority": "low", "category": "none"}]

        # Sort queue: high > medium > low
        prio_map = {"hard": 3, "medium": 2, "easy": 1, "high": 3, "low": 1}
        self.queue.sort(key=lambda x: prio_map.get(x["priority"], 2), reverse=True)
        
        current = self.queue[0]
        self.current_ticket = current
        
        # Preserve history if it exists
        old_history = getattr(self, "_state", None) and self._state.history or []
        
        self._state = TicketState(
            ticket_id=current["id"],
            ticket_text=current["text"],
            customer_history=current["history"],
            current_step=0,
            status="open",
            history=old_history,
            priority=current["priority"],
            queue_remaining=len(self.queue) - 1
        )
        return self._state

    def state(self) -> TicketState:
        return self._state

    def step(self, action: Action) -> Tuple[TicketState, float, bool, Dict[str, Any]]:
        self._state.current_step += 1
        reward_val = 0.0
        reason = ""
        done = False

        action_type = action.action_type
        self._state.history.append({"step": self._state.current_step, "action": action_type, "payload": action.payload})

        # Logic for classifications
        if action_type == "classify_issue":
            if self._state.status == "open":
                expected = self.current_ticket["category"].lower()
                provided = (action.payload or "").lower()
                if provided == expected or (expected in provided and len(provided) > 2):
                    reward_val = 0.3
                    reason = f"Correct classification: {action.payload}"
                    self._state.status = "classified"
                    self._state.issue_category = action.payload
                else:
                    reward_val = 0.0
                    reason = f"Incorrect classification: {action.payload}. (No reward)"
            else:
                reward_val = 0.0
                reason = "Already classified. (No reward)"

        # Logic for resolutions
        elif action_type == "offer_refund":
            if self.current_ticket["category"] == "refund":
                reward_val = 0.5
                reason = "Correct resolution: Refund offered"
                self._state.status = "resolved"
            else:
                reward_val = 0.0
                reason = "Refund was not the correct action. (No reward)"
        
        elif action_type == "offer_replacement":
            if self.current_ticket["category"] == "replacement":
                reward_val = 0.5
                reason = "Correct resolution: Replacement offered"
                self._state.status = "resolved"
            else:
                reward_val = 0.0
                reason = "Replacement was not the correct action. (No reward)"

        elif action_type == "request_more_info":
            if self.current_ticket["category"] == "shipping_delay":
                reward_val = 0.4
                reason = "Correct step: Requesting info for delay"
            else:
                reward_val = 0.0
                reason = "Unnecessary info request. (No reward)"

        elif action_type == "close_ticket":
            if self._state.status in ["resolved", "classified"] or self.current_ticket["category"] == "shipping_delay":
                reward_val = 0.2
                reason = "Ticket closed correctly"
                self._state.status = "closed"
                
                # Check if there are more tickets in queue
                self.queue.pop(0)
                if self.queue:
                    reason += ". Loading next ticket from queue..."
                    self.reset()
                else:
                    done = True
            else:
                reward_val = 0.0
                reason = "Closed ticket without resolution. (No reward)"
                done = True

        elif action_type == "escalate":
            reward_val = 0.0
            reason = "Escalated to human supervisor. (No reward)"
            self._state.status = "escalated"
            
            # Logic: Load next ticket even if escalated?
            self.queue.pop(0)
            if self.queue:
                self.reset()
            else:
                done = True

        # Penalty for too many steps on one ticket (Efficiency)
        if self._state.current_step > 5:
            # We don't subtract, we just might cap future rewards or just log it
            reason += " (Efficiency warning: took more than 5 steps)"
            if self._state.current_step >= 8:
                # Force next ticket or end
                self.queue.pop(0)
                if self.queue:
                    self.reset()
                else:
                    done = True

        info = {"reason": reason, "task_id": self.task_id}
        return self._state, safe_score(reward_val), done, info
