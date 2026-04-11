from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from openenv.core.env_server.types import Action as BaseAction, Observation as BaseObservation

class Ticket(BaseModel):
    subject: str
    body: str
    priority: Literal["low", "medium", "high"] = "medium"

class TicketState(BaseObservation):
    ticket_id: str
    ticket_text: str
    customer_history: str
    current_step: int
    status: Literal["open", "classified", "resolved", "closed", "escalated"]
    issue_category: Optional[str] = None
    resolution_action: Optional[str] = None
    history: List[Dict[str, Any]] = []
    priority: str = "medium"
    queue_remaining: int = 0

class Action(BaseAction):
    action_type: Literal["classify_issue", "request_more_info", "offer_refund", "offer_replacement", "escalate", "close_ticket"]
    payload: Optional[str] = None

class Reward(BaseModel):
    value: float
    reason: str
    done: bool
