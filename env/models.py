from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

class TicketState(BaseModel):
    ticket_id: str
    ticket_text: str
    customer_history: str
    current_step: int
    status: Literal["open", "classified", "resolved", "closed", "escalated"]
    issue_category: Optional[str] = None
    resolution_action: Optional[str] = None
    history: List[Dict[str, Any]] = []

class Action(BaseModel):
    action_type: Literal["classify_issue", "request_more_info", "offer_refund", "offer_replacement", "escalate", "close_ticket"]
    payload: Optional[str] = None

class Reward(BaseModel):
    value: float
    reason: str
    done: bool
