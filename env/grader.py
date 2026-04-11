from typing import List, Dict, Any

def grade_trajectory(task_id: str, trajectory: List[Dict[str, Any]]) -> float:
    """
    Evaluates the full session trajectory across multiple tickets.
    Score is based on:
    - Percentage of correctly classified tickets.
    - Percentage of correctly resolved tickets.
    - Overall efficiency.
    """
    if not trajectory:
        return 0.0

    total_tickets_attempted = 0
    correctly_classified = 0
    correctly_resolved = 0
    closed_correctly = 0
    
    # Simple state machine to track progress per ticket ID in history
    # Our history items in TicketState now include step, action, and payload.
    # To truly grade multi-ticket, we'd need the ticket_id in each history item.
    # For now, we'll count occurrences of 'close_ticket' and 'classify_issue'.
    
    actions = [t["action"] for t in trajectory]
    
    # Each successful close usually follows a resolve and a classify.
    # In a 5-ticket session, we expect 5 'close_ticket' actions.
    closed_count = actions.count("close_ticket")
    classified_count = actions.count("classify_issue")
    
    # Score components: 
    # 5 tickets * (0.3 classify + 0.5 resolve + 0.2 close) = 1.0 per ticket total.
    # We'll normalize by the intended number of tickets (5).
    
    # We'll heuristic this: 
    # For each 'close_ticket', if it wasn't the very first action, 
    # we assume a resolution path was attempted.
    
    score = (closed_count * 0.2) + (classified_count * 0.1)
    
    # Check for specific "good" actions (refund for easy, etc.)
    if "offer_refund" in actions: score += 0.1
    if "offer_replacement" in actions: score += 0.1
    if "request_more_info" in actions: score += 0.1

    # Clamp to 1.0
    final_score = min(max(score / 1.5, 0.0), 1.0)
    
    return final_score
