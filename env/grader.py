from typing import List, Dict, Any
from .tasks import TASKS

def grade_trajectory(task_id: str, trajectory: List[Dict[str, Any]]) -> float:
    task = TASKS.get(task_id)
    if not task:
        return 0.0

    score = 0.0
    actions_taken = [t["action"] for t in trajectory]
    
    # Check classification
    correct_class = False
    for step in trajectory:
        if step["action"] == "classify_issue":
            if step.get("payload") and step["payload"].lower() in task["expected_category"].lower():
                correct_class = True
                score += 0.3
                break # Only count first correct classification

    # Check sequence / goal
    if task_id == "easy_refund":
        if "offer_refund" in actions_taken:
            score += 0.5
    elif task_id == "medium_replacement":
        if "offer_replacement" in actions_taken:
            score += 0.5
    elif task_id == "hard_delay_case":
        if "request_more_info" in actions_taken and "offer_refund" not in actions_taken:
            score += 0.5

    # Check closing
    if actions_taken and actions_taken[-1] == "close_ticket":
        score += 0.2

    # Efficiency bonus
    if len(actions_taken) <= len(task["expected_sequence"]) + 1:
        score += 0.1

    return min(max(score, 0.0), 1.0)
