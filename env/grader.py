from typing import List, Dict, Any
from .tasks import TASKS

def grade_trajectory(task_id: str, trajectory: List[Dict[str, Any]]) -> float:
    task = TASKS.get(task_id)
    if not task:
        return 0.0

    score = 0.0
    actions_taken = [t["action"] for t in trajectory]
    
    # Check sequence correctness
    # If the set of required actions is found in the trajectory in any order (though we prefer sequence)
    # However, for support, sequence matters.
    
    if task_id == "easy_spam_case":
        if "close_ticket" in actions_taken:
            score = 1.0
    
    elif task_id == "medium_refund":
        if "classify_issue" in actions_taken and "offer_refund" in actions_taken and actions_taken[-1] == "close_ticket":
            score = 1.0
        elif "classify_issue" in actions_taken and "offer_refund" in actions_taken:
            score = 0.7
        elif "classify_issue" in actions_taken:
            score = 0.3
            
    elif task_id == "hard_complex_replacement":
        has_class = "classify_issue" in actions_taken
        has_info = "request_more_info" in actions_taken
        has_replace = "offer_replacement" in actions_taken
        is_closed = actions_taken[-1] == "close_ticket"
        
        if has_class and has_info and has_replace and is_closed:
            score = 1.0
        elif has_class and has_info and has_replace:
            score = 0.8
        elif has_class and has_info:
            score = 0.5
        elif has_class:
            score = 0.2

    # Efficiency Penalty: Lose 0.05 per action over the minimum needed
    min_steps = len(task["expected_sequence"])
    extra_steps = max(0, len(actions_taken) - min_steps)
    score -= (extra_steps * 0.05)

    return min(max(float(score), 0.0), 1.0)
