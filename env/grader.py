from typing import List, Dict, Any
from .tasks import TASKS

def grade_trajectory(task_id: str, trajectory: List[Dict[str, Any]]) -> float:
    """
    Refined deterministic grader for AI Ticket Resolver.
    Evaluates:
    - Task Success (0.5)
    - Correct Sequence/Logic (0.3)
    - Efficiency (0.2)
    """
    task = TASKS.get(task_id)
    if not task or not trajectory:
        return 0.0

    score = 0.0
    actions = [t["action"] for t in trajectory]
    payloads = [str(t.get("payload") or "").lower() for t in trajectory]
    
    # 1. Goal Success (0.5)
    success = False
    if task_id == "easy_spam_filter":
        if "classify_issue" in actions:
            idx = actions.index("classify_issue")
            if payloads[idx] == "spam": success = True
    elif task_id == "medium_refund":
        if "offer_refund" in actions and "classify_issue" in actions:
            success = True
    elif task_id == "hard_damaged_replacement":
        if "offer_replacement" in actions and "close_ticket" in actions:
            success = True

    if success: score += 0.5

    # 2. Sequence/Logic (0.3)
    # Check if classify came before anything else
    if "classify_issue" in actions:
        if actions[0] == "classify_issue":
            score += 0.1
            # Check payload correctness
            if payloads[0] == task["expected_category"].lower():
                score += 0.1
    
    # Hard mode sequence check: classify -> info -> resolution
    if task_id == "hard_damaged_replacement" and success:
        if "request_more_info" in actions:
            idx_info = actions.index("request_more_info")
            idx_res = actions.index("offer_replacement")
            if idx_info < idx_res:
                score += 0.1 # Correct sequence

    # 3. Efficiency (0.2)
    ideal_steps = len(task["expected_sequence"])
    actual_steps = len(actions)
    
    if actual_steps == ideal_steps:
        score += 0.2
    elif actual_steps <= ideal_steps + 1:
        score += 0.1
    elif actual_steps <= ideal_steps + 2:
        score += 0.05

    return float(min(max(score, 0.0), 1.0))
