from typing import List, Dict, Any
from .tasks import TASKS

def grade_trajectory(task_id: str, trajectory: List[Dict[str, Any]]) -> float:
    """
    Programmatic deterministic grader for AI Ticket Resolver.
    Returns a score between 0.0 and 1.0.
    """
    task = TASKS.get(task_id)
    if not task:
        return 0.0

    score = 0.0
    actions_taken = [t["action"] for t in trajectory]
    payloads = [str(t.get("payload") or "").lower() for t in trajectory]
    
    # 1. Classification check (0.3 points)
    correct_class = False
    for i, action in enumerate(actions_taken):
        if action == "classify_issue":
            expected = task["expected_category"].lower()
            if payloads[i] == expected or expected in payloads[i]:
                correct_class = True
                score += 0.3
                break

    # 2. Sequence quality and Goal Completion (0.5 points)
    if task_id == "easy_spam":
        if "close_ticket" in actions_taken:
            score += 0.5 # Simple goal
    elif task_id == "medium_refund":
        if correct_class and "offer_refund" in actions_taken:
            score += 0.5
    elif task_id == "hard_damaged_replacement":
        # Must Classify -> Info -> Replacement
        if correct_class:
            has_info = "request_more_info" in actions_taken
            has_replace = "offer_replacement" in actions_taken
            if has_info and has_replace:
                # Check order: info must come before replacement
                info_idx = actions_taken.index("request_more_info")
                replace_idx = actions_taken.index("offer_replacement")
                if info_idx < replace_idx:
                    score += 0.5
                else:
                    score += 0.25 # Wrong order penalty

    # 3. Proper Termination (0.1 points)
    if actions_taken and actions_taken[-1] == "close_ticket":
        score += 0.1

    # 4. Efficiency Bonus (0.1 points)
    # Max bonus if steps equal to expected sequence
    expected_len = len(task["expected_sequence"])
    if len(actions_taken) <= expected_len:
        score += 0.1
    elif len(actions_taken) <= expected_len + 2:
        score += 0.05

    return float(min(max(score, 0.0), 1.0))
