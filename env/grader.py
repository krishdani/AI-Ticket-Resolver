from typing import List, Dict, Any
from .tasks import TASKS

def grade_trajectory(task_id: str, trajectory: List[Dict[str, Any]]) -> float:
    """
    Stage-aware deterministic grader.
    Evaluates progression through required milestones.
    """
    task = TASKS.get(task_id)
    if not task or not trajectory:
        return 0.0

    score = 0.0
    actions = [t["action"] for t in trajectory]
    payloads = [str(t.get("payload") or "").lower() for t in trajectory]
    
    # Check for Goal Completion
    success = False
    if task_id == "easy_spam_filter":
        success = "classify_issue" in actions and payloads[actions.index("classify_issue")] == "spam"
    elif task_id == "medium_refund":
        success = "offer_refund" in actions and actions.index("classify_issue") < actions.index("offer_refund")
    elif task_id == "hard_damaged_replacement":
        if "offer_replacement" in actions and "close_ticket" in actions:
            i_idx = actions.index("classify_issue")
            m_idx = actions.index("request_more_info")
            r_idx = actions.index("offer_replacement")
            c_idx = actions.index("close_ticket")
            success = i_idx < m_idx < r_idx < c_idx

    if success:
        score += 0.6 # Base success score
    
    # Sequence Quality & Efficiency (0.4)
    # Penalize if repeated actions exist (other than info request)
    uniques = set()
    has_repeats = False
    for a in actions:
        if a in uniques and a != "request_more_info":
            has_repeats = True
        uniques.add(a)
    
    if success and not has_repeats:
        score += 0.2
        
    # Step count efficiency
    ideal = len(task["expected_sequence"])
    actual = len(actions)
    if success and actual <= ideal:
        score += 0.2
    elif success and actual <= ideal + 1:
        score += 0.1

    return float(min(max(score, 0.0), 1.0))
