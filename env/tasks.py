from typing import Dict, Any

TASKS: Dict[str, Dict[str, Any]] = {
    "easy_spam_filter": {
        "name": "Easy — Spam Detection",
        "ticket_text": "URGENT: Win a free $500 gift card! Click here now!",
        "customer_history": "No previous purchases. Unknown sender.",
        "expected_category": "spam",
        "expected_sequence": ["classify_issue"],
        "difficulty": "easy",
        "description": "Identify the ticket as spam. This is a single-step task."
    },
    "medium_refund": {
        "name": "Medium — Standard Refund",
        "ticket_text": "I received my order #9988 but it's the wrong color. I'd like a refund please.",
        "customer_history": "Loyal customer since 2023. 3 previous orders.",
        "expected_category": "refund",
        "expected_sequence": ["classify_issue", "offer_refund"],
        "difficulty": "medium",
        "description": "Classify as a refund request and process the refund. Two steps required."
    },
    "hard_damaged_replacement": {
        "name": "Hard — Complex Replacement",
        "ticket_text": "The tablet I ordered for my kid's birthday arrived with a shattered screen. I need a replacement sent today, not a refund. Can you confirm if you have it in stock?",
        "customer_history": "VIP member. 15+ orders. High value account.",
        "expected_category": "replacement",
        "expected_sequence": ["classify_issue", "request_more_info", "offer_replacement", "close_ticket"],
        "difficulty": "hard",
        "description": "Handle a VIP replacement. Must classify, verify details, resolve, and close manually. 4 steps required."
    }
}
