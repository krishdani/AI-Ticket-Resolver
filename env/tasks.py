from typing import Dict, Any

TASKS: Dict[str, Dict[str, Any]] = {
    "easy_spam": {
        "name": "Easy — Spam Filtering",
        "ticket_text": "Buy crypto now! Limited time offer at www.scam-site.com",
        "customer_history": "Unknown sender. Marked as potential spam by system.",
        "expected_category": "spam",
        "expected_sequence": ["close_ticket"],
        "difficulty": "easy",
        "description": "Identify spam and close the ticket immediately without wasting company time."
    },
    "medium_refund": {
        "name": "Medium — Refund Request",
        "ticket_text": "My Order #4455 was delivered but the box was empty. I need my money back.",
        "customer_history": "Regular customer. One previous order 6 months ago.",
        "expected_category": "refund",
        "expected_sequence": ["classify_issue", "offer_refund", "close_ticket"],
        "difficulty": "medium",
        "description": "Classify the refund request, offer the refund, and then close the ticket."
    },
    "hard_damaged_replacement": {
        "name": "Hard — Complex Replacement",
        "ticket_text": "The laptop I bought from your store has a dead pixel. I need it for a presentation on Tuesday. Can you send a new one fast? I don't want a refund, I want the unit.",
        "customer_history": "VIP Customer. High spend. Has bought 10+ items.",
        "expected_category": "replacement",
        "expected_sequence": ["classify_issue", "request_more_info", "offer_replacement", "close_ticket"],
        "difficulty": "hard",
        "description": "A VIP customer needs urgent help. Must classify, ask for proof/details, offer replacement, and close."
    }
}
