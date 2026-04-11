from .models import TicketState

TASKS = {
    "easy_spam_case": {
        "ticket_text": "Spam: Buy cheap followers now! Check this link for deals.",
        "customer_history": "Unknown user. No previous interactions.",
        "expected_category": "spam",
        "expected_sequence": ["close_ticket"],
        "difficulty": "easy",
        "description": "Simply close the spam ticket."
    },
    "medium_refund": {
        "ticket_text": "The item I bought is broken. I want a refund.",
        "customer_history": "Regular customer. Order #77889.",
        "expected_category": "refund",
        "expected_sequence": ["classify_issue", "offer_refund", "close_ticket"],
        "difficulty": "medium",
        "description": "Classify as refund and then offer the refund."
    },
    "hard_complex_replacement": {
        "ticket_text": "I received a different model than what I ordered. I need the correct one for a wedding on Saturday. Can you fix this?",
        "customer_history": "Prime member. 12 orders this year.",
        "expected_category": "replacement",
        "expected_sequence": ["classify_issue", "request_more_info", "offer_replacement", "close_ticket"],
        "difficulty": "hard",
        "description": "Must classify, ask for the correct model details (request_more_info), then offer replacement."
    }
}
