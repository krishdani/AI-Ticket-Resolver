from .models import TicketState

TASKS = {
    "easy_refund": {
        "ticket_text": "I received my order but it's broken. I want a full refund immediately.",
        "customer_history": "First time customer. Order #12345.",
        "expected_category": "refund",
        "expected_sequence": ["classify_issue", "offer_refund", "close_ticket"],
        "difficulty": "easy"
    },
    "medium_replacement": {
        "ticket_text": "The phone I ordered has a cracked screen. I don't want my money back, I just want a working phone.",
        "customer_history": "Loyal customer since 2022. 5 previous orders.",
        "expected_category": "replacement",
        "expected_sequence": ["classify_issue", "offer_replacement", "close_ticket"],
        "difficulty": "medium"
    },
    "hard_delay_case": {
        "ticket_text": "My package was supposed to arrive yesterday. It still says 'In Transit'. I'm very frustrated, but I still really need the item for a birthday tomorrow.",
        "customer_history": "Regular customer. Previous delay reported 2 months ago.",
        "expected_category": "shipping_delay",
        "expected_sequence": ["classify_issue", "request_more_info", "close_ticket"],
        "difficulty": "hard",
        "note": "Agent should NOT refund if the customer still wants the item."
    }
}
