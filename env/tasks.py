from .models import TicketState

# We now define lists of tickets for each difficulty level
TASKS = {
    "easy_refund": [
        {
            "ticket_text": "I received my order but it's broken. I want a full refund immediately.",
            "customer_history": "First time customer. Order #12345.",
            "category": "refund",
            "priority": "low"
        },
        {
            "ticket_text": "Changed my mind about the blue shirt. I want a refund so I can buy the red one.",
            "customer_history": "Regular Shopper. Order #99283.",
            "category": "refund",
            "priority": "low"
        },
        {
            "ticket_text": "Double charged on my credit card for the last transaction. Please refund the extra $40.",
            "customer_history": "New user. No previous history.",
            "category": "refund",
            "priority": "medium"
        },
        {
            "ticket_text": "I bought a subscription but I haven't used it at all. Can I get a refund for this month?",
            "customer_history": "Subscriber since last month.",
            "category": "refund",
            "priority": "low"
        },
        {
            "ticket_text": "The item I received is completely different from the photo. Refund please.",
            "customer_history": "First order. Order #44552.",
            "category": "refund",
            "priority": "medium"
        }
    ],
    "medium_replacement": [
        {
            "ticket_text": "The phone I ordered has a cracked screen. I just want a working phone, no refund.",
            "customer_history": "Loyal customer since 2022. 5 previous orders.",
            "category": "replacement",
            "priority": "medium"
        },
        {
            "ticket_text": "Laptop keyboard is missing the 'E' key. Please send a replacement unit.",
            "customer_history": "Business account. Order #77812.",
            "category": "replacement",
            "priority": "high"
        },
        {
            "ticket_text": "Ordered size Large, received Small. Send the right size immediately.",
            "customer_history": "Occasional shopper.",
            "category": "replacement",
            "priority": "medium"
        },
        {
            "ticket_text": "Coffee machine arrived with a broken water tank. Swap it for a new one.",
            "customer_history": "Recent order #22312.",
            "category": "replacement",
            "priority": "medium"
        },
        {
            "ticket_text": "The headphones only work in one ear. Replacement requested.",
            "customer_history": "First time customer.",
            "category": "replacement",
            "priority": "low"
        }
    ],
    "hard_delay_case": [
        {
            "ticket_text": "My package says 'In Transit' but it's 2 days late. I need it for a birthday tomorrow!",
            "customer_history": "Regular customer. Previous delay reported 2 months ago.",
            "category": "shipping_delay",
            "priority": "high"
        },
        {
            "ticket_text": "Tracking number is invalid. Where is my $500 GPU?",
            "customer_history": "High-value customer.",
            "category": "shipping_delay",
            "priority": "high"
        },
        {
            "ticket_text": "Status hasn't updated in 4 days. Is it lost?",
            "customer_history": "First order.",
            "category": "shipping_delay",
            "priority": "medium"
        },
        {
            "ticket_text": "Missed the delivery window. When will it be re-attempted?",
            "customer_history": "Business address.",
            "category": "shipping_delay",
            "priority": "medium"
        },
        {
            "ticket_text": "Shipment stalled at the regional facility. Please check status.",
            "customer_history": "Long-term client.",
            "category": "shipping_delay",
            "priority": "low"
        }
    ]
}
