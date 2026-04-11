---
title: AI Customer Support Resolver
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - customer-support
---

# 🎫 AI Customer Support Ticket Resolver (OpenEnv)

This repository contains a Reinforcement Learning (RL) environment for training and evaluating AI agents on real-world customer support tasks. Built using the **OpenEnv** specification.

## 🚀 Quick Start

The environment simulates a customer support dashboard. You can run it locally with Docker or via Python.

### 🏃 Running with Docker
```bash
docker build -t ai-ticket-resolver .
docker run -p 7860:7860 ai-ticket-resolver
```

### 🐍 Running with Python
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Start the Server**:
   ```bash
   python app.py
   ```
   Open `http://localhost:7860` to access the interactive dashboard.

## 🧠 Environment Design

### 🎯 Use Case
The environment models a genuine customer support task where an agent must handle incoming tickets. It requires multi-step reasoning:
1. **Understand** the customer's issue from text and history.
2. **Classify** the problem correctly.
3. **Decide** on the best resolution (Refund, Replacement, or more Info).
4. **Close** the ticket only when the goal is met.

### 🛠️ Project Structure
```text
ai-ticket-resolver/
├── env/
│   ├── env.py          # Core RL Environment logic (OpenEnv compatible)
│   ├── models.py       # Pydantic Action/Observation schemas
│   ├── tasks.py        # Task definitions (Easy, Medium, Hard)
│   └── grader.py       # Deterministic trajectory scoring (0.0 - 1.0)
├── static/             # Premium HTML/CSS dashboard
├── openenv.yaml        # OpenEnv manifest and schemas
├── inference.py        # Baseline LLM inference script (Mandatory logs)
├── app.py              # FastAPI Server (HTTP endpoints)
└── Dockerfile          # Container image definition
```

### 🎮 Action Space
The agent takes actions using the `Action` model:
- `classify_issue`: Categorize the ticket (e.g., "refund", "replacement").
- `request_more_info`: Ask for details if the case is ambiguous.
- `offer_refund`: Process a refund for broken items.
- `offer_replacement`: Process a replacement for defective items.
- `escalate`: Transfer to a human supervisor (useful for complex/unsolvable cases).
- `close_ticket`: Finalize the interaction.

### 📊 Observation Space
The `TicketState` observation includes:
- `ticket_text`: The current customer complaint.
- `customer_history`: Past interactions and loyalty info.
- `status`: Current lifecycle (open, classified, resolved, etc.).
- `history`: List of all actions taken in the current episode.

## 🏆 Scoring & Rewards
- **Dense Rewards**: Immediate feedback for each step (e.g., +0.3 for correct classification).
- **Deterministic Grading**: Final score (0.0 - 1.0) calculated by `grader.py` based on the efficiency and outcome of the full trajectory.

| Milestone | Reward |
| :--- | :--- |
| Correct Classification | +0.3 |
| Correct Resolution | +0.5 |
| Closing Resolved Ticket | +0.2 |
| Incorrect Action | -0.3 |

## 🧪 Baseline Inference
The `inference.py` script runs a baseline agent against all tasks. It uses the OpenAI API client and outputs mandatory structured logs:
```bash
export API_BASE_URL="your-api-url"
export MODEL_NAME="your-model-name"
export HF_TOKEN="your-hf-token"
python inference.py
```

---
*Created for the OpenEnv RL Hackathon. This environment fills a gap in real-world reasoning tasks for LLM-based agents.*
