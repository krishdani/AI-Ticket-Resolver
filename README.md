# 🎫 AI Customer Support Ticket Resolver (OpenEnv)

This repository contains a Reinforcement Learning (RL) environment for training and evaluating AI agents on customer support tasks. Built using the **OpenEnv** specification.

## 🚀 Overview

The environment simulates a customer support dashboard where an RL agent must resolve incoming tickets. The agent can classify issues, request information, offer refunds or replacements, and escalate or close tickets.

### ✨ Key Features
- **Real-world Scenarios**: Tasks range from simple refunds to complex shipping delays.
- **Dense Reward System**: Immediate feedback for correct classifications and actions.
- **Deterministic Grading**: Transparent scoring logic (0.01 - 0.99).
- **Premium UI**: Interactive dashboard for manual testing and visualization.

## 🛠️ Project Structure
```text
project/
├── env/
│   ├── env.py      # Core RL logic
│   ├── models.py   # Pydantic schemas
│   ├── tasks.py    # Task definitions
│   └── grader.py   # Trajectory scoring
├── static/         # Frontend assets
├── openenv.yaml    # Environment configuration
├── inference.py    # Baseline agent script
├── app.py          # FastAPI Server
└── Dockerfile      # Containerization
```

## 🏃 Running Locally

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Environment**:
   ```bash
   python app.py
   ```
   Open `http://localhost:7860` to access the interactive dashboard.

3. **Run Baseline Inference**:
   ```bash
   python inference.py
   ```

## 🎮 Action Space
- `classify_issue`: Identify the core problem.
- `request_more_info`: Ask the customer for details.
- `offer_refund`: Resolve via money back.
- `offer_replacement`: Resolve via new product.
- `escalate`: Send to a human agent.
- `close_ticket`: Finalize the session.

## 📊 Reward Function
| Action | Reward |
| :--- | :--- |
| Correct Classification | +0.3 |
| Correct Resolution | +0.5 |
| Closing Resolved Ticket | +0.2 |
| Incorrect Action | -0.3 |
| Unnecessary Step | -0.1 |

---
*Created for the OpenEnv RL Hackathon.*
