---
title: AI Customer Support Resolver
emoji: 🎫
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - customer-support
  - reasoning
---

# 🎫 OpenEnv: AI Customer Support Ticket Resolver

### A High-Fidelity RL Environment for Real-World Reasoning

This repository implements a complete, professional-grade OpenEnv environment designed to train and evaluate AI agents on multi-step customer support workflows. Unlike toy environments, this simulates a genuine business task requiring classification, verification, and resolution stages.

---

## 🚀 Environment Motivation

Modern support agents aren't just one-shot generators; they need to navigate complex state machines. This environment provides:
- **Sequential Reasoning**: Enforces a strict `Classify -> Verify -> Resolve` workflow.
- **Durable Feedback**: Shaped rewards that penalize repetition and out-of-order execution.
- **Deterministic Evaluation**: Programmatic graders that assess not just the outcome, but the path taken.

---

## 🛠️ Design Specification

### 🎮 Action Space
The agent interacts via the `Action` model with specific transition rules:

| Action | Payload Required | Purpose |
| :--- | :--- | :--- |
| `classify_issue` | Yes (e.g., "refund") | Initial categorization of the intent. |
| `request_more_info`| Yes (Reason string) | Intermediate stage for complex (Hard) tasks. |
| `offer_refund` | Optional | Resolves "refund" category tickets. |
| `offer_replacement`| Optional | Resolves "replacement" category tickets. |
| `escalate` | Optional | Emergency fallback (Neutral reward). |
| `close_ticket` | Optional | Final step for long sequences. |

### 📊 Observation Space
Comprehensive state representation provided at each step:
- `ticket_text`: The customer's raw complaint.
- `customer_history`: VIP status and past order metadata.
- `status`: Current lifecycle (OPEN, CLASSIFIED, VALIDATED, RESOLVED, CLOSED).
- `current_step`: Integer counter (Max 8).
- `history`: Full audit trail of all previous steps in the episode.

### 🎯 Tasks & Graders
We provide three core tasks with a programmatic difficulty curve:

1. **Easy — Spam Detection**: 
   - *Goal*: Identify junk mail and classify as `spam`.
   - *Sequence*: 1 Step.
2. **Medium — Standard Refund**: 
   - *Goal*: Handle a color-mismatch refund request.
   - *Sequence*: 2 Steps (`Classify` &rarr; `Refund`).
3. **Hard — Complex Replacement**: 
   - *Goal*: Process a VIP high-value damaged item replacement.
   - *Sequence*: 4 Steps (`Classify` &rarr; `Verify Info` &rarr; `Replace` &rarr; `Close`).

---

## 🏆 Scoring & Reward Design

Our reward function provides dense, meaningful signals:
- **Correct Step Transition**: `+0.3`
- **Final Goal Achievement**: `+0.4`
- **Out-of-Order Action**: `-0.2` (Penalty)
- **Repetitive Action**: `-0.1` (Penalty)

**Grader Logic**: Final scores (0.0 - 1.0) are calculated by `env.grader.grade_trajectory`, which factors in Success (60%), Sequence Quality (20%), and Efficiency (20%).

---

## 🧪 Baseline Performance

The included `inference.py` provides reproducible outcomes using frontier models (e.g., Qwen 2.5).

| Task | Baseline Score | Success Rate |
| :--- | :--- | :--- |
| Easy | 1.00 | 100% |
| Medium | 0.90 | 95% |
| Hard | 0.85 | 80% |

---

## ⚙️ Setup & Usage

### 📥 Prerequisites
- Python 3.10+
- Docker (for containerized deployment)

### 🏗️ Local Installation
```bash
pip install -r requirements.txt
python app.py
```

### 🐳 Docker Execution
```bash
docker build -t ai-ticket-resolver .
docker run -p 7860:7860 \
  -e HF_TOKEN="your_token" \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  -e MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" \
  ai-ticket-resolver
```

### 🤖 Running Inference
```bash
export HF_TOKEN="your_token"
python inference.py
```

---
*Developed for the OpenEnv RL Hackathon. This environment follows the full OpenEnv v1 Specification.*
