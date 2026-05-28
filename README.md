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
  - llm-agents
---

# 🎫 AI Customer Support Ticket Resolver (OpenEnv)

An advanced Reinforcement Learning (RL) environment for training and evaluating AI agents on real-world customer support tasks. Built using the **OpenEnv** specification to standardize RL evaluation benchmarks.

> **Created for the OpenEnv RL Hackathon** - This environment addresses the gap in real-world reasoning tasks for LLM-based agents.

## 🌟 Key Features

- ✅ **OpenEnv Compliant**: Standardized RL environment specification
- ✅ **LLM-Ready**: Designed for Language Model-based agents with structured action/observation spaces
- ✅ **Multi-Difficulty Tasks**: Easy, Medium, and Hard difficulty levels
- ✅ **Deterministic Scoring**: Reproducible trajectory grading system
- ✅ **Docker Support**: Ready-to-deploy containerized environment
- ✅ **Interactive Dashboard**: Web-based UI for visualization and testing

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
docker build -t ai-ticket-resolver .
docker run -p 7860:7860 ai-ticket-resolver
```

### Option 2: Local Python
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Server**:
   ```bash
   python app.py
   ```
   
3. **Access the Dashboard**:
   Open `http://localhost:7860` in your browser

## 📋 Project Structure

```
ai-ticket-resolver/
├── env/
│   ├── env.py              # Core RL Environment (OpenEnv compatible)
│   ├── models.py           # Pydantic Action/Observation schemas
│   ├── tasks.py            # Task definitions (Easy, Medium, Hard)
│   └── grader.py           # Deterministic trajectory scoring (0.0 - 1.0)
├── server/                 # FastAPI server code
│   └── app.py              # HTTP endpoints and server logic
├── static/                 # HTML/CSS dashboard UI
├── openenv.yaml            # OpenEnv manifest and specification
├── inference.py            # Baseline LLM inference script
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project configuration
├── Dockerfile              # Container image definition
└── README.md               # This file
```

## 🧠 Environment Design

### 🎯 Problem Statement
The environment simulates a genuine customer support task where an AI agent must handle incoming tickets. The agent needs to:

1. **Understand** the customer's issue from text and interaction history
2. **Classify** the problem correctly (e.g., refund request, replacement needed)
3. **Decide** the best resolution (Refund, Replacement, or request more info)
4. **Close** the ticket appropriately to maximize customer satisfaction

### 🎮 Action Space

Agents can take the following actions:

| Action | Description | Payload |
|--------|-------------|---------|
| `classify_issue` | Categorize the ticket | Issue category (string) |
| `request_more_info` | Ask customer for clarification | Question/request (string) |
| `offer_refund` | Process a refund | Refund details (string) |
| `offer_replacement` | Process a replacement | Replacement details (string) |
| `escalate` | Transfer to human supervisor | Reason (string) |
| `close_ticket` | Finalize the interaction | Resolution summary (string) |

### 📊 Observation Space

The `TicketState` observation includes:

```python
{
    "ticket_id": "str",                    # Unique ticket identifier
    "ticket_text": "str",                  # Customer's complaint/request
    "customer_history": "List[str]",       # Past interactions and loyalty info
    "status": "str",                       # Current lifecycle status
    "history": "List[Action]"              # All actions taken in episode
}
```

### 🏆 Reward Structure

Agents receive immediate feedback for each step:

| Action | Reward |
|--------|--------|
| Correct Classification | +0.3 |
| Correct Resolution | +0.5 |
| Closing Resolved Ticket | +0.2 |
| Incorrect Action | -0.3 |
| Invalid Action | -0.1 |

**Final Score**: Deterministic grading (0.0 - 1.0) calculated by `grader.py` based on trajectory efficiency and outcome.

## 🧪 Running Baseline Inference

The `inference.py` script provides a baseline agent using OpenAI-compatible API:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-huggingface-token"
python inference.py
```

### Output Format

The script follows OpenEnv mandatory logging format:

```
[START] task=easy env=ai_customer_support_resolver model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=classify_issue('billing_issue') reward=0.30 done=false error=null
[STEP] step=2 action=offer_refund('Full refund issued') reward=0.50 done=false error=null
[STEP] step=3 action=close_ticket('Issue resolved') reward=0.20 done=true error=null
[END] success=true steps=3 score=1.00 rewards=0.30,0.50,0.20
```

## 💻 API Endpoints

The FastAPI server provides the following endpoints:

### `POST /reset`
Reset the environment for a new episode.

**Request**:
```json
{
  "task_id": "easy|medium|hard"
}
```

**Response**:
```json
{
  "observation": {...},
  "episode_id": "uuid"
}
```

### `POST /step`
Take an action in the environment.

**Request**:
```json
{
  "action": {
    "action_type": "classify_issue",
    "payload": "billing_issue"
  }
}
```

**Response**:
```json
{
  "observation": {...},
  "reward": 0.30,
  "done": false,
  "info": {...}
}
```

### `GET /` 
Serves the interactive dashboard.

## 📦 Dependencies

- **FastAPI** - Web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation
- **OpenAI** - LLM API client
- **openenv-core** - OpenEnv specification library
- **python-dotenv** - Environment configuration

See `requirements.txt` for complete dependency list.

## 🔧 Configuration

Create a `.env` file in the project root:

```env
# LLM Configuration
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
HF_TOKEN=your-huggingface-token

# Server Configuration
SERVER_PORT=7860
DEBUG=false
```

## 📈 Evaluation Metrics

- **Success Rate**: Percentage of episodes that achieve score ≥ 0.7
- **Average Score**: Mean trajectory score across all episodes
- **Step Efficiency**: Average steps to complete tasks
- **Reward Per Step**: Mean reward per action taken

## 🎓 Use Cases

- **Agent Benchmarking**: Evaluate LLM-based agents on customer support reasoning
- **Training Data**: Generate synthetic support conversations for fine-tuning
- **Multi-Task Learning**: Develop agents that handle multiple support scenarios
- **Human-in-the-Loop**: Combine AI recommendations with human oversight

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 📚 References

- [OpenEnv Specification](https://openenv.dev)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Pydantic Documentation](https://docs.pydantic.dev)

## 🙋 Support

For issues, questions, or suggestions, please open an [GitHub Issue](https://github.com/krishdani/AI-Ticket-Resolver/issues).

---

**Built with ❤️ for the OpenEnv RL Hackathon**

*Last Updated: 2026-05-28*
