# 🤖 Multi-Tool AI Agent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent-1C3C3C)
![Tavily](https://img.shields.io/badge/Search-Tavily-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A ReAct-style agent that reasons through multi-step tasks, deciding on its own when to search the web, run Python, or do arithmetic — and streams its thinking live in a chat UI.

## Demo

> _Add a screenshot or short GIF here: `docs/demo.gif`_

Example: *"What's the population of the 3 largest cities in Japan, and what's their combined total?"* → agent searches the web for each city, then uses the calculator to sum them, showing every step.

## How it works

Built as a small [LangGraph](https://langchain-ai.github.io/langgraph/) state machine:

```
        ┌─────────┐   tool call?    ┌─────────┐
 input →│  agent  │ ──────────────→ │  tools  │
        │ (LLM)   │ ←────────────── │(execute)│
        └────┬────┘   tool result   └─────────┘
             │ no tool call
             ▼
        final answer
```

The LLM sees all three tool definitions (with their docstrings as descriptions) and decides per-turn whether to call one, several in sequence, or none at all.

## Features

- Three tools: live web search (Tavily), a sandboxed Python REPL, and a calculator
- Streams every reasoning step to the UI — tool calls, tool outputs, and the final answer — not just the end result
- Multi-turn conversation memory within a session
- Clean separation between agent logic (`agent.py`), tool definitions (`tools.py`), and UI (`app.py`)

## Project structure

```
multi-tool-agent/
├── app.py       # Streamlit chat UI with step-by-step streaming
├── agent.py      # LangGraph agent graph (ReAct loop)
├── tools.py       # web_search, python_repl, calculator tool definitions
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/multi-tool-agent.git
cd multi-tool-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY and TAVILY_API_KEY (free tier at tavily.com)
streamlit run app.py
```

## Tech stack

`LangChain` · `LangGraph` · `Tavily` · `Streamlit` · `numexpr`

## Possible extensions

- Add a code-execution sandbox with resource limits (the current `python_repl` runs in-process — fine for a demo, not for untrusted input in production)
- Add memory/RAG as a fourth tool
- Swap the linear graph for a supervisor + sub-agent architecture for more complex tasks
- Add human-in-the-loop approval before executing `python_repl` calls

---

### Resume bullet points

- Built a multi-tool reasoning agent with LangGraph that autonomously selects between web search, code execution, and calculation tools based on task needs
- Implemented real-time streaming of agent reasoning steps (tool selection, tool output, final synthesis) to a Streamlit chat interface for full transparency into the agent's decision process
- Designed a modular tool-calling architecture separating agent orchestration, tool definitions, and UI, making it straightforward to add new tools

### Recruiter talking points

- **What it demonstrates:** understanding of agentic patterns beyond simple prompt chaining — state graphs, conditional routing, and tool-use loops.
- **Design decisions worth discussing:** why LangGraph over a plain while-loop agent; how the "should_continue" routing decision is made; trade-offs of streaming intermediate steps (transparency) vs. token cost.
- **What you'd improve at scale:** sandbox the Python REPL (e.g. via a container or restricted subprocess), add retries/timeouts per tool call, add tracing (LangSmith) for debugging agent behavior in production.
