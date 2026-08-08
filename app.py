"""
app.py
Streamlit chat UI for the multi-tool agent. Streams each reasoning step
(tool calls, tool results, final answer) as they happen.
"""

import os

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from agent import build_agent, run_agent_streaming

st.set_page_config(page_title="Multi-Tool AI Agent", page_icon="🤖", layout="wide")

st.title("🤖 Multi-Tool AI Agent")
st.caption("Reasons step by step using web search, a Python REPL, and a calculator.")

if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
    st.warning("Set OPENAI_API_KEY and TAVILY_API_KEY in your .env file before chatting.", icon="⚠️")

if "agent" not in st.session_state:
    st.session_state.agent = build_agent()
if "history" not in st.session_state:
    st.session_state.history = []  # LangChain message objects, for the agent
if "display_log" not in st.session_state:
    st.session_state.display_log = []  # what's rendered in the chat window

with st.sidebar:
    st.header("Tools available")
    st.markdown("- 🌐 **web_search** — Tavily live search\n- 🐍 **python_repl** — sandboxed Python\n- 🧮 **calculator** — arithmetic")
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.session_state.display_log = []
        st.rerun()

for entry in st.session_state.display_log:
    role = "user" if entry["role"] == "user" else "assistant"
    with st.chat_message(role):
        if entry["role"] == "user":
            st.markdown(entry["content"])
        else:
            for step_type, content in entry["steps"]:
                if step_type == "final":
                    st.markdown(content)
                else:
                    st.markdown(f"<small>{content}</small>", unsafe_allow_html=True)

user_input = st.chat_input("Ask me anything — I can search, calculate, or run code...")

if user_input:
    st.session_state.display_log.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        steps_container = st.container()
        rendered_steps = []

        for step_type, content in run_agent_streaming(
            st.session_state.agent, user_input, st.session_state.history
        ):
            rendered_steps.append((step_type, content))
            with steps_container:
                if step_type == "final":
                    st.markdown(content)
                else:
                    st.markdown(f"<small>{content}</small>", unsafe_allow_html=True)

        st.session_state.display_log.append({"role": "assistant", "steps": rendered_steps})

    st.session_state.history.append(HumanMessage(content=user_input))
    final_answers = [c for t, c in rendered_steps if t == "final"]
    if final_answers:
        st.session_state.history.append(AIMessage(content=final_answers[-1]))
