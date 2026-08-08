"""
agent.py
A ReAct-style agent built with LangGraph: the LLM decides which tool to call
(if any), the tool executes, the result goes back to the LLM, and the loop
repeats until the LLM produces a final answer. Each step is yielded so the
UI can stream the agent's reasoning live.
"""

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from tools import ALL_TOOLS

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are a helpful, precise assistant with access to three tools:

- web_search: for current events, facts, or anything you're unsure about
- python_repl: for multi-step logic, data manipulation, or code
- calculator: for arithmetic and math expressions

Think step by step. Only call a tool when you actually need it — for things \
you already know confidently, just answer directly. When you do use a tool, \
briefly say why before calling it. Once you have enough information, give a \
clear final answer and stop calling tools."""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_agent():
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    tools_by_name = {t.name: t for t in ALL_TOOLS}

    def call_model(state: AgentState) -> AgentState:
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def call_tools(state: AgentState) -> AgentState:
        last_message: AIMessage = state["messages"][-1]
        tool_messages = []
        for tool_call in last_message.tool_calls:
            tool = tools_by_name[tool_call["name"]]
            result = tool.invoke(tool_call["args"])
            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"], name=tool_call["name"])
            )
        return {"messages": tool_messages}

    def should_continue(state: AgentState) -> str:
        last_message: AIMessage = state["messages"][-1]
        return "tools" if getattr(last_message, "tool_calls", None) else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", call_tools)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


def run_agent_streaming(agent, user_input: str, chat_history: list[BaseMessage]):
    """
    Yields (step_type, content) tuples as the agent works, so the caller can
    render each reasoning step / tool call / tool result live:
      ("tool_call", "Calling web_search(query='...')")
      ("tool_result", "...")
      ("final", "...")
    """
    state = {"messages": chat_history + [("user", user_input)]}

    for step in agent.stream(state, stream_mode="values"):
        last = step["messages"][-1]

        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            for tc in last.tool_calls:
                args_str = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                yield "tool_call", f"🔧 Calling **{tc['name']}**({args_str})"

        elif isinstance(last, ToolMessage):
            preview = last.content if len(last.content) < 500 else last.content[:500] + "..."
            yield "tool_result", f"↳ {preview}"

        elif isinstance(last, AIMessage) and last.content and not getattr(last, "tool_calls", None):
            yield "final", last.content
