"""
tools.py
Defines the three tools available to the agent: web search, a Python REPL,
and a calculator. Each is wrapped as a LangChain @tool so the LLM can decide
when to call it based on its docstring.
"""

import numexpr
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

# --- Web search -------------------------------------------------------------

web_search_tool = TavilySearchResults(
    max_results=4,
    name="web_search",
    description=(
        "Search the live web for current information — news, facts, prices, "
        "anything that might have changed since training or that you don't "
        "know. Input should be a short search query."
    ),
)

# --- Python REPL --------------------------------------------------------------

_repl = PythonREPL()


@tool
def python_repl(code: str) -> str:
    """
    Execute Python code and return stdout. Use this for data manipulation,
    multi-step logic, string processing, or anything a calculator can't do.
    Always print() the value you want returned — the REPL only returns stdout.
    Example: python_repl("print(sum(range(1, 101)))")
    """
    try:
        result = _repl.run(code)
        return result if result.strip() else "(code ran with no printed output)"
    except Exception as exc:  # noqa: BLE001 — surface the error to the agent
        return f"Error executing code: {exc}"


# --- Calculator ---------------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a single mathematical expression, e.g. "(4.5 * 12) / 3 - 7 ** 2".
    Faster and safer than python_repl for pure arithmetic — prefer this for
    simple math. Supports +, -, *, /, **, sqrt, sin, cos, log, etc.
    """
    try:
        result = numexpr.evaluate(expression).item()
        return str(result)
    except Exception as exc:  # noqa: BLE001
        return f"Could not evaluate expression: {exc}"


ALL_TOOLS = [web_search_tool, python_repl, calculator]
