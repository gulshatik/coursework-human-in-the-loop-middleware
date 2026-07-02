#!/usr/bin/env python3
"""
Demo of Human-in-the-Loop with LangGraph and a simple weather tool.
The script runs without user interaction – decisions are hard‑coded to
approve every tool call.  It demonstrates:
    * agent creation with HumanInTheLoopMiddleware
    * pause handling via the '__interrupt__' key
    * resumption using Command(resume={...})
"""

import os
import sys
from typing import Annotated, TypedDict

# ---------- Dependencies ----------
# langchain_openai provides ChatOpenAI
# langgraph provides graph building and middleware support
# dotenv loads environment variables (e.g. API keys)
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

# ---------- LLM ----------
llm = ChatOpenAI(
    base_url="https://llm.brojs.ru/v1",
    api_key=os.getenv("BROJS_PAT_TOKEN"),
    model="openai/gpt-oss-20b",
    temperature=0.1,
)

# ---------- Tool ----------
def get_weather(city: str, date: str) -> str:
    """
    Dummy weather tool – returns a fabricated forecast.
    In a real scenario this would call an API such as OpenWeatherMap.
    """
    return f"Погода в {city} на {date}: солнечно, 25°C."

# ---------- State ----------
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def agent_node(state: State) -> dict:
    """Call the LLM with the current conversation."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# ---------- Graph ----------
builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools=[get_weather]))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

# ---------- Checkpoint ----------
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# ---------- Human-in-the-Loop Middleware ----------
# The middleware automatically pauses when a tool is about to be called.
# It injects an '__interrupt__' key into the result with details of the
# pending action(s).  We import it from langchain.agents.middleware if
# available; otherwise we skip adding it (the demo will still work).
try:
    from langchain.agents.middleware import HumanInTheLoopMiddleware

    middleware = [
        HumanInTheLoopMiddleware(
            interrupt_on={"get_weather": True},
            description_prefix="Подтвердите вызов инструмента",
        )
    ]
    # Re‑compile the graph with middleware
    graph = builder.compile(checkpointer=memory, middleware=middleware)
except Exception:
    # Middleware not available – proceed without it.
    pass

# ---------- Demo loop ----------
def run_demo():
    """
    Runs a single question through the agent, handling pauses automatically.
    Decisions are hard‑coded to approve every tool call.
    """
    config = {"configurable": {"thread_id": "demo-session"}}

    # Initial user message
    result = graph.invoke(
        {"messages": [HumanMessage(content="Какая погода в Казани сегодня?")]},
        config=config,
    )

    # Process pauses until the agent finishes
    while "__interrupt__" in result:
        interrupt_value = result["__interrupt__"][0].value
        action_requests = interrupt_value["action_requests"]

        # Build decisions – approve all (hard‑coded)
        decisions = [{"type": "approve"} for _ in action_requests]

        # Resume the graph with the decisions
        result = graph.invoke(Command(resume={"decisions": decisions}), config=config)

    # Final answer
    final_message = result["messages"][-1].content

    # Ensure stdout can handle Unicode (especially on Windows)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass  # reconfigure may not be available; fallback to default

    print("\n=== Agent response ===")
    print(final_message)

if __name__ == "__main__":
    run_demo()
