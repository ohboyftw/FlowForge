"""
FlowForge — Unified Agent Orchestration Framework

Combines the best of PocketFlow, LangGraph, CrewAI, and Agno
into ~1600 lines of core Python with Pydantic-typed state.

Quick start:
    from flowforge import Agent, Team, StoreBase

    # Simple
    agent = Agent("Researcher", "Find papers")
    result = agent.run("What's new in AI?")

    # Team
    team = Team([agent1, agent2], strategy="sequential")
    team.run("Build something")

    # Full control
    flow = Flow()
    flow.add("step1", MyUnit())
    flow.wire("step1", "step2")
    flow.run(MyState())
"""

from flowforge.core import (
    AsyncUnit,
    FlexStore,
    Flow,
    FlowExhaustedError,
    FunctionUnit,
    InterruptSignal,
    ReducerRegistry,
    StoreBase,
    Unit,
    Wire,
)
from flowforge.harness import (
    Agent,
    AsyncLLMUnit,
    LLMUnit,
    Team,
    agent,
    team,
)
from flowforge.identity import (
    Persona,
    Personas,
    Task,
    TaskResult,
)

__version__ = "0.1.0"
__all__ = [
    "StoreBase",
    "FlexStore",
    "Unit",
    "AsyncUnit",
    "FunctionUnit",
    "Wire",
    "Flow",
    "FlowExhaustedError",
    "InterruptSignal",
    "ReducerRegistry",
    "Persona",
    "Task",
    "TaskResult",
    "Personas",
    "Agent",
    "Team",
    "LLMUnit",
    "AsyncLLMUnit",
    "agent",
    "team",
]
