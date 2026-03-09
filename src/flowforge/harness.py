"""
FlowForge — Harness Layer
Agno-grade ergonomics: Agent and Team as the developer-facing API.

Agent wraps a Persona + tools + LLM into a single callable.
Team compiles multiple Agents into a Flow with a chosen strategy.

The key design principle: everything the Harness does, you could
do manually with Flow + Unit + Wire. The Harness just saves keystrokes
and encodes best-practice patterns.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .core import FlexStore, Flow, FunctionUnit, ReducerRegistry, StoreBase, Unit
from .identity import Persona, Task

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM UNIT — A Unit that calls an LLM with Persona context
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Type for LLM calling functions
# Signature: (system: str, user: str, tools: list) -> str
LLMFunction = Callable[..., str]


class LLMUnit(Unit):
    """
    A Unit that calls an LLM with Persona system prompt and Task user prompt.

    This is the bridge between the Identity layer (Persona/Task) and
    the Primitive layer (Unit lifecycle).

    The LLM function is injected — FlowForge doesn't care if you use
    OpenAI, Anthropic, Ollama, or LiteLLM. Just pass a callable.
    """

    def __init__(
        self,
        persona: Persona,
        task: Task,
        llm_fn: LLMFunction,
        tools: list = None,
        output_field: str = None,
    ):
        self.persona = persona
        self.task = task
        self.llm_fn = llm_fn
        self.tools = tools or []
        self.output_field = output_field or task.output_field or task.description

    def prep(self, store: StoreBase) -> dict:
        """Extract context from store and compile prompts."""
        # Build context dict from store fields
        context = {}
        for field_name in self.task.context_from:
            if hasattr(store, field_name):
                context[field_name] = getattr(store, field_name)

        return {
            "system": self.persona.to_prompt(),
            "user": self.task.to_prompt(context),
            "tools": self.tools,
        }

    def exec(self, prep_result: dict) -> str:
        """Call the LLM. Pure computation — no store access."""
        start = time.monotonic()
        result = self.llm_fn(
            system=prep_result["system"],
            user=prep_result["user"],
            tools=prep_result.get("tools", []),
        )
        elapsed = (time.monotonic() - start) * 1000
        return {"output": result, "latency_ms": elapsed}

    def post(self, store: StoreBase, exec_result: dict) -> str:
        """Write result to store. Return action for routing."""
        output = exec_result["output"]

        # Write to the designated output field
        if hasattr(store, self.output_field) or isinstance(store, FlexStore):
            setattr(store, self.output_field, output)

        return "default"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT — Single-agent harness
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class Agent:
    """
    Single-agent harness. The simplest way to use FlowForge.

    Usage:
        # Quick — just role + goal
        agent = Agent("Researcher", "Find papers on X",
                       model="claude-sonnet-4-20250514")
        result = agent.run("Search for recent LLM agent papers")

        # Full control
        agent = Agent(
            role="Senior Researcher",
            goal="Find accurate information",
            backstory="PhD in ML",
            tools=[web_search, arxiv_search],
            llm_fn=my_litellm_caller,
            constraints=["Cite sources"],
        )
        result = agent.run("Find papers on X", store=my_state)

    Escape hatch: agent.as_unit(task) returns the underlying LLMUnit
    for manual Flow composition.
    """

    def __init__(
        self,
        role: str,
        goal: str,
        *,
        backstory: str = "",
        tools: list = None,
        model: str = None,
        llm_fn: LLMFunction = None,
        constraints: list[str] = None,
    ):
        self.persona = Persona(
            role=role,
            goal=goal,
            backstory=backstory,
            constraints=constraints or [],
        )
        self.tools = tools or []
        self.model = model
        self.llm_fn = llm_fn or _make_default_llm(model)
        self._name = role.lower().replace(" ", "_")

    @property
    def name(self) -> str:
        return self._name

    def run(
        self,
        task_desc: str,
        *,
        store: StoreBase = None,
        expected_output: str = "",
        context_from: list[str] = None,
        output_field: str = None,
    ) -> Any:
        """
        Execute a single task. Returns the result string.

        If no store is provided, creates a FlexStore.
        """
        store = store or FlexStore()
        task = Task(
            description=task_desc,
            expected_output=expected_output,
            context_from=context_from or [],
            output_field=output_field or "result",
        )
        unit = self.as_unit(task)
        unit.run(store)

        # Return from the output field
        out_field = output_field or "result"
        if hasattr(store, out_field):
            return getattr(store, out_field)
        return None

    def as_unit(self, task: Task) -> LLMUnit:
        """
        Get the underlying LLMUnit for manual Flow composition.

        Escape hatch: use this when you need to add the agent
        as a node in a custom Flow.
        """
        return LLMUnit(
            persona=self.persona,
            task=task,
            llm_fn=self.llm_fn,
            tools=self.tools,
        )

    def __repr__(self) -> str:
        return f"Agent(role='{self.persona.role}', goal='{self.persona.goal}')"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEAM — Multi-agent harness that compiles to a Flow
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class Team:
    """
    Multi-agent harness. Compiles Agents into a Flow graph.

    Strategies:
        "sequential"    — agents run one after another (pipeline)
        "parallel"      — all agents run, results merge via reducers
        "hierarchical"  — manager decomposes and delegates to workers
        "consensus"     — all agents run, then a vote/merge step

    Usage:
        team = Team(
            [researcher, coder, reviewer],
            strategy="sequential",
        )
        result = team.run("Build a CLI tool")

    Escape hatch: team.graph gives you the compiled Flow
    for manual Wire additions.

        g = team.graph
        g.wire("reviewer", "coder",
               on="needs_fix",
               when=lambda s: s.confidence < 0.8)
    """

    STRATEGIES = ["sequential", "parallel", "hierarchical", "consensus"]

    def __init__(
        self,
        agents: list[Agent],
        *,
        strategy: str = "sequential",
        manager: Agent = None,
        store_class: type[StoreBase] = None,
        reducers: dict[str, str] = None,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Options: {self.STRATEGIES}")
        self.agents = agents
        self.strategy = strategy
        self.manager = manager
        self.store_class = store_class
        self._reducers = ReducerRegistry(reducers) if reducers else ReducerRegistry()
        self._flow: Flow | None = None

    def compile(self, task_desc: str = "") -> Flow:
        """
        Compile agents + strategy into a Flow graph.

        Returns the Flow for inspection or modification before execution.
        """
        builder = getattr(self, f"_build_{self.strategy}")
        self._flow = builder(task_desc)
        return self._flow

    @property
    def graph(self) -> Flow:
        """Access the compiled Flow. Compiles if needed."""
        if not self._flow:
            self.compile()
        return self._flow

    def run(
        self,
        task_desc: str,
        *,
        store: StoreBase = None,
    ) -> StoreBase:
        """
        Execute the team on a task. Returns the final store state.
        """
        flow = self.compile(task_desc)

        if store is None:
            if self.store_class:
                store = self.store_class(task=task_desc)
            else:
                store = FlexStore(task=task_desc)

        return flow.run(store)

    def describe(self) -> str:
        """Human-readable description of the compiled flow."""
        if not self._flow:
            self.compile()
        return self._flow.describe()

    # ── Strategy builders ──

    def _build_sequential(self, task_desc: str = "") -> Flow:
        """Chain agents linearly: A → B → C"""
        flow = Flow(reducers=self._reducers)

        for i, agent in enumerate(self.agents):
            task = Task(
                description=task_desc if i == 0 else f"Continue: {task_desc}",
                context_from=[self.agents[j].name for j in range(i)],
                output_field=agent.name,
            )
            unit = agent.as_unit(task)
            flow.add(agent.name, unit)

            if i > 0:
                flow.wire(self.agents[i - 1].name, agent.name)

        flow.entry(self.agents[0].name)
        return flow

    def _build_parallel(self, task_desc: str = "") -> Flow:
        """Fan-out to all agents, fan-in at the end."""
        flow = Flow(reducers=self._reducers)

        # Dispatcher node
        flow.add("dispatch", FunctionUnit(lambda s: "default"))

        # Worker nodes
        for agent in self.agents:
            task = Task(
                description=task_desc,
                output_field=agent.name,
            )
            flow.add(agent.name, agent.as_unit(task))

        # Fan-out from dispatch to all workers
        targets = [a.name for a in self.agents]
        flow.wire("dispatch", targets)

        flow.entry("dispatch")
        return flow

    def _build_hierarchical(self, task_desc: str = "") -> Flow:
        """Manager decomposes, workers execute, manager synthesizes."""
        flow = Flow(reducers=self._reducers)

        mgr = self.manager or Agent(
            "Manager",
            "Decompose and delegate tasks",
            llm_fn=self.agents[0].llm_fn,  # inherit LLM from first agent
        )

        # Manager decomposition
        decompose_task = Task(
            description=f"Decompose this task into subtasks for your team: {task_desc}",
            expected_output="List of subtasks with assignments",
            output_field="subtasks",
        )
        flow.add("decompose", mgr.as_unit(decompose_task))

        # Worker nodes
        for agent in self.agents:
            task = Task(
                description=task_desc,
                context_from=["subtasks"],
                output_field=agent.name,
            )
            flow.add(agent.name, agent.as_unit(task))
            flow.wire("decompose", agent.name)

        # Manager synthesis
        synthesize_task = Task(
            description="Synthesize team outputs into final result",
            context_from=[a.name for a in self.agents],
            output_field="final_result",
        )
        flow.add("synthesize", mgr.as_unit(synthesize_task))

        for agent in self.agents:
            flow.wire(agent.name, "synthesize")

        flow.entry("decompose")
        return flow

    def _build_consensus(self, task_desc: str = "") -> Flow:
        """All agents work in parallel, then vote/merge."""
        flow = self._build_parallel(task_desc)

        # Add a consensus node that merges outputs
        def consensus_fn(store):
            outputs = {}
            for agent in self.agents:
                val = getattr(store, agent.name, None)
                if val:
                    outputs[agent.name] = val
            # Store all outputs for the merge
            if hasattr(store, "consensus_inputs"):
                store.consensus_inputs = outputs
            return "default"

        flow.add("consensus", FunctionUnit(consensus_fn))

        # Wire all workers to consensus (override fan-out termination)
        for agent in self.agents:
            flow.wire(agent.name, "consensus")

        return flow


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEFAULT LLM — Placeholder that routes through LiteLLM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _make_default_llm(model: str = None) -> LLMFunction:
    """
    Create a default LLM function.

    In production, this would call LiteLLM, OpenAI, Anthropic, etc.
    Override with your own llm_fn for full control.
    """
    _model = model or "gpt-4o-mini"

    def _call(system: str, user: str, tools: list = None, **kwargs) -> str:
        try:
            import litellm

            response = litellm.completion(
                model=_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response.choices[0].message.content
        except ImportError:
            # Fallback: return a placeholder for testing
            return f"[{_model}] Would respond to: {user[:100]}..."

    return _call


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONVENIENCE — Quick agent creation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def agent(role: str, goal: str, **kwargs) -> Agent:
    """Shorthand for Agent creation."""
    return Agent(role, goal, **kwargs)


def team(agents: list[Agent], **kwargs) -> Team:
    """Shorthand for Team creation."""
    return Team(agents, **kwargs)
