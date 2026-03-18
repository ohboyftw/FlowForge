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
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar

from .core import AsyncUnit, FlexStore, Flow, FunctionUnit, ReducerRegistry, StoreBase, Unit
from .identity import Persona, Task

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM UNIT — A Unit that calls an LLM with Persona context
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Type for LLM calling functions
# Signature: (system: str, user: str, tools: list) -> str
LLMFunction = Callable[..., str]

# Type for async LLM calling functions
AsyncLLMFunction = Callable[..., Awaitable[str]]


def _llm_prep(unit, store: StoreBase) -> dict:
    """Shared prep logic for LLMUnit and AsyncLLMUnit."""
    context = {}
    for field_name in unit.task.context_from:
        if hasattr(store, field_name):
            context[field_name] = getattr(store, field_name)
    return {
        "system": unit.persona.to_prompt(),
        "user": unit.task.to_prompt(context),
        "tools": unit.tools,
    }


def _llm_post(unit, store: StoreBase, exec_result: dict) -> str:
    """Shared post logic for LLMUnit and AsyncLLMUnit."""
    output = exec_result["output"]
    if hasattr(store, unit.output_field) or isinstance(store, FlexStore):
        setattr(store, unit.output_field, output)
    return "default"


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
        return _llm_prep(self, store)

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
        return _llm_post(self, store, exec_result)


class AsyncLLMUnit(AsyncUnit):
    """
    Async variant of LLMUnit. Same prep/post (sync), async exec.

    Supports optional streaming via on_token callback.
    """

    def __init__(
        self,
        persona: Persona,
        task: Task,
        llm_fn: AsyncLLMFunction,
        tools: list = None,
        output_field: str = None,
    ):
        self.persona = persona
        self.task = task
        self.llm_fn = llm_fn
        self.tools = tools or []
        self.output_field = output_field or task.output_field or task.description
        self._on_token: Callable | None = None

    def prep(self, store: StoreBase) -> dict:
        return _llm_prep(self, store)

    async def exec(self, prep_result: dict) -> dict:
        start = time.monotonic()
        result = await self.llm_fn(
            system=prep_result["system"],
            user=prep_result["user"],
            tools=prep_result.get("tools", []),
            on_token=self._on_token,
        )
        elapsed = (time.monotonic() - start) * 1000
        return {"output": result, "latency_ms": elapsed}

    def post(self, store: StoreBase, exec_result: dict) -> str:
        return _llm_post(self, store, exec_result)


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

    Built-in strategies:
        "sequential"    — agents run one after another (pipeline)
        "parallel"      — all agents run, results merge via reducers
        "hierarchical"  — manager decomposes and delegates to workers
        "consensus"     — all agents run, then a vote/merge step

    Custom strategies via Team.register_strategy():
        Team.register_strategy("my_strategy", my_builder_fn)
        # builder_fn signature: (team: Team, task_desc: str) -> Flow

    Escape hatch: team.graph gives you the compiled Flow
    for manual Wire additions.
    """

    _strategy_registry: ClassVar[dict[str, Callable]] = {}

    def __init__(
        self,
        agents: list[Agent],
        *,
        strategy: str = "sequential",
        manager: Agent = None,
        store_class: type[StoreBase] = None,
        reducers: dict[str, str] = None,
    ):
        all_strategies = list(self._strategy_registry.keys()) + [
            "sequential",
            "parallel",
            "hierarchical",
            "consensus",
        ]
        if strategy not in all_strategies:
            raise ValueError(f"Unknown strategy '{strategy}'. Options: {all_strategies}")
        self.agents = agents
        self.strategy = strategy
        self.manager = manager
        self.store_class = store_class
        self._reducers = ReducerRegistry(reducers) if reducers else ReducerRegistry()
        self._flow: Flow | None = None

    @classmethod
    def register_strategy(cls, name: str, builder_fn: Callable) -> None:
        """Register a custom strategy. builder_fn(team, task_desc) -> Flow"""
        cls._strategy_registry[name] = builder_fn

    def compile(self, task_desc: str = "") -> Flow:
        """
        Compile agents + strategy into a Flow graph.

        Returns the Flow for inspection or modification before execution.
        """
        # Check custom registry first
        if self.strategy in self._strategy_registry:
            self._flow = self._strategy_registry[self.strategy](self, task_desc)
            return self._flow
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

    async def arun(
        self,
        task_desc: str,
        *,
        store: StoreBase = None,
        max_steps: int = 100,
        raise_on_exhaust: bool = False,
    ) -> StoreBase:
        """Async execution. Mirrors run() but uses flow.arun()."""
        flow = self.compile(task_desc)
        if store is None:
            if self.store_class:
                store = self.store_class(task=task_desc)
            else:
                store = FlexStore(task=task_desc)
        return await flow.arun(
            store, max_steps=max_steps, raise_on_exhaust=raise_on_exhaust
        )

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
        # Build reducer registry: store class __reducers__ + explicit reducers
        if self.store_class:
            store_reg = ReducerRegistry.from_store_class(self.store_class)
            flow = Flow(reducers=store_reg.merge(self._reducers))
        else:
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
        """Manager decomposes, workers execute in parallel, manager synthesizes."""
        flow = Flow(reducers=self._reducers)

        mgr = self.manager or Agent(
            "Manager",
            "Decompose and delegate tasks",
            llm_fn=self.agents[0].llm_fn,
        )

        # Manager decomposition
        decompose_task = Task(
            description=f"Decompose this task into subtasks for your team: {task_desc}",
            expected_output="List of subtasks with assignments",
            output_field="subtasks",
        )
        flow.add("decompose", mgr.as_unit(decompose_task))

        # Worker nodes
        worker_names = []
        for agent in self.agents:
            task = Task(
                description=task_desc,
                context_from=["subtasks"],
                output_field=agent.name,
            )
            flow.add(agent.name, agent.as_unit(task))
            worker_names.append(agent.name)

        # Fan-out: decompose → all workers in parallel
        flow.wire("decompose", worker_names)

        # Continuation: after fan-out merges → synthesize
        synthesize_task = Task(
            description="Synthesize team outputs into final result",
            context_from=[a.name for a in self.agents],
            output_field="final_result",
        )
        flow.add("synthesize", mgr.as_unit(synthesize_task))
        flow.wire("decompose", "synthesize")

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
    Create a default LLM function using LiteLLM.

    Bring-your-own llm_fn for full control:

        # OpenAI direct:
        from openai import OpenAI
        client = OpenAI()
        def my_llm(system, user, tools=None, **kw):
            r = client.chat.completions.create(
                model="gpt-4o", messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ])
            return r.choices[0].message.content

        # Anthropic direct:
        from anthropic import Anthropic
        client = Anthropic()
        def my_llm(system, user, tools=None, **kw):
            r = client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=4096,
                system=system, messages=[{"role": "user", "content": user}])
            return r.content[0].text
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
            return f"[{_model}] Would respond to: {user[:100]}..."
        except Exception as exc:
            return f"[{_model}] LLM error: {exc}"

    return _call


def _make_default_async_llm(model: str = None) -> AsyncLLMFunction:
    """
    Create a default async LLM function using LiteLLM acompletion.

    Supports streaming via on_token callback.
    """
    _model = model or "gpt-4o-mini"

    async def _call(
        system: str,
        user: str,
        tools: list = None,
        on_token: Callable | None = None,
        **kwargs,
    ) -> str:
        try:
            import litellm

            if on_token:
                # Streaming mode
                response = await litellm.acompletion(
                    model=_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    stream=True,
                )
                chunks = []
                async for chunk in response:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        chunks.append(delta)
                        on_token(delta)
                return "".join(chunks)
            else:
                response = await litellm.acompletion(
                    model=_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                return response.choices[0].message.content
        except ImportError:
            return f"[{_model}] Would respond to: {user[:100]}..."
        except Exception as exc:
            return f"[{_model}] LLM error: {exc}"

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
