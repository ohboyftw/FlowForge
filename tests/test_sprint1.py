"""
Sprint 1 Tests — Real Parallelism, Typed Reducers, Eval-Optimizer Loop
"""

import time

from pydantic import Field

from flowforge import (
    Agent,
    Flow,
    FunctionUnit,
    ReducerRegistry,
    StoreBase,
    Team,
    Unit,
)

# ═══════════════════════════════════════════════════════════
# Test state models
# ═══════════════════════════════════════════════════════════


class ParallelState(StoreBase):
    __reducers__ = {"findings": "extend", "scores": "extend"}

    task: str = ""
    findings: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)


class LoopState(StoreBase):
    draft: str = ""
    feedback: str = ""
    quality: float = 0.0
    iterations: int = 0


# ═══════════════════════════════════════════════════════════
# Parallel execution tests
# ═══════════════════════════════════════════════════════════


class SlowUnit(Unit):
    """A unit that sleeps for a fixed duration then appends to findings."""

    def __init__(self, name: str, sleep_time: float = 0.1):
        self._name = name
        self._sleep_time = sleep_time

    def prep(self, store):
        return self._name

    def exec(self, name):
        time.sleep(self._sleep_time)
        return f"result_from_{name}"

    def post(self, store, result):
        store.findings = store.findings + [result]
        return "default"


def test_parallel_concurrent_execution():
    """3 units with 0.1s sleep should finish in ~0.1s, not ~0.3s."""
    flow = Flow()
    flow.add("dispatch", FunctionUnit(lambda s: "default"))
    flow.add("a", SlowUnit("a", 0.1))
    flow.add("b", SlowUnit("b", 0.1))
    flow.add("c", SlowUnit("c", 0.1))
    flow.wire("dispatch", ["a", "b", "c"])
    flow.entry("dispatch")

    state = ParallelState(task="parallel test")
    start = time.monotonic()
    flow.run(state)
    elapsed = time.monotonic() - start

    # Should be ~0.1s not ~0.3s (allow some overhead)
    assert elapsed < 0.25, f"Expected <0.25s, got {elapsed:.2f}s — not truly parallel"
    assert len(state.findings) == 3


def test_reducer_extend_on_parallel():
    """Parallel units append to list field via 'extend' reducer."""
    flow = Flow()
    flow.add("dispatch", FunctionUnit(lambda s: "default"))

    for name in ["x", "y", "z"]:

        class _AppendUnit(Unit):
            def __init__(self, n):
                self._n = n

            def post(self, store, _):
                store.findings = [f"from_{self._n}"]
                return "default"

        flow.add(name, _AppendUnit(name))

    flow.wire("dispatch", ["x", "y", "z"])
    flow.entry("dispatch")

    state = ParallelState(task="reducer test")
    flow.run(state)

    # All three should be present thanks to "extend" reducer
    assert len(state.findings) == 3
    assert all(f.startswith("from_") for f in state.findings)


def test_reducer_custom_fn():
    """Custom lambda reducer merges correctly."""

    class CustomState(StoreBase):
        __reducers__ = {"total": lambda old, new: (old or 0) + new}
        total: float = 0.0
        task: str = ""

    flow = Flow()
    flow.add("dispatch", FunctionUnit(lambda s: "default"))

    class AddUnit(Unit):
        def __init__(self, val):
            self._val = val

        def post(self, store, _):
            store.total = self._val
            return "default"

    flow.add("a", AddUnit(10.0))
    flow.add("b", AddUnit(20.0))
    flow.wire("dispatch", ["a", "b"])
    flow.entry("dispatch")

    state = CustomState(task="custom reducer")
    flow.run(state)

    # 10 + 20 = 30 (additive reducer)
    assert state.total == 30.0


def test_reducer_registry_from_store_class():
    """ReducerRegistry.from_store_class builds from __reducers__."""
    reg = ReducerRegistry.from_store_class(ParallelState)
    assert reg.reduce(None, "findings", ["a"], ["b"]) == ["a", "b"]
    # Non-reducer field defaults to replace
    assert reg.reduce(None, "task", "old", "new") == "new"


# ═══════════════════════════════════════════════════════════
# Team parallel with typed store
# ═══════════════════════════════════════════════════════════


def _mock_llm(system: str, user: str, tools: list = None, **kwargs) -> str:
    return f"MOCK: {user[:50]}"


class TeamParallelState(StoreBase):
    __reducers__ = {"analyst_a": "replace", "analyst_b": "replace"}

    task: str = ""
    analyst_a: str = ""
    analyst_b: str = ""


def test_typed_store_parallel_team():
    """Team parallel works with typed StoreBase + __reducers__."""
    a1 = Agent("Analyst_A", "Analyze from angle A", llm_fn=_mock_llm)
    a2 = Agent("Analyst_B", "Analyze from angle B", llm_fn=_mock_llm)
    t = Team(
        [a1, a2],
        strategy="parallel",
        store_class=TeamParallelState,
    )

    store = TeamParallelState(task="team parallel test")
    result = t.run("analyze this", store=store)

    assert result.analyst_a != ""
    assert result.analyst_b != ""


# ═══════════════════════════════════════════════════════════
# Loop tests
# ═══════════════════════════════════════════════════════════


class GeneratorUnit(Unit):
    def post(self, store, _):
        store.draft = f"draft_v{store.iterations + 1}"
        store.iterations += 1
        return "default"


class EvaluatorUnit(Unit):
    def post(self, store, _):
        # Improve quality each round
        store.quality += 0.3
        store.feedback = f"feedback_on_{store.draft}"
        return "default"


def test_loop_terminates_on_condition():
    """Loop stops when until=True."""
    flow = Flow()
    flow.add("generate", GeneratorUnit())
    flow.add("evaluate", EvaluatorUnit())
    flow.entry("generate")
    flow.loop("generate", "evaluate", until=lambda s: s.quality >= 0.5, max_rounds=10)

    state = LoopState()
    flow.run(state)

    assert state.quality >= 0.5
    assert state.iterations <= 3  # should stop at 2 rounds (0.3 + 0.3 = 0.6)


def test_loop_respects_max_rounds():
    """Loop stops at max_rounds even if until=False."""
    flow = Flow()
    flow.add("generate", GeneratorUnit())
    flow.add("evaluate", EvaluatorUnit())
    flow.entry("generate")
    flow.loop("generate", "evaluate", until=lambda s: False, max_rounds=3)

    state = LoopState()
    flow.run(state)

    assert state.iterations == 3


def test_loop_fluent_api():
    """flow.loop() returns self for chaining."""
    flow = Flow()
    flow.add("gen", GeneratorUnit())
    flow.add("eval", EvaluatorUnit())

    result = flow.loop("gen", "eval", until=lambda s: True, max_rounds=1)
    assert result is flow  # fluent API
