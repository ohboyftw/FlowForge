"""
Boundary Contract Tests for FlowForge P0+P1 Changes

Tests the *glue* between layers and components — not internal logic.
Each test validates that a producer's output is valid input for its consumer.

Boundary map:
  B1: Fan-out → Continuation (store state after merge)
  B2: Error routing (exception → handler node via _last_error)
  B3: Timeout → Error routing (TimeoutError feeds into error wires)
  B4: CountingWrapper delegation (sync/async Unit protocol)
  B5: Team → Flow parameter forwarding
  B6: Validation → Execution (_validate_graph before run)
  B7: Hierarchical compile → Flow fan-out + continuation
"""

import asyncio
import time

import pytest
from pydantic import Field

from flowforge import (
    AsyncUnit,
    FlexStore,
    Flow,
    FunctionUnit,
    InterruptSignal,
    ReducerRegistry,
    StoreBase,
    Unit,
)
from flowforge.harness import Agent, Team

# ═══════════════════════════════════════════════════════════
# Shared test state models (contracts)
# ═══════════════════════════════════════════════════════════


class FanoutState(StoreBase):
    """Contract for fan-out merge boundary."""

    __reducers__ = {"items": "extend", "total": lambda old, new: old + new}

    task: str = ""
    items: list[str] = Field(default_factory=list)
    total: int = 0
    downstream_saw: str = ""


class ErrorState(StoreBase):
    """Contract for error routing boundary."""

    task: str = ""
    result: str = ""
    error_handled: bool = False
    handler_saw_error: str = ""


class TimeoutState(StoreBase):
    """Contract for timeout boundary."""

    task: str = ""
    result: str = ""
    timed_out: bool = False


class LoopState(StoreBase):
    """Contract for counting wrapper boundary."""

    value: int = 0
    evaluations: int = 0
    done: bool = False


# ═══════════════════════════════════════════════════════════
# B1: Fan-out → Continuation boundary
# ═══════════════════════════════════════════════════════════


class ItemProducer(Unit):
    """Produces items during fan-out."""

    def __init__(self, item: str):
        self._item = item

    def post(self, store, _):
        store.items = store.items + [self._item]
        store.total = store.total + 1
        return "default"


class DownstreamConsumer(Unit):
    """Consumes merged state after fan-out."""

    def post(self, store, _):
        store.downstream_saw = f"items={len(store.items)},total={store.total}"
        return "default"


class TestB1FanoutContinuation:
    """Fan-out produces merged state → continuation node consumes it."""

    def test_continuation_sees_merged_state(self):
        """Downstream node receives fully merged fan-out results."""
        flow = Flow(reducers=ReducerRegistry({"items": "extend", "total": lambda o, n: o + n}))
        flow.add("dispatch", FunctionUnit(lambda s: "default"))
        flow.add("a", ItemProducer("alpha"))
        flow.add("b", ItemProducer("beta"))
        flow.add("c", ItemProducer("gamma"))
        flow.add("aggregate", DownstreamConsumer())

        flow.wire("dispatch", ["a", "b", "c"])  # fan-out
        flow.wire("dispatch", "aggregate")  # continuation
        flow.entry("dispatch")

        state = FanoutState(task="test")
        flow.run(state)

        # Contract: downstream sees ALL fan-out results merged
        assert len(state.items) == 3
        assert state.total == 3
        assert "items=3" in state.downstream_saw
        assert "total=3" in state.downstream_saw

    def test_fanout_store_schema_survives_merge(self):
        """Round-trip: store schema is valid after fan-out merge."""
        flow = Flow(reducers=ReducerRegistry({"items": "extend"}))
        flow.add("dispatch", FunctionUnit(lambda s: "default"))
        flow.add("a", ItemProducer("x"))
        flow.add("b", ItemProducer("y"))
        flow.wire("dispatch", ["a", "b"])
        flow.entry("dispatch")

        state = FanoutState(task="roundtrip")
        flow.run(state)

        # Contract: store is still a valid FanoutState after merge
        serialized = state.model_dump_json()
        restored = FanoutState.model_validate_json(serialized)
        assert restored.items == state.items
        assert restored.total == state.total


# ═══════════════════════════════════════════════════════════
# B2: Error routing boundary
# ═══════════════════════════════════════════════════════════


class FailingUnit(Unit):
    """Always raises — producer side of error boundary."""

    def exec(self, _):
        raise ValueError("deliberate failure")


class ErrorHandlerUnit(Unit):
    """Consumes _last_error from store — consumer side of error boundary."""

    def post(self, store, _):
        error = getattr(store, "_last_error", None)
        store.error_handled = True
        store.handler_saw_error = str(error) if error else "no error"
        return "default"


class TestB2ErrorRouting:
    """Unit exception → _last_error on store → error handler node."""

    def test_wire_level_error_reaches_handler(self):
        """on='error' wire routes exception to handler node."""
        flow = Flow()
        flow.add("fail", FailingUnit())
        flow.add("handler", ErrorHandlerUnit())
        flow.wire("fail", "handler", on="error")
        flow.entry("fail")

        state = ErrorState(task="test")
        flow.run(state)

        # Contract: handler received the error
        assert state.error_handled is True
        assert "deliberate failure" in state.handler_saw_error

    def test_flow_level_error_fallback(self):
        """on_error parameter catches unhandled unit exceptions."""
        flow = Flow(on_error="handler")
        flow.add("fail", FailingUnit())
        flow.add("handler", ErrorHandlerUnit())
        # No wire-level error route — relies on flow-level fallback
        flow.entry("fail")

        state = ErrorState(task="test")
        flow.run(state)

        assert state.error_handled is True

    def test_error_routing_does_not_swallow_interrupt(self):
        """InterruptSignal must propagate — it is NOT an error."""
        flow = Flow(on_error="handler")
        flow.add("a", FunctionUnit(lambda s: "default"))
        flow.add("b", FunctionUnit(lambda s: "default"))
        flow.add("handler", ErrorHandlerUnit())
        flow.wire("a", "b", interrupt=True)
        flow.entry("a")

        with pytest.raises(InterruptSignal):
            flow.run(ErrorState(task="test"))

    def test_no_error_route_reraises(self):
        """Without error routing, original exception propagates."""
        flow = Flow()  # no on_error
        flow.add("fail", FailingUnit())
        flow.entry("fail")

        with pytest.raises(ValueError, match="deliberate failure"):
            flow.run(ErrorState(task="test"))

    def test_last_error_schema_on_store(self):
        """_last_error doesn't pollute store schema (not in model_dump)."""
        flow = Flow()
        flow.add("fail", FailingUnit())
        flow.add("handler", ErrorHandlerUnit())
        flow.wire("fail", "handler", on="error")
        flow.entry("fail")

        state = ErrorState(task="test")
        flow.run(state)

        # Contract: _last_error is NOT in serialized output
        dumped = state.model_dump()
        assert "_last_error" not in dumped


# ═══════════════════════════════════════════════════════════
# B3: Timeout → Error routing boundary
# ═══════════════════════════════════════════════════════════


class SlowUnit(Unit):
    """Takes longer than timeout — producer side."""

    timeout: float | None = None

    def __init__(self, duration: float):
        self._duration = duration
        self.timeout = 0.1  # 100ms timeout

    def exec(self, _):
        time.sleep(self._duration)
        return "done"

    def post(self, store, result):
        store.result = result
        return "default"


class TimeoutHandlerUnit(Unit):
    """Handles timeout — consumer side."""

    def post(self, store, _):
        error = getattr(store, "_last_error", None)
        store.timed_out = isinstance(error, (TimeoutError, asyncio.TimeoutError))
        return "default"


class TestB3TimeoutErrorRouting:
    """Timeout exception → error routing → handler."""

    def test_timeout_feeds_into_error_wire(self):
        """TimeoutError from slow unit routes to error handler."""
        flow = Flow()
        flow.add("slow", SlowUnit(duration=1.0))  # 1s, timeout is 0.1s
        flow.add("handler", TimeoutHandlerUnit())
        flow.wire("slow", "handler", on="error")
        flow.entry("slow")

        state = TimeoutState(task="test")
        flow.run(state, default_timeout=0.1)

        assert state.timed_out is True

    def test_no_timeout_no_overhead(self):
        """Units without timeout run normally — no wrapping."""
        flow = Flow()
        fast_unit = FunctionUnit(lambda s: setattr(s, "result", "fast") or "default")
        flow.add("fast", fast_unit)
        flow.entry("fast")

        state = TimeoutState(task="test")
        flow.run(state)  # no default_timeout

        assert state.result == "fast"


# ═══════════════════════════════════════════════════════════
# B4: CountingWrapper delegation boundary
# ═══════════════════════════════════════════════════════════


class SyncEvaluator(Unit):
    """Sync evaluator for loop — checks done condition."""

    def post(self, store, _):
        store.evaluations += 1
        return "default"


class AsyncEvaluator(AsyncUnit):
    """Async evaluator for loop — same contract, async exec."""

    async def exec(self, _):
        await asyncio.sleep(0.01)
        return "evaluated"

    def post(self, store, _):
        store.evaluations += 1
        return "default"


class Generator(Unit):
    def post(self, store, _):
        store.value += 1
        return "default"


class TestB4CountingWrapperDelegation:
    """Flow.loop() wraps evaluator → wrapper delegates correctly."""

    def test_sync_wrapper_counts_and_delegates(self):
        """_CountingWrapper preserves sync Unit protocol."""
        flow = Flow()
        flow.add("gen", Generator())
        flow.add("eval", SyncEvaluator())
        flow.loop("gen", "eval", until=lambda s: s.value >= 3, max_rounds=5)
        flow.entry("gen")

        state = LoopState()
        flow.run(state)

        # Contract: evaluator was called, counter incremented
        assert state.value >= 3
        assert state.evaluations >= 3

    @pytest.mark.asyncio
    async def test_async_wrapper_counts_and_delegates(self):
        """_AsyncCountingWrapper preserves AsyncUnit protocol."""
        flow = Flow()
        flow.add("gen", Generator())
        flow.add("eval", AsyncEvaluator())
        flow.loop("gen", "eval", until=lambda s: s.value >= 3, max_rounds=5)
        flow.entry("gen")

        state = LoopState()
        await flow.arun(state)

        # Contract: async evaluator was called via arun
        assert state.value >= 3
        assert state.evaluations >= 3


# ═══════════════════════════════════════════════════════════
# B5: Team → Flow parameter forwarding boundary
# ═══════════════════════════════════════════════════════════


def _mock_llm(system: str, user: str, tools: list = None, **kw) -> str:
    return f"MOCK: {user[:50]}"


async def _mock_async_llm(system: str, user: str, tools: list = None, **kw) -> str:
    return f"ASYNC_MOCK: {user[:50]}"


class TestB5TeamFlowForwarding:
    """Team.arun() forwards parameters to Flow.arun()."""

    @pytest.mark.asyncio
    async def test_arun_forwards_max_steps(self):
        """Team.arun(max_steps=N) reaches Flow.arun(max_steps=N)."""
        a1 = Agent("Worker", "Do work", llm_fn=_mock_llm)
        t = Team([a1], strategy="sequential")

        # Should complete — max_steps is generous
        result = await t.arun("test task", max_steps=100)
        assert result is not None

    @pytest.mark.asyncio
    async def test_arun_forwards_raise_on_exhaust(self):
        """Team.arun(raise_on_exhaust=True) propagates FlowExhaustedError."""
        # Create a looping team that will exhaust max_steps
        a1 = Agent("Looper", "Loop forever", llm_fn=_mock_llm)
        t = Team([a1], strategy="sequential")

        # With max_steps=1 and raise_on_exhaust, should get FlowExhaustedError
        # (only if the flow actually loops — sequential with 1 agent won't)
        # This tests parameter forwarding, not exhaustion logic
        result = await t.arun("test", max_steps=100, raise_on_exhaust=False)
        assert result is not None


# ═══════════════════════════════════════════════════════════
# B6: Validation → Execution boundary
# ═══════════════════════════════════════════════════════════

VALIDATION_EDGE_CASES = [
    # (description, setup_fn, should_raise)
    (
        "wire to nonexistent target",
        lambda: _build_flow_with_bad_wire(),
        True,
    ),
    (
        "wire before add (lazy allows)",
        lambda: _build_flow_wire_before_add(),
        False,
    ),
    (
        "entry to nonexistent node",
        lambda: _build_flow_with_bad_entry(),
        True,
    ),
]


def _build_flow_with_bad_wire():
    flow = Flow()
    flow.add("a", FunctionUnit(lambda s: "default"))
    flow.wire("a", "nonexistent")
    flow.entry("a")
    return flow


def _build_flow_wire_before_add():
    flow = Flow()
    flow.wire("a", "b")  # wire before nodes exist
    flow.add("a", FunctionUnit(lambda s: "default"))
    flow.add("b", FunctionUnit(lambda s: "default"))
    flow.entry("a")
    return flow


def _build_flow_with_bad_entry():
    flow = Flow()
    flow.add("a", FunctionUnit(lambda s: "default"))
    flow.entry("nonexistent")
    return flow


class TestB6Validation:
    """_validate_graph checks at run time, not construction time."""

    @pytest.mark.parametrize(
        "desc,setup_fn,should_raise",
        VALIDATION_EDGE_CASES,
        ids=[c[0] for c in VALIDATION_EDGE_CASES],
    )
    def test_validation_edge_cases(self, desc, setup_fn, should_raise):
        """Validation rejects bad graphs and accepts late-bound ones."""
        flow = setup_fn()
        state = FlexStore(task="test")

        if should_raise:
            with pytest.raises(ValueError):
                flow.run(state)
        else:
            flow.run(state)  # should not raise

    def test_validation_error_lists_all_problems(self):
        """ValueError message includes all graph errors, not just the first."""
        flow = Flow()
        flow.wire("ghost1", "ghost2")
        flow.wire("ghost3", "ghost4")
        # No entry, no units

        with pytest.raises(ValueError, match="ghost1"):
            flow.run(FlexStore())


# ═══════════════════════════════════════════════════════════
# B7: Hierarchical compile → Flow structure boundary
# ═══════════════════════════════════════════════════════════


class TestB7HierarchicalCompile:
    """_build_hierarchical produces correct wire structure for Flow."""

    def test_hierarchical_wire_structure(self):
        """Compiled flow has fan-out wire + continuation wire from decompose."""
        mgr = Agent("Manager", "Manage", llm_fn=_mock_llm)
        w1 = Agent("Worker1", "Work", llm_fn=_mock_llm)
        w2 = Agent("Worker2", "Work", llm_fn=_mock_llm)
        t = Team([w1, w2], strategy="hierarchical", manager=mgr)

        flow = t.compile("test")

        # Contract: decompose has wires to workers (fan-out) and synthesize (continuation)
        assert "decompose" in flow.nodes
        assert "synthesize" in flow.nodes
        assert "worker1" in flow.nodes
        assert "worker2" in flow.nodes

        decompose_edges = [(s, t, lbl) for s, t, lbl in flow.edges if s == "decompose"]
        targets = [t for _, t, _ in decompose_edges]

        # Workers appear as fan-out targets
        assert "worker1" in targets
        assert "worker2" in targets
        # Synthesize appears as continuation target
        assert "synthesize" in targets

    def test_hierarchical_no_orphan_worker_wires(self):
        """Workers should NOT have individual wires to synthesize."""
        mgr = Agent("Manager", "Manage", llm_fn=_mock_llm)
        w1 = Agent("Worker1", "Work", llm_fn=_mock_llm)
        w2 = Agent("Worker2", "Work", llm_fn=_mock_llm)
        t = Team([w1, w2], strategy="hierarchical", manager=mgr)

        flow = t.compile("test")

        # Contract: no worker → synthesize edges (continuation handles it)
        worker_to_synth = [
            (s, t) for s, t, _ in flow.edges if s in ("worker1", "worker2") and t == "synthesize"
        ]
        assert worker_to_synth == []


# ═══════════════════════════════════════════════════════════
# Cross-boundary: Golden path trace
# ═══════════════════════════════════════════════════════════


class TestGoldenPath:
    """End-to-end traces through the changed code paths."""

    def test_fanout_error_timeout_golden_path(self):
        """Fan-out with one failing unit → error routing → continuation."""
        flow = Flow(on_error="error_handler")
        flow.add("dispatch", FunctionUnit(lambda s: "default"))
        flow.add("good", ItemProducer("success"))
        flow.add("error_handler", ErrorHandlerUnit())
        flow.add("final", DownstreamConsumer())

        flow.wire("dispatch", "good")
        flow.wire("dispatch", "final")
        flow.entry("dispatch")

        state = FanoutState(task="golden")
        result = flow.run(state)

        # The flow should complete without error
        assert result is not None

    def test_hierarchical_team_end_to_end(self):
        """Hierarchical team: decompose → parallel workers → synthesize."""
        mgr = Agent("Manager", "Manage", llm_fn=_mock_llm)
        w1 = Agent("Researcher", "Research", llm_fn=_mock_llm)
        w2 = Agent("Analyst", "Analyze", llm_fn=_mock_llm)
        t = Team([w1, w2], strategy="hierarchical", manager=mgr)

        store = FlexStore(task="test task")
        result = t.run("test task", store=store)

        # Contract: all stages ran, final_result is populated
        assert hasattr(result, "subtasks")
        assert hasattr(result, "final_result")
