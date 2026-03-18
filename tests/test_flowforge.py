"""
FlowForge Test Suite
28 tests covering all layers: Store, Identity, Unit, Flow, Harness
"""

import json
import logging
import time as time_mod

from pydantic import Field, ValidationError

from flowforge import (
    Agent,
    FlexStore,
    Flow,
    FlowExhaustedError,
    FunctionUnit,
    InterruptSignal,
    LLMUnit,
    Persona,
    Personas,
    ReducerRegistry,
    StoreBase,
    Task,
    TaskResult,
    Team,
    Unit,
)

# ═══════════════════════════════════════════════════════════
# Test state models
# ═══════════════════════════════════════════════════════════


class PipelineState(StoreBase):
    task: str = ""
    findings: list[str] = Field(default_factory=list)
    code: str = ""
    confidence: float = 0.0
    approved: bool = False
    attempts: int = 0


class CounterState(StoreBase):
    count: int = 0
    log: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════
# Store (Pydantic) tests
# ═══════════════════════════════════════════════════════════


class TestStore:
    def test_creation(self):
        state = PipelineState(task="test task")
        assert state.task == "test task"
        assert state.findings == []
        assert state.confidence == 0.0

    def test_validation_on_assignment(self):
        state = PipelineState()
        state.confidence = 0.85
        assert state.confidence == 0.85

        try:
            state.confidence = "not a float"
            raise AssertionError("Should have raised ValidationError")
        except ValidationError:
            pass

    def test_extra_fields_forbidden(self):
        try:
            PipelineState(taks="typo")
            raise AssertionError("Should have raised ValidationError")
        except ValidationError:
            pass

    def test_checkpoint_rollback(self):
        state = PipelineState(task="research")
        state.findings = ["paper1", "paper2"]
        state.confidence = 0.7
        state.checkpoint("after_research")

        state.findings = []
        state.confidence = 0.0
        assert state.findings == []

        state.rollback("after_research")
        assert len(state.findings) == 2
        assert state.confidence == 0.7

    def test_json_serialization(self):
        state = PipelineState(task="test", findings=["a", "b"], confidence=0.9)
        json_str = state.to_json()
        data = json.loads(json_str)
        assert data["task"] == "test"

        restored = PipelineState.from_json(json_str)
        assert restored.task == state.task
        assert restored.findings == state.findings

    def test_diff(self):
        s1 = PipelineState(task="test", confidence=0.5)
        s2 = s1.model_copy(deep=True)
        s2.confidence = 0.9
        s2.findings = ["new"]

        diff = s2.diff(s1)
        assert "confidence" in diff
        assert "findings" in diff

    def test_schema_introspection(self):
        fields = PipelineState.describe_fields()
        assert "task" in fields
        assert "confidence" in fields

    def test_frozen_copy(self):
        state = PipelineState(task="test")
        frozen = state.frozen_copy()
        frozen.task = "modified"
        assert state.task == "test"

    def test_flex_store(self):
        store = FlexStore()
        store.anything = "works"
        store.number = 42
        assert store.anything == "works"


# ═══════════════════════════════════════════════════════════
# Identity tests
# ═══════════════════════════════════════════════════════════


class TestIdentity:
    def test_persona_to_prompt(self):
        p = Persona(
            role="Researcher", goal="Find data", backstory="PhD in ML", constraints=["Cite sources"]
        )
        prompt = p.to_prompt()
        assert "Researcher" in prompt
        assert "Find data" in prompt
        assert "Cite sources" in prompt

    def test_persona_serialization(self):
        p = Persona(role="Coder", goal="Write code")
        json_str = p.model_dump_json()
        restored = Persona.model_validate_json(json_str)
        assert restored.role == "Coder"

    def test_persona_with_overrides(self):
        p = Personas.researcher()
        p2 = p.with_overrides(goal="Find specific data on X")
        assert p2.goal == "Find specific data on X"
        assert p2.role == p.role

    def test_task_to_prompt(self):
        t = Task(
            description="Research LLM agents",
            expected_output="5 papers",
            context_from=["query", "previous"],
        )
        prompt = t.to_prompt(
            context={
                "query": "agentic frameworks",
                "previous": ["PocketFlow"],
                "irrelevant": "should not appear",
            }
        )
        assert "agentic frameworks" in prompt
        assert "irrelevant" not in prompt

    def test_task_result(self):
        r = TaskResult(task_description="test", output="result", confidence=0.9)
        assert r.succeeded
        r_fail = TaskResult(task_description="test", output=None, error="timeout")
        assert not r_fail.succeeded


# ═══════════════════════════════════════════════════════════
# Unit + Flow tests
# ═══════════════════════════════════════════════════════════


class IncrementUnit(Unit):
    def __init__(self, name: str, amount: int = 1):
        self._name = name
        self._amount = amount

    def prep(self, store):
        return store.count

    def exec(self, current_count):
        return current_count + self._amount

    def post(self, store, new_count):
        store.count = new_count
        store.log = store.log + [f"{self._name}: {new_count}"]
        return "default"


class ConditionalUnit(Unit):
    def post(self, store, _):
        return "done" if store.count >= 3 else "continue"


class TestUnitAndFlow:
    def test_unit_lifecycle(self):
        state = CounterState()
        unit = IncrementUnit("inc", 5)
        action = unit.run(state)
        assert state.count == 5
        assert action == "default"

    def test_function_unit(self):
        state = CounterState()
        unit = FunctionUnit(lambda s: setattr(s, "count", 42) or "done")
        action = unit.run(state)
        assert state.count == 42
        assert action == "done"

    def test_linear_flow(self):
        flow = Flow()
        flow.add("a", IncrementUnit("a", 1))
        flow.add("b", IncrementUnit("b", 10))
        flow.add("c", IncrementUnit("c", 100))
        flow.wire("a", "b").wire("b", "c")
        flow.entry("a")

        result = flow.run(CounterState())
        assert result.count == 111
        assert len(result.log) == 3

    def test_conditional_flow(self):
        flow = Flow()
        flow.add("check", ConditionalUnit())
        flow.add("inc", IncrementUnit("inc", 1))
        flow.add("done", FunctionUnit(lambda s: setattr(s, "log", s.log + ["done"]) or "default"))

        flow.wire("check", "inc", on="continue")
        flow.wire("check", "done", on="done")
        flow.wire("inc", "check")
        flow.entry("check")

        result = flow.run(CounterState(count=0))
        assert result.count == 3
        assert "done" in result.log[-1]

    def test_flow_interrupt(self):
        flow = Flow()
        flow.add("a", IncrementUnit("a", 1))
        flow.add("b", IncrementUnit("b", 10))
        flow.wire("a", "b", interrupt=True)
        flow.entry("a")

        try:
            flow.run(CounterState())
            raise AssertionError("Should have raised InterruptSignal")
        except InterruptSignal as sig:
            assert sig.from_node == "a"
            assert sig.store.count == 1

    def test_flow_trace(self):
        flow = Flow()
        flow.add("a", IncrementUnit("a", 1))
        flow.add("b", IncrementUnit("b", 10))
        flow.wire("a", "b")
        flow.entry("a")

        flow.run(CounterState())
        assert len(flow.trace) == 2

    def test_flow_describe(self):
        flow = Flow()
        flow.add("a", IncrementUnit("a", 1))
        flow.add("b", IncrementUnit("b", 10))
        flow.wire("a", "b")
        flow.entry("a")

        desc = flow.describe()
        assert "a" in desc and "b" in desc


# ═══════════════════════════════════════════════════════════
# Reducer tests
# ═══════════════════════════════════════════════════════════


class TestReducers:
    def test_builtin_reducers(self):
        reg = ReducerRegistry({"items": "extend", "meta": "merge"})
        state = CounterState()

        assert reg.reduce(state, "items", [1], [2, 3]) == [1, 2, 3]
        assert reg.reduce(state, "meta", {"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
        assert reg.reduce(state, "other", "old", "new") == "new"


# ═══════════════════════════════════════════════════════════
# Harness tests
# ═══════════════════════════════════════════════════════════


def _mock_llm(system: str, user: str, tools: list = None) -> str:
    return f"MOCK_RESPONSE: {user[:80]}"


class TestHarness:
    def test_agent_run(self):
        a = Agent("Tester", "Test things", llm_fn=_mock_llm)
        result = a.run("Do a test")
        assert "MOCK_RESPONSE" in result

    def test_agent_as_unit(self):
        a = Agent("Tester", "Test things", llm_fn=_mock_llm)
        task = Task(description="custom task", output_field="output")
        unit = a.as_unit(task)
        assert isinstance(unit, LLMUnit)

    def test_team_sequential(self):
        a1 = Agent("Step1", "Do step 1", llm_fn=_mock_llm)
        a2 = Agent("Step2", "Do step 2", llm_fn=_mock_llm)
        t = Team([a1, a2], strategy="sequential")

        flow = t.compile("test task")
        assert "step1" in flow.nodes
        assert "step2" in flow.nodes

    def test_team_hierarchical(self):
        mgr = Agent("Manager", "Manage team", llm_fn=_mock_llm)
        w1 = Agent("Worker1", "Do work", llm_fn=_mock_llm)
        w2 = Agent("Worker2", "Do work", llm_fn=_mock_llm)
        t = Team([w1, w2], strategy="hierarchical", manager=mgr)

        flow = t.compile("build something")
        assert "decompose" in flow.nodes
        assert "synthesize" in flow.nodes

    def test_team_describe(self):
        a1 = Agent("Alpha", "Do alpha", llm_fn=_mock_llm)
        a2 = Agent("Beta", "Do beta", llm_fn=_mock_llm)
        t = Team([a1, a2], strategy="sequential")
        t.compile("test")
        assert "alpha" in t.describe()

    def test_team_graph_escape_hatch(self):
        a1 = Agent("Alpha", "Do alpha", llm_fn=_mock_llm)
        a2 = Agent("Beta", "Do beta", llm_fn=_mock_llm)
        t = Team([a1, a2], strategy="sequential")

        g = t.graph
        g.wire("beta", "alpha", on="retry", when=lambda s: True)
        edges = g.edges
        assert any(e[0] == "beta" and e[1] == "alpha" for e in edges)


# ═══════════════════════════════════════════════════════════
# Wire validation tests
# ═══════════════════════════════════════════════════════════


class TestWireValidation:
    def test_wire_validation_catches_typo(self):
        """Wire to nonexistent node raises ValueError at run time."""
        flow = Flow()
        flow.add("a", FunctionUnit(lambda s: "default"))
        flow.wire("a", "nonexistent")
        flow.entry("a")

        import pytest

        with pytest.raises(ValueError, match="nonexistent"):
            flow.run(CounterState())

    def test_wire_validation_allows_late_add(self):
        """Wiring before adding nodes is OK — validated at run time."""
        flow = Flow()
        flow.wire("a", "b")  # wire before add
        flow.add("a", IncrementUnit("a", 1))
        flow.add("b", IncrementUnit("b", 10))
        flow.entry("a")

        result = flow.run(CounterState())
        assert result.count == 11

    def test_wire_validation_bad_source(self):
        """Wire from nonexistent source raises ValueError."""
        flow = Flow()
        flow.add("b", FunctionUnit(lambda s: "default"))
        flow.wire("ghost", "b")
        flow.entry("b")

        import pytest

        with pytest.raises(ValueError, match="ghost"):
            flow.run(CounterState())

    def test_entry_allows_lazy_set(self):
        """entry() no longer validates eagerly — accepts unknown names."""
        flow = Flow()
        flow.entry("future_node")  # should NOT raise
        flow.add("future_node", FunctionUnit(lambda s: "default"))
        flow.run(CounterState())  # validates at run time — should pass


# ═══════════════════════════════════════════════════════════
# Max steps tests
# ═══════════════════════════════════════════════════════════


class TestMaxSteps:
    def test_max_steps_warning(self, caplog):
        """Flow logs warning when max_steps is hit."""
        flow = Flow()
        flow.add("loop", FunctionUnit(lambda s: "default"))
        flow.wire("loop", "loop")
        flow.entry("loop")

        with caplog.at_level(logging.WARNING, logger="flowforge.core"):
            flow.run(CounterState(), max_steps=5)

        assert "max_steps=5" in caplog.text

    def test_max_steps_raises_when_opted_in(self):
        """FlowExhaustedError raised when raise_on_exhaust=True."""
        flow = Flow()
        flow.add("loop", FunctionUnit(lambda s: "default"))
        flow.wire("loop", "loop")
        flow.entry("loop")

        import pytest

        with pytest.raises(FlowExhaustedError) as exc_info:
            flow.run(CounterState(), max_steps=3, raise_on_exhaust=True)
        assert exc_info.value.steps == 3

    def test_max_steps_no_warning_when_not_exhausted(self, caplog):
        """No warning when flow completes before max_steps."""
        flow = Flow()
        flow.add("a", IncrementUnit("a", 1))
        flow.entry("a")

        with caplog.at_level(logging.WARNING, logger="flowforge.core"):
            flow.run(CounterState())

        assert "max_steps" not in caplog.text


# ═══════════════════════════════════════════════════════════
# Trace timing tests
# ═══════════════════════════════════════════════════════════


class TestTraceTiming:
    def test_trace_has_duration(self):
        """Each trace entry includes duration_ms."""
        flow = Flow()
        flow.add("a", IncrementUnit("a", 1))
        flow.add("b", IncrementUnit("b", 10))
        flow.wire("a", "b")
        flow.entry("a")

        flow.run(CounterState())

        for entry in flow.trace:
            assert "duration_ms" in entry
            assert isinstance(entry["duration_ms"], float)
            assert entry["duration_ms"] >= 0


# ═══════════════════════════════════════════════════════════
# Fan-out continuation tests (DESIGN-006)
# ═══════════════════════════════════════════════════════════


class TestFanoutContinuation:
    def test_fanout_then_continue(self):
        """After fan-out completes, next wire from same source runs."""
        flow = Flow()
        flow.add("dispatch", FunctionUnit(lambda s: "default"))
        flow.add("w1", IncrementUnit("w1", 1))
        flow.add("w2", IncrementUnit("w2", 10))
        flow.add("aggregate", IncrementUnit("aggregate", 100))

        flow.wire("dispatch", ["w1", "w2"])  # fan-out
        flow.wire("dispatch", "aggregate")  # continuation
        flow.entry("dispatch")

        result = flow.run(CounterState())
        # w1=1, w2=10 merged, then aggregate=+100
        assert result.count >= 100  # aggregate ran

    def test_multiple_fanouts_sequential_barrier(self):
        """Multiple fan-out wires on same node run sequentially."""
        flow = Flow()
        flow.add("dispatch", FunctionUnit(lambda s: "default"))
        flow.add("a", IncrementUnit("a", 1))
        flow.add("b", IncrementUnit("b", 10))
        flow.add("final", IncrementUnit("final", 100))

        flow.wire("dispatch", ["a"])  # first fan-out
        flow.wire("dispatch", ["b"])  # second fan-out
        flow.wire("dispatch", "final")  # continuation
        flow.entry("dispatch")

        result = flow.run(CounterState())
        assert result.count >= 100  # final ran after both fan-outs


# ═══════════════════════════════════════════════════════════
# Error routing tests (FEAT-010)
# ═══════════════════════════════════════════════════════════


class FailingExecUnit(Unit):
    def exec(self, _):
        raise ValueError("boom")


class ErrorCatcherUnit(Unit):
    def post(self, store, _):
        err = getattr(store, "_last_error", None)
        store.log = store.log + [f"caught:{err}"]
        return "default"


class TestErrorRouting:
    def test_wire_level_error_route(self):
        """on='error' wire catches unit exception."""
        flow = Flow()
        flow.add("fail", FailingExecUnit())
        flow.add("handler", ErrorCatcherUnit())
        flow.wire("fail", "handler", on="error")
        flow.entry("fail")

        state = CounterState()
        flow.run(state)
        assert any("caught:" in entry for entry in state.log)

    def test_flow_level_error_fallback(self):
        """Flow-level on_error catches when no wire-level route."""
        flow = Flow(on_error="handler")
        flow.add("fail", FailingExecUnit())
        flow.add("handler", ErrorCatcherUnit())
        flow.entry("fail")

        state = CounterState()
        flow.run(state)
        assert any("caught:" in entry for entry in state.log)

    def test_no_route_reraises(self):
        """Without error routing, exception propagates."""
        flow = Flow()
        flow.add("fail", FailingExecUnit())
        flow.entry("fail")

        import pytest

        with pytest.raises(ValueError, match="boom"):
            flow.run(CounterState())

    def test_interrupt_not_caught_by_error_routing(self):
        """InterruptSignal bypasses error routing."""
        flow = Flow(on_error="handler")
        flow.add("a", FunctionUnit(lambda s: "default"))
        flow.add("b", FunctionUnit(lambda s: "default"))
        flow.add("handler", ErrorCatcherUnit())
        flow.wire("a", "b", interrupt=True)
        flow.entry("a")

        import pytest

        with pytest.raises(InterruptSignal):
            flow.run(CounterState())

    def test_error_trace_entry(self):
        """Error produces a trace entry with error info."""
        flow = Flow()
        flow.add("fail", FailingExecUnit())
        flow.add("handler", ErrorCatcherUnit())
        flow.wire("fail", "handler", on="error")
        flow.entry("fail")

        flow.run(CounterState())
        error_entries = [e for e in flow.trace if e.get("action") == "error"]
        assert len(error_entries) == 1
        assert "boom" in error_entries[0]["error"]


# ═══════════════════════════════════════════════════════════
# Timeout tests (FEAT-005)
# ═══════════════════════════════════════════════════════════


class SlowExecUnit(Unit):
    timeout = None  # will be set per instance

    def __init__(self, duration: float, timeout: float = None):
        self._duration = duration
        self.timeout = timeout

    def exec(self, _):
        time_mod.sleep(self._duration)
        return "done"

    def post(self, store, result):
        store.log = store.log + ["completed"]
        return "default"


class TestTimeout:
    def test_unit_timeout_raises(self):
        """Unit with timeout raises TimeoutError when exceeded."""
        flow = Flow()
        flow.add("slow", SlowExecUnit(duration=2.0, timeout=0.1))
        flow.add("handler", ErrorCatcherUnit())
        flow.wire("slow", "handler", on="error")
        flow.entry("slow")

        state = CounterState()
        flow.run(state)
        assert any("caught:" in entry for entry in state.log)

    def test_default_timeout_applies(self):
        """Flow-level default_timeout applies to units without explicit timeout."""
        flow = Flow()
        flow.add("slow", SlowExecUnit(duration=2.0))  # no unit timeout
        flow.add("handler", ErrorCatcherUnit())
        flow.wire("slow", "handler", on="error")
        flow.entry("slow")

        state = CounterState()
        flow.run(state, default_timeout=0.1)
        assert any("caught:" in entry for entry in state.log)

    def test_no_timeout_no_overhead(self):
        """Units without timeout run normally."""
        flow = Flow()
        flow.add("fast", IncrementUnit("fast", 1))
        flow.entry("fast")

        result = flow.run(CounterState())
        assert result.count == 1
