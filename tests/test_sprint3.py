"""
Sprint 3 Tests — Strategy Registry, Mermaid Export, Unit Retry
"""

import pytest

from flowforge import Agent, Flow, FunctionUnit, StoreBase, Team, Unit

# ═══════════════════════════════════════════════════════════
# Test state models
# ═══════════════════════════════════════════════════════════


class RetryState(StoreBase):
    task: str = ""
    output: str = ""
    count: int = 0


# ═══════════════════════════════════════════════════════════
# Strategy registry tests
# ═══════════════════════════════════════════════════════════


def _mock_llm(system: str, user: str, tools: list = None, **kwargs) -> str:
    return f"MOCK: {user[:50]}"


def test_register_custom_strategy():
    """Custom strategy can be registered and used."""

    def round_robin_builder(team_obj, task_desc):
        flow = Flow()
        for i, agent in enumerate(team_obj.agents):
            from flowforge.identity import Task

            task = Task(description=task_desc, output_field=agent.name)
            flow.add(agent.name, agent.as_unit(task))
            if i > 0:
                flow.wire(team_obj.agents[i - 1].name, agent.name)
        # Wire last back to first for a round
        flow.wire(team_obj.agents[-1].name, team_obj.agents[0].name, on="retry")
        flow.entry(team_obj.agents[0].name)
        return flow

    Team.register_strategy("round_robin", round_robin_builder)

    a1 = Agent("Alice", "Do A", llm_fn=_mock_llm)
    a2 = Agent("Bob", "Do B", llm_fn=_mock_llm)
    t = Team([a1, a2], strategy="round_robin")

    flow = t.compile("test round robin")
    assert "alice" in flow.nodes
    assert "bob" in flow.nodes
    # Verify the cycle edge exists
    assert any(e[0] == "bob" and e[1] == "alice" for e in flow.edges)

    # Clean up registry to avoid test pollution
    del Team._strategy_registry["round_robin"]


def test_unknown_strategy_raises():
    """Unknown strategy raises ValueError."""
    with pytest.raises(ValueError, match="Unknown strategy"):
        Team([Agent("A", "a", llm_fn=_mock_llm)], strategy="nonexistent")


# ═══════════════════════════════════════════════════════════
# Mermaid export tests
# ═══════════════════════════════════════════════════════════


def test_mermaid_contains_nodes_and_edges():
    """to_mermaid() output contains expected graph elements."""
    flow = Flow()
    flow.add("search", FunctionUnit(lambda s: "default"))
    flow.add("analyze", FunctionUnit(lambda s: "default"))
    flow.add("report", FunctionUnit(lambda s: "default"))
    flow.wire("search", "analyze")
    flow.wire("analyze", "report")
    flow.entry("search")

    mermaid = flow.to_mermaid()

    assert "graph TD" in mermaid
    assert "search --> analyze" in mermaid
    assert "analyze --> report" in mermaid
    # Entry node should have styling
    assert "style search" in mermaid


def test_mermaid_conditional_edges():
    """Mermaid includes action labels on conditional edges."""
    flow = Flow()
    flow.add("check", FunctionUnit(lambda s: "default"))
    flow.add("pass", FunctionUnit(lambda s: "default"))
    flow.add("fail", FunctionUnit(lambda s: "default"))
    flow.wire("check", "pass", on="ok")
    flow.wire("check", "fail", on="error")
    flow.entry("check")

    mermaid = flow.to_mermaid()

    assert "|ok|" in mermaid
    assert "|error|" in mermaid


# ═══════════════════════════════════════════════════════════
# Unit retry tests
# ═══════════════════════════════════════════════════════════


class FailThenSucceedUnit(Unit):
    """Fails N times, then succeeds."""

    def __init__(self, fail_count: int):
        self._fail_count = fail_count
        self._attempts = 0
        self.max_retries = fail_count

    def exec(self, prep_result):
        self._attempts += 1
        if self._attempts <= self._fail_count:
            raise ValueError(f"Attempt {self._attempts} failed")
        return "success"

    def post(self, store, result):
        store.output = result
        return "default"


class FallbackUnit(Unit):
    """Uses exec_fallback after max_retries."""

    def __init__(self):
        self.max_retries = 2
        self.fallback_called = False

    def exec(self, prep_result):
        raise RuntimeError("always fails")

    def exec_fallback(self, prep_result, error):
        self.fallback_called = True
        return "fallback_result"

    def post(self, store, result):
        store.output = result
        return "default"


def test_retry_succeeds_after_failures():
    """Unit retries exec and eventually succeeds."""
    unit = FailThenSucceedUnit(fail_count=2)
    state = RetryState(task="retry test")
    unit.run(state)

    assert state.output == "success"
    assert unit._attempts == 3  # 2 fails + 1 success


def test_retry_calls_fallback():
    """exec_fallback is called after max_retries exhausted."""
    unit = FallbackUnit()
    state = RetryState(task="fallback test")
    unit.run(state)

    assert unit.fallback_called
    assert state.output == "fallback_result"


def test_retry_raises_without_fallback():
    """Without custom fallback, exception propagates after retries."""

    class AlwaysFailUnit(Unit):
        def __init__(self):
            self.max_retries = 1

        def exec(self, prep_result):
            raise RuntimeError("boom")

    unit = AlwaysFailUnit()
    state = RetryState(task="fail")
    with pytest.raises(RuntimeError, match="boom"):
        unit.run(state)


def test_no_retry_by_default():
    """max_retries=0 means no retries — exception raised immediately."""

    class FailUnit(Unit):
        def exec(self, prep_result):
            raise RuntimeError("instant fail")

    unit = FailUnit()
    state = RetryState(task="no retry")
    with pytest.raises(RuntimeError, match="instant fail"):
        unit.run(state)
