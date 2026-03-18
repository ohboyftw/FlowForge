"""
Sprint 2 Tests — Async Core, Streaming, Mixed Sync/Async
"""

import asyncio
import time

import pytest
from pydantic import Field

from flowforge import Agent, AsyncUnit, Flow, FunctionUnit, StoreBase, Team, Unit
from flowforge.harness import AsyncLLMUnit
from flowforge.identity import Persona, Task

# ═══════════════════════════════════════════════════════════
# Test state models
# ═══════════════════════════════════════════════════════════


class AsyncState(StoreBase):
    __reducers__ = {"results": "extend"}

    task: str = ""
    results: list[str] = Field(default_factory=list)
    output: str = ""
    count: int = 0


# ═══════════════════════════════════════════════════════════
# AsyncUnit lifecycle
# ═══════════════════════════════════════════════════════════


class SimpleAsyncUnit(AsyncUnit):
    def __init__(self, name: str, delay: float = 0.0):
        self._name = name
        self._delay = delay

    def prep(self, store):
        return store.task

    async def exec(self, prep_result):
        if self._delay:
            await asyncio.sleep(self._delay)
        return f"async_{self._name}_{prep_result}"

    def post(self, store, result):
        store.results = store.results + [result]
        return "default"


class SyncIncrementUnit(Unit):
    def post(self, store, _):
        store.count += 1
        return "default"


@pytest.mark.asyncio
async def test_async_unit_lifecycle():
    """prep is sync, exec is async, post is sync."""
    unit = SimpleAsyncUnit("test")
    state = AsyncState(task="hello")

    action = await unit.arun(state)

    assert action == "default"
    assert "async_test_hello" in state.results


@pytest.mark.asyncio
async def test_flow_arun_with_async_unit():
    """Flow.arun() runs async units."""
    flow = Flow()
    flow.add("step", SimpleAsyncUnit("one"))
    flow.entry("step")

    state = AsyncState(task="world")
    result = await flow.arun(state)

    assert "async_one_world" in result.results


@pytest.mark.asyncio
async def test_flow_arun_mixed_sync_async():
    """Flow.arun() handles both sync and async units in a chain."""
    flow = Flow()
    flow.add("sync_step", SyncIncrementUnit())
    flow.add("async_step", SimpleAsyncUnit("mixed"))
    flow.wire("sync_step", "async_step")
    flow.entry("sync_step")

    state = AsyncState(task="mix")
    result = await flow.arun(state)

    assert result.count == 1
    assert "async_mixed_mix" in result.results


@pytest.mark.asyncio
async def test_async_parallel_fanout():
    """Async fan-out uses asyncio.gather, runs concurrently."""
    flow = Flow()
    flow.add("dispatch", FunctionUnit(lambda s: "default"))
    flow.add("a", SimpleAsyncUnit("a", delay=0.1))
    flow.add("b", SimpleAsyncUnit("b", delay=0.1))
    flow.add("c", SimpleAsyncUnit("c", delay=0.1))
    flow.wire("dispatch", ["a", "b", "c"])
    flow.entry("dispatch")

    state = AsyncState(task="parallel")
    start = time.monotonic()
    await flow.arun(state)
    elapsed = time.monotonic() - start

    assert elapsed < 0.25, f"Expected <0.25s, got {elapsed:.2f}s"
    assert len(state.results) == 3


# ═══════════════════════════════════════════════════════════
# Streaming tests
# ═══════════════════════════════════════════════════════════


class StreamingUnit(AsyncUnit):
    """An async unit that simulates streaming."""

    def prep(self, store):
        return "prompt"

    async def exec(self, prep_result):
        on_token = getattr(self, "_on_token", None)
        chunks = ["Hello", " ", "World"]
        result = []
        for chunk in chunks:
            if on_token:
                on_token(chunk)
            result.append(chunk)
            await asyncio.sleep(0.01)
        return "".join(result)

    def post(self, store, result):
        store.output = result
        return "default"


@pytest.mark.asyncio
async def test_streaming_callback_receives_tokens():
    """on_token callback receives streaming tokens."""
    flow = Flow()
    flow.add("stream", StreamingUnit())
    flow.entry("stream")

    tokens = []
    state = AsyncState(task="stream test")
    await flow.arun(state, on_token=lambda t: tokens.append(t))

    assert tokens == ["Hello", " ", "World"]
    assert state.output == "Hello World"


# ═══════════════════════════════════════════════════════════
# AsyncLLMUnit tests
# ═══════════════════════════════════════════════════════════


async def _mock_async_llm(
    system: str, user: str, tools: list = None, on_token=None, **kwargs
) -> str:
    await asyncio.sleep(0.01)
    result = f"ASYNC_MOCK: {user[:50]}"
    if on_token:
        for word in result.split():
            on_token(word + " ")
    return result


@pytest.mark.asyncio
async def test_async_llm_unit():
    """AsyncLLMUnit works with async llm_fn."""
    persona = Persona(role="Tester", goal="Test things")
    task = Task(description="Test task", output_field="output")
    unit = AsyncLLMUnit(persona=persona, task=task, llm_fn=_mock_async_llm)

    state = AsyncState(task="hello")
    action = await unit.arun(state)

    assert action == "default"
    assert "ASYNC_MOCK" in state.output


# ═══════════════════════════════════════════════════════════
# AsyncUnit retry tests
# ═══════════════════════════════════════════════════════════


class FailingAsyncUnit(AsyncUnit):
    def __init__(self, fail_count: int):
        self._fail_count = fail_count
        self._attempts = 0
        self.max_retries = fail_count

    async def exec(self, prep_result):
        self._attempts += 1
        if self._attempts <= self._fail_count:
            raise ValueError(f"Fail {self._attempts}")
        return "success"

    def post(self, store, result):
        store.output = result
        return "default"


@pytest.mark.asyncio
async def test_async_retry():
    """AsyncUnit retries on failure."""
    unit = FailingAsyncUnit(fail_count=2)
    state = AsyncState(task="retry")
    await unit.arun(state)
    assert state.output == "success"


# ═══════════════════════════════════════════════════════════
# Backward compatibility
# ═══════════════════════════════════════════════════════════


def test_sync_flow_still_works():
    """Existing sync Flow.run() is unaffected by async additions."""
    flow = Flow()
    flow.add("a", SyncIncrementUnit())
    flow.add("b", SyncIncrementUnit())
    flow.wire("a", "b")
    flow.entry("a")

    state = AsyncState(task="sync")
    result = flow.run(state)
    assert result.count == 2


class AsyncEvalUnit(AsyncUnit):
    """Async evaluator that counts via post."""

    async def exec(self, prep_result):
        await asyncio.sleep(0.01)
        return "evaluated"

    def post(self, store, result):
        store.count += 1
        return "default"


@pytest.mark.asyncio
async def test_async_loop_with_async_evaluator():
    """flow.loop() works with AsyncUnit evaluator via arun()."""
    flow = Flow()

    class GenUnit(Unit):
        def post(self, store, _):
            store.results = store.results + [f"gen_{store.count}"]
            return "default"

    flow.add("gen", GenUnit())
    flow.add("eval", AsyncEvalUnit())
    flow.loop("gen", "eval", until=lambda s: s.count >= 3, max_rounds=5)
    flow.entry("gen")

    state = AsyncState(task="async_loop")
    await flow.arun(state)

    assert state.count >= 3
    assert len(state.results) >= 3


@pytest.mark.asyncio
async def test_async_trace_has_duration():
    """arun() trace entries include duration_ms."""
    flow = Flow()
    flow.add("step", SimpleAsyncUnit("timed"))
    flow.entry("step")

    state = AsyncState(task="timing")
    await flow.arun(state)

    for entry in flow.trace:
        assert "duration_ms" in entry
        assert isinstance(entry["duration_ms"], float)
        assert entry["duration_ms"] >= 0


# ═══════════════════════════════════════════════════════════
# Team.arun() tests
# ═══════════════════════════════════════════════════════════


def _mock_sync_team_llm(system: str, user: str, tools: list = None, **kw) -> str:
    return f"MOCK: {user[:50]}"


@pytest.mark.asyncio
async def test_team_arun():
    """Team.arun() runs the compiled flow asynchronously."""
    a1 = Agent("Step1", "Do step 1", llm_fn=_mock_sync_team_llm)
    a2 = Agent("Step2", "Do step 2", llm_fn=_mock_sync_team_llm)
    t = Team([a1, a2], strategy="sequential")

    result = await t.arun("test task")

    assert result is not None
    assert hasattr(result, "step1") or hasattr(result, "task")


@pytest.mark.asyncio
async def test_team_arun_forwards_params():
    """Team.arun() forwards max_steps and raise_on_exhaust."""
    a1 = Agent("Worker", "Work", llm_fn=_mock_sync_team_llm)
    t = Team([a1], strategy="sequential")

    # Should complete with generous max_steps
    result = await t.arun("test", max_steps=200, raise_on_exhaust=False)
    assert result is not None


# ═══════════════════════════════════════════════════════════
# Async fan-out continuation + error routing + timeout
# ═══════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_async_fanout_continuation():
    """Async fan-out continues to next wire after merge."""
    flow = Flow()
    flow.add("dispatch", FunctionUnit(lambda s: "default"))
    flow.add("a", SimpleAsyncUnit("a"))
    flow.add("b", SimpleAsyncUnit("b"))
    flow.add("final", FunctionUnit(lambda s: setattr(s, "output", "done") or "default"))

    flow.wire("dispatch", ["a", "b"])
    flow.wire("dispatch", "final")
    flow.entry("dispatch")

    state = AsyncState(task="continue")
    await flow.arun(state)
    assert state.output == "done"


@pytest.mark.asyncio
async def test_async_unit_timeout():
    """AsyncUnit timeout via asyncio.wait_for."""

    class SlowAsyncUnit(AsyncUnit):
        timeout = 0.1

        async def exec(self, _):
            await asyncio.sleep(2.0)
            return "done"

        def post(self, store, result):
            store.output = result
            return "default"

    class AsyncErrorHandler(AsyncUnit):
        async def exec(self, _):
            return "handled"

        def post(self, store, _):
            store.output = "timeout_handled"
            return "default"

    flow = Flow()
    flow.add("slow", SlowAsyncUnit())
    flow.add("handler", AsyncErrorHandler())
    flow.wire("slow", "handler", on="error")
    flow.entry("slow")

    state = AsyncState(task="timeout")
    await flow.arun(state)
    assert state.output == "timeout_handled"


@pytest.mark.asyncio
async def test_async_error_routing():
    """Error routing works in arun path."""

    class AsyncFailUnit(AsyncUnit):
        async def exec(self, _):
            raise ValueError("async boom")

    class AsyncHandlerUnit(AsyncUnit):
        async def exec(self, _):
            return "handled"

        def post(self, store, _):
            store.output = "error_handled"
            return "default"

    flow = Flow()
    flow.add("fail", AsyncFailUnit())
    flow.add("handler", AsyncHandlerUnit())
    flow.wire("fail", "handler", on="error")
    flow.entry("fail")

    state = AsyncState(task="error")
    await flow.arun(state)
    assert state.output == "error_handled"
