"""
Sprint 2 Tests — Async Core, Streaming, Mixed Sync/Async
"""

import asyncio
import time

import pytest
from pydantic import Field

from flowforge import AsyncUnit, Flow, FunctionUnit, StoreBase, Unit
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
