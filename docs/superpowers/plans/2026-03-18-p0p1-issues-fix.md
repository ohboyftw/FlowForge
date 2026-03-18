# P0+P1 Issues Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 10 issues (4 P0, 6 P1) across FlowForge's core and harness layers, adding wire validation, error routing, timeout support, fan-out chaining, and async completeness.

**Architecture:** Four dependency-ordered groups: (1) Foundation — cleanup + safety nets in core.py, (2) Async fixes — CountingWrapper + Team.arun(), (3) Flow engine — fan-out continuation, timeout, error routing, (4) Harness — hierarchical parallel fix. Each group builds on the previous.

**Tech Stack:** Python 3.10+, Pydantic v2, pytest, pytest-asyncio (strict mode)

**Spec:** `docs/superpowers/specs/2026-03-18-p0p1-issues-design.md`

**Boundary tests (acceptance criteria):** `tests/test_boundary_contracts.py` — 14 of 21 tests currently fail; all 21 must pass when implementation is complete.

---

## File Map

| File | Responsibility | Tasks |
|------|---------------|-------|
| `src/flowforge/core.py` | Primitive layer: Store, Unit, AsyncUnit, Flow | 1–7 |
| `src/flowforge/harness.py` | Harness layer: Agent, Team, LLMUnit | 8–10 |
| `src/flowforge/__init__.py` | Public exports | 3 |
| `tests/test_flowforge.py` | Sync unit + flow tests | 1–4, 7 |
| `tests/test_async.py` | Async unit + flow tests | 5–7 |
| `tests/test_smoke.py` | Harness smoke tests | 10 |

---

## Group 1: Foundation

### Task 1: Remove unreachable retry code (MINOR-001)

**Files:**
- Modify: `src/flowforge/core.py:295` and `src/flowforge/core.py:333`
- Test: `tests/test_sprint3.py` (existing retry tests)

- [ ] **Step 1: Verify existing retry tests pass**

Run: `py -m pytest tests/test_sprint3.py -v -k retry`
Expected: 4 tests PASS

- [ ] **Step 2: Remove dead code in Unit._exec_with_retry**

In `src/flowforge/core.py`, delete line 295:

```python
# DELETE this line:
        raise last_error  # unreachable but satisfies type checker
```

The method ends after the for-loop (which always returns from `exec()` or `exec_fallback()`).

- [ ] **Step 3: Remove dead code in AsyncUnit._exec_with_retry**

In `src/flowforge/core.py`, delete line 333:

```python
# DELETE this line:
        raise last_error  # unreachable
```

- [ ] **Step 4: Run full test suite to confirm no regressions**

Run: `py -m pytest tests/ -v`
Expected: All 58 existing tests PASS (+ 7 boundary tests that already pass)

- [ ] **Step 5: Commit**

```bash
git add src/flowforge/core.py
git commit -m "fix: remove unreachable code in retry loops (MINOR-001)"
```

---

### Task 2: Add lazy wire validation (DESIGN-005)

**Files:**
- Modify: `src/flowforge/core.py:489-493` (relax `entry()`), `src/flowforge/core.py:498` (add `_validate_graph` call to `run()`), `src/flowforge/core.py:540` (add to `arun()`)
- Test: `tests/test_flowforge.py`

- [ ] **Step 1: Write failing tests for wire validation**

Add to `tests/test_flowforge.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -m pytest tests/test_flowforge.py::TestWireValidation -v`
Expected: `test_wire_validation_catches_typo` FAILS (current code raises RuntimeError, not ValueError), `test_wire_validation_allows_late_add` PASSES, `test_entry_allows_lazy_set` FAILS (current `entry()` raises eagerly)

- [ ] **Step 3: Add `_validate_graph` method to Flow**

In `src/flowforge/core.py`, add method to `Flow` class (after `entry()`, before `run()`):

```python
    def _validate_graph(self) -> None:
        """Validate all wire targets reference registered units. Called at run time."""
        errors = []
        for src, wires in self._wires.items():
            if src not in self._units:
                errors.append(f"Wire source '{src}' is not a registered unit")
            for w in wires:
                targets = w.target if isinstance(w.target, list) else [w.target]
                for t in targets:
                    if t not in self._units:
                        errors.append(
                            f"Wire target '{t}' (from '{src}') is not a registered unit"
                        )
        if not self._entry:
            errors.append("No entry point set")
        elif self._entry not in self._units:
            errors.append(f"Entry point '{self._entry}' is not a registered unit")
        if errors:
            raise ValueError("Invalid flow graph:\n  " + "\n  ".join(errors))
```

- [ ] **Step 4: Relax `entry()` — remove eager validation**

In `src/flowforge/core.py`, change `entry()` method from:

```python
    def entry(self, name: str) -> Flow[S]:
        """Set the entry point node."""
        if name not in self._units:
            raise ValueError(f"Unknown unit '{name}'")
        self._entry = name
        return self
```

To:

```python
    def entry(self, name: str) -> Flow[S]:
        """Set the entry point node. Validated lazily at run time."""
        self._entry = name
        return self
```

- [ ] **Step 5: Call `_validate_graph()` at start of `run()`**

In `src/flowforge/core.py`, in `Flow.run()`, replace the existing entry check:

```python
        if not self._entry:
            raise RuntimeError("No entry point set. Call flow.entry('name')")
```

With:

```python
        self._validate_graph()
```

- [ ] **Step 6: Call `_validate_graph()` at start of `arun()`**

In `src/flowforge/core.py`, in `Flow.arun()`, replace the same entry check with:

```python
        self._validate_graph()
```

- [ ] **Step 7: Run tests**

Run: `py -m pytest tests/test_flowforge.py::TestWireValidation -v && py -m pytest tests/ -v`
Expected: All new tests PASS, all existing tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/flowforge/core.py tests/test_flowforge.py
git commit -m "feat: add lazy wire validation at run time (DESIGN-005)"
```

---

### Task 3: Add max_steps warning + FlowExhaustedError (DESIGN-007)

**Files:**
- Modify: `src/flowforge/core.py` (add imports, exception class, parameter, warning)
- Modify: `src/flowforge/__init__.py` (export)
- Test: `tests/test_flowforge.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_flowforge.py`:

```python
import logging
from flowforge import FlowExhaustedError

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -m pytest tests/test_flowforge.py::TestMaxSteps -v`
Expected: FAIL — `FlowExhaustedError` not importable, no warning logged

- [ ] **Step 3: Add imports and FlowExhaustedError to core.py**

At the top of `src/flowforge/core.py`, add after existing imports:

```python
import logging
import time

logger = logging.getLogger(__name__)
```

Before the `Unit` class, add the exception:

```python
class FlowExhaustedError(Exception):
    """Raised when Flow.run() hits the max_steps limit."""

    def __init__(self, steps: int, last_unit: str):
        self.steps = steps
        self.last_unit = last_unit
        super().__init__(f"Flow exhausted after {steps} steps at unit '{last_unit}'")
```

- [ ] **Step 4: Add `raise_on_exhaust` parameter and warning to `run()`**

Change `Flow.run()` signature to:

```python
    def run(self, store: S, *, max_steps: int = 100, raise_on_exhaust: bool = False) -> S:
```

After the while loop (before `return store`), add:

```python
        if steps >= max_steps:
            logger.warning(
                "Flow hit max_steps=%d at unit '%s'", max_steps, current
            )
            if raise_on_exhaust:
                raise FlowExhaustedError(steps, current)
```

- [ ] **Step 5: Same for `arun()`**

Change `Flow.arun()` signature to:

```python
    async def arun(
        self,
        store: S,
        *,
        max_steps: int = 100,
        raise_on_exhaust: bool = False,
        on_token: Callable | None = None,
    ) -> S:
```

Add the same warning/raise block after the while loop.

- [ ] **Step 6: Export FlowExhaustedError**

In `src/flowforge/__init__.py`, add to the core imports:

```python
from flowforge.core import (
    ...
    FlowExhaustedError,
    ...
)
```

And add `"FlowExhaustedError"` to `__all__`.

- [ ] **Step 7: Run tests**

Run: `py -m pytest tests/test_flowforge.py::TestMaxSteps -v && py -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/flowforge/core.py src/flowforge/__init__.py tests/test_flowforge.py
git commit -m "feat: add max_steps warning + FlowExhaustedError (DESIGN-007)"
```

---

### Task 4: Add timing to trace entries (FEAT-004)

**Files:**
- Modify: `src/flowforge/core.py` (`run()` and `arun()` trace entries)
- Test: `tests/test_flowforge.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_flowforge.py`:

```python
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
```

Also add an async timing test to `tests/test_async.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_flowforge.py::TestTraceTiming -v`
Expected: FAIL — `duration_ms` not in trace entries

- [ ] **Step 3: Add timing to `Flow.run()`**

In `Flow.run()`, wrap the unit execution and trace append. Change the body of the while loop from:

```python
            # Execute unit
            action = unit.run(store)
            self._trace.append(
                {
                    "step": steps,
                    "unit": current,
                    "action": action,
                    "state_snapshot": store.model_dump(),
                }
            )
```

To:

```python
            # Execute unit
            start = time.monotonic()
            action = unit.run(store)
            end = time.monotonic()
            self._trace.append(
                {
                    "step": steps,
                    "unit": current,
                    "action": action,
                    "duration_ms": round((end - start) * 1000, 2),
                    "state_snapshot": store.model_dump(),
                }
            )
```

- [ ] **Step 4: Add timing to `Flow.arun()`**

Same pattern in `arun()`. Wrap the unit execution block:

```python
            start = time.monotonic()
            if isinstance(unit, AsyncUnit):
                if on_token:
                    unit._on_token = on_token
                action = await unit.arun(store)
            else:
                action = unit.run(store)
            end = time.monotonic()
            self._trace.append(
                {
                    "step": steps,
                    "unit": current,
                    "action": action,
                    "duration_ms": round((end - start) * 1000, 2),
                    "state_snapshot": store.model_dump(),
                }
            )
```

- [ ] **Step 5: Run tests**

Run: `py -m pytest tests/test_flowforge.py::TestTraceTiming -v && py -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/flowforge/core.py tests/test_flowforge.py
git commit -m "feat: add duration_ms timing to flow trace entries (FEAT-004)"
```

---

## Group 2: Async Fixes

### Task 5: Add _AsyncCountingWrapper (BUG-002)

**Files:**
- Modify: `src/flowforge/core.py` (add `_AsyncCountingWrapper`, update `loop()`)
- Test: `tests/test_async.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_async.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_async.py::test_async_loop_with_async_evaluator -v`
Expected: FAIL — `_CountingWrapper` wraps as `Unit`, `arun()` calls `unit.run()` for non-AsyncUnit

- [ ] **Step 3: Add `_AsyncCountingWrapper` class**

In `src/flowforge/core.py`, after the `_CountingWrapper` class, add:

```python
class _AsyncCountingWrapper(AsyncUnit):
    """Wraps an AsyncUnit to count executions for loop() termination."""

    def __init__(self, inner: AsyncUnit, counter: list[int]):
        self._inner = inner
        self._counter = counter
        self.max_retries = getattr(inner, "max_retries", 0)

    def prep(self, store):
        return self._inner.prep(store)

    async def exec(self, prep_result):
        return await self._inner.exec(prep_result)

    def post(self, store, exec_result):
        self._counter[0] += 1
        return self._inner.post(store, exec_result)
```

- [ ] **Step 4: Update `Flow.loop()` to select wrapper by type**

In `Flow.loop()`, change:

```python
        self._units[evaluator] = _CountingWrapper(original_unit, round_counter)
```

To:

```python
        if isinstance(original_unit, AsyncUnit):
            self._units[evaluator] = _AsyncCountingWrapper(original_unit, round_counter)
        else:
            self._units[evaluator] = _CountingWrapper(original_unit, round_counter)
```

- [ ] **Step 5: Run tests**

Run: `py -m pytest tests/test_async.py::test_async_loop_with_async_evaluator -v && py -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/flowforge/core.py tests/test_async.py
git commit -m "fix: add _AsyncCountingWrapper for async loop evaluators (BUG-002)"
```

---

### Task 6: Add Team.arun() (DESIGN-003)

**Files:**
- Modify: `src/flowforge/harness.py` (add `arun` method to `Team`)
- Test: `tests/test_async.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_async.py`:

```python
from flowforge import Agent, FlexStore, Team


async def _mock_async_team_llm(system: str, user: str, tools: list = None, **kw) -> str:
    return f"MOCK: {user[:50]}"


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -m pytest tests/test_async.py::test_team_arun -v`
Expected: FAIL — `Team` has no `arun` method

- [ ] **Step 3: Add `Team.arun()` to harness.py**

In `src/flowforge/harness.py`, add after the `run()` method in the `Team` class:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `py -m pytest tests/test_async.py -k "team_arun" -v && py -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/flowforge/harness.py tests/test_async.py
git commit -m "feat: add Team.arun() for async team execution (DESIGN-003)"
```

---

## Group 3: Flow Engine

### Task 7: Fan-out continuation + error routing + timeout (DESIGN-006, FEAT-005, FEAT-010)

These three features are tightly coupled in `Flow.run()` and `Flow.arun()` — the execution loop changes overlap. Implementing them together avoids rewriting the loop three times.

**Files:**
- Modify: `src/flowforge/core.py` (Flow.__init__, run(), arun(), _resolve_next, _aresolve_next, Unit, AsyncUnit)
- Test: `tests/test_flowforge.py`, `tests/test_async.py`

- [ ] **Step 1: Write failing tests for fan-out continuation**

Add to `tests/test_flowforge.py`:

```python
class TestFanoutContinuation:
    def test_fanout_then_continue(self):
        """After fan-out completes, next wire from same source runs."""
        flow = Flow()
        flow.add("dispatch", FunctionUnit(lambda s: "default"))
        flow.add("w1", IncrementUnit("w1", 1))
        flow.add("w2", IncrementUnit("w2", 10))
        flow.add("aggregate", IncrementUnit("aggregate", 100))

        flow.wire("dispatch", ["w1", "w2"])  # fan-out
        flow.wire("dispatch", "aggregate")   # continuation
        flow.entry("dispatch")

        result = flow.run(CounterState())
        # w1=1, w2=10 merged, then aggregate=+100
        # Fan-out uses last-write-wins by default, so count = max(1,10) = 10
        # then aggregate adds 100 → 110
        assert result.count >= 100  # aggregate ran

    def test_multiple_fanouts_sequential_barrier(self):
        """Multiple fan-out wires on same node run sequentially."""
        flow = Flow()
        flow.add("dispatch", FunctionUnit(lambda s: "default"))
        flow.add("a", IncrementUnit("a", 1))
        flow.add("b", IncrementUnit("b", 10))
        flow.add("final", IncrementUnit("final", 100))

        flow.wire("dispatch", ["a"])      # first fan-out
        flow.wire("dispatch", ["b"])      # second fan-out
        flow.wire("dispatch", "final")    # continuation
        flow.entry("dispatch")

        result = flow.run(CounterState())
        assert result.count >= 100  # final ran after both fan-outs
```

- [ ] **Step 2: Write failing tests for error routing**

Add to `tests/test_flowforge.py`:

```python
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
```

- [ ] **Step 3: Write failing tests for timeout**

Add to `tests/test_flowforge.py`:

```python
import time as time_mod


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
```

- [ ] **Step 4: Run all new tests to verify they fail**

Run: `py -m pytest tests/test_flowforge.py::TestFanoutContinuation tests/test_flowforge.py::TestErrorRouting tests/test_flowforge.py::TestTimeout -v`
Expected: All FAIL

- [ ] **Step 5: Add `on_error` to `Flow.__init__()`**

In `src/flowforge/core.py`, change `Flow.__init__` from:

```python
    def __init__(self, reducers: ReducerRegistry = None):
        self._units: dict[str, Unit] = {}
        self._wires: dict[str, list[Wire]] = {}
        self._entry: str | None = None
        self._reducers = reducers or ReducerRegistry()
        self._trace: list[dict] = []
```

To:

```python
    def __init__(self, reducers: ReducerRegistry = None, on_error: str | None = None):
        self._units: dict[str, Unit] = {}
        self._wires: dict[str, list[Wire]] = {}
        self._entry: str | None = None
        self._reducers = reducers or ReducerRegistry()
        self._trace: list[dict] = []
        self._on_error = on_error
```

- [ ] **Step 6: Add `timeout` attribute to Unit and AsyncUnit**

In `src/flowforge/core.py`, add to `Unit` class (after `max_retries`):

```python
    max_retries: int = 0
    timeout: float | None = None
```

And to `AsyncUnit` (after `max_retries`):

```python
    max_retries: int = 0
    timeout: float | None = None
```

- [ ] **Step 7: Add `_resolve_error_target` helper**

In `src/flowforge/core.py`, add to `Flow` class (after `_resolve_next`):

```python
    def _resolve_error_target(self, current: str) -> str | None:
        """Find a wire with on='error' from the current node."""
        for w in self._wires.get(current, []):
            if w.on == "error":
                return w.target if isinstance(w.target, str) else None
        return None
```

- [ ] **Step 8: Change `_resolve_next` — fan-out continues instead of returning None**

In `src/flowforge/core.py`, change `_resolve_next`:

```python
    def _resolve_next(self, current: str, action: str, store: S) -> str | None:
        """Resolve the next node based on wires, action label, and conditions."""
        wires = self._wires.get(current, [])

        for w in wires:
            if w.on != action and w.on != "*":
                continue
            if w.when is not None and not w.when(store):
                continue
            if w.interrupt:
                raise InterruptSignal(w, store, current)
            if isinstance(w.target, list):
                self._run_fanout(w.target, store)
                continue  # fan-out done, check next wire for continuation
            return w.target

        return None
```

- [ ] **Step 9: Change `_aresolve_next` — same fan-out continuation for async**

```python
    async def _aresolve_next(
        self, current: str, action: str, store: S, on_token: Callable | None = None
    ) -> str | None:
        """Async version of _resolve_next with asyncio.gather for fan-out."""
        wires = self._wires.get(current, [])

        for w in wires:
            if w.on != action and w.on != "*":
                continue
            if w.when is not None and not w.when(store):
                continue
            if w.interrupt:
                raise InterruptSignal(w, store, current)
            if isinstance(w.target, list):
                await self._arun_fanout(w.target, store, on_token)
                continue  # fan-out done, check next wire for continuation
            return w.target

        return None
```

- [ ] **Step 10: Rewrite `Flow.run()` with error routing + timeout + timing**

Replace the entire `Flow.run()` method body (keep signature from Task 3):

```python
    def run(
        self, store: S, *, max_steps: int = 100, raise_on_exhaust: bool = False,
        default_timeout: float | None = None,
    ) -> S:
        """Execute the flow. Returns the final store state."""
        self._validate_graph()

        self._trace = []
        current = self._entry
        steps = 0

        while current and steps < max_steps:
            steps += 1
            unit = self._units.get(current)
            if unit is None:
                raise RuntimeError(f"Unknown unit '{current}'")

            start = time.monotonic()
            object.__setattr__(store, "_last_error", None)
            try:
                effective_timeout = getattr(unit, "timeout", None) or default_timeout
                if effective_timeout is not None:
                    store_copy = store.model_copy(deep=True)
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(unit.run, store_copy)
                        action = future.result(timeout=effective_timeout)
                        for field_name in store.__class__.model_fields:
                            setattr(store, field_name, getattr(store_copy, field_name))
                else:
                    action = unit.run(store)
            except InterruptSignal:
                raise
            except Exception as exc:
                end = time.monotonic()
                object.__setattr__(store, "_last_error", exc)
                self._trace.append(
                    {
                        "step": steps,
                        "unit": current,
                        "action": "error",
                        "duration_ms": round((end - start) * 1000, 2),
                        "error": str(exc),
                        "state_snapshot": store.model_dump(),
                    }
                )
                error_target = self._resolve_error_target(current)
                if error_target:
                    current = error_target
                    continue
                if self._on_error and self._on_error in self._units:
                    current = self._on_error
                    continue
                raise
            else:
                end = time.monotonic()
                self._trace.append(
                    {
                        "step": steps,
                        "unit": current,
                        "action": action,
                        "duration_ms": round((end - start) * 1000, 2),
                        "state_snapshot": store.model_dump(),
                    }
                )

            next_node = self._resolve_next(current, action, store)
            current = next_node

        if steps >= max_steps:
            logger.warning("Flow hit max_steps=%d at unit '%s'", max_steps, current)
            if raise_on_exhaust:
                raise FlowExhaustedError(steps, current)

        return store
```

- [ ] **Step 11: Rewrite `Flow.arun()` with error routing + timeout + timing**

Replace the entire `Flow.arun()` method body:

```python
    async def arun(
        self,
        store: S,
        *,
        max_steps: int = 100,
        raise_on_exhaust: bool = False,
        on_token: Callable | None = None,
        default_timeout: float | None = None,
    ) -> S:
        """Async execution path with error routing and timeout support."""
        self._validate_graph()

        self._trace = []
        current = self._entry
        steps = 0

        while current and steps < max_steps:
            steps += 1
            unit = self._units.get(current)
            if unit is None:
                raise RuntimeError(f"Unknown unit '{current}'")

            start = time.monotonic()
            object.__setattr__(store, "_last_error", None)
            try:
                effective_timeout = getattr(unit, "timeout", None) or default_timeout
                if isinstance(unit, AsyncUnit):
                    if on_token:
                        unit._on_token = on_token
                    if effective_timeout is not None:
                        action = await asyncio.wait_for(
                            unit.arun(store), timeout=effective_timeout
                        )
                    else:
                        action = await unit.arun(store)
                else:
                    if effective_timeout is not None:
                        store_copy = store.model_copy(deep=True)
                        with ThreadPoolExecutor(max_workers=1) as pool:
                            future = pool.submit(unit.run, store_copy)
                            action = future.result(timeout=effective_timeout)
                            for field_name in store.__class__.model_fields:
                                setattr(
                                    store, field_name,
                                    getattr(store_copy, field_name),
                                )
                    else:
                        action = unit.run(store)
            except InterruptSignal:
                raise
            except Exception as exc:
                end = time.monotonic()
                object.__setattr__(store, "_last_error", exc)
                self._trace.append(
                    {
                        "step": steps,
                        "unit": current,
                        "action": "error",
                        "duration_ms": round((end - start) * 1000, 2),
                        "error": str(exc),
                        "state_snapshot": store.model_dump(),
                    }
                )
                error_target = self._resolve_error_target(current)
                if error_target:
                    current = error_target
                    continue
                if self._on_error and self._on_error in self._units:
                    current = self._on_error
                    continue
                raise
            else:
                end = time.monotonic()
                self._trace.append(
                    {
                        "step": steps,
                        "unit": current,
                        "action": action,
                        "duration_ms": round((end - start) * 1000, 2),
                        "state_snapshot": store.model_dump(),
                    }
                )

            next_node = await self._aresolve_next(current, action, store, on_token)
            current = next_node

        if steps >= max_steps:
            logger.warning("Flow hit max_steps=%d at unit '%s'", max_steps, current)
            if raise_on_exhaust:
                raise FlowExhaustedError(steps, current)

        return store
```

- [ ] **Step 12: Write async tests for fan-out continuation + error routing**

Add to `tests/test_async.py`:

```python
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
```

- [ ] **Step 13: Run all tests**

Run: `py -m pytest tests/ -v`
Expected: All existing + new tests PASS

- [ ] **Step 14: Run boundary contract tests**

Run: `py -m pytest tests/test_boundary_contracts.py -v`
Expected: Most should now PASS (Groups 1–3 complete). B5 (Team forwarding), B7 (hierarchical) may still fail.

- [ ] **Step 15: Commit**

```bash
git add src/flowforge/core.py tests/test_flowforge.py tests/test_async.py
git commit -m "feat: add fan-out continuation, error routing, timeout (DESIGN-006, FEAT-005, FEAT-010)"
```

---

## Group 4: Harness

### Task 8: Fix hierarchical strategy — parallel workers (BUG-001)

**Files:**
- Modify: `src/flowforge/harness.py` (`_build_hierarchical`)
- Test: `tests/test_smoke.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_smoke.py`:

```python
def test_hierarchical_parallel_workers():
    """Hierarchical team wires workers as fan-out, not sequential."""
    mgr = Agent("Manager", "Manage", llm_fn=_mock_llm)
    w1 = Agent("Dev1", "Code", llm_fn=_mock_llm)
    w2 = Agent("Dev2", "Test", llm_fn=_mock_llm)
    t = Team([w1, w2], strategy="hierarchical", manager=mgr)

    flow = t.compile("build")

    # Workers should appear in a fan-out from decompose
    decompose_edges = [(s, tgt, lbl) for s, tgt, lbl in flow.edges if s == "decompose"]
    targets = [tgt for _, tgt, _ in decompose_edges]

    # Both workers and synthesize should be targets of decompose
    assert "dev1" in targets
    assert "dev2" in targets
    assert "synthesize" in targets

    # No direct worker → synthesize edges (continuation handles it)
    worker_synth = [(s, t) for s, t, _ in flow.edges if s in ("dev1", "dev2") and t == "synthesize"]
    assert worker_synth == []

    # End-to-end: should run to completion
    store = FlexStore(task="hierarchical test")
    result = t.run("build something", store=store)
    assert hasattr(result, "final_result")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_smoke.py::test_hierarchical_parallel_workers -v`
Expected: FAIL — current code wires workers individually

- [ ] **Step 3: Replace `_build_hierarchical` in harness.py**

In `src/flowforge/harness.py`, replace the `_build_hierarchical` method:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `py -m pytest tests/test_smoke.py -v && py -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/flowforge/harness.py tests/test_smoke.py
git commit -m "fix: hierarchical strategy uses fan-out for parallel workers (BUG-001)"
```

---

### Task 9: Fix consensus strategy for new fan-out semantics

**Files:**
- Modify: `src/flowforge/harness.py` (`_build_consensus`)
- Test: `tests/test_smoke.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_smoke.py`:

```python
def test_consensus_reaches_consensus_node():
    """Consensus strategy reaches the consensus node with new fan-out semantics."""
    agents = [
        Agent("ReviewerA", "Review A", llm_fn=_mock_llm),
        Agent("ReviewerB", "Review B", llm_fn=_mock_llm),
    ]
    team = Team(agents, strategy="consensus")

    store = FlexStore(task="consensus test")
    result = team.run("review code", store=store)

    # The flow should complete — consensus node ran
    assert result is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_smoke.py::test_consensus_reaches_consensus_node -v`
Expected: FAIL — fan-out continuation means old per-worker wires don't reach consensus

- [ ] **Step 3: Replace `_build_consensus` in harness.py**

```python
    def _build_consensus(self, task_desc: str = "") -> Flow:
        """All agents work in parallel, then vote/merge."""
        flow = self._build_parallel(task_desc)

        def consensus_fn(store):
            outputs = {}
            for agent in self.agents:
                val = getattr(store, agent.name, None)
                if val:
                    outputs[agent.name] = val
            if hasattr(store, "consensus_inputs"):
                store.consensus_inputs = outputs
            return "default"

        flow.add("consensus", FunctionUnit(consensus_fn))

        # Continuation wire from dispatch → consensus (after fan-out completes)
        flow.wire("dispatch", "consensus")

        return flow
```

- [ ] **Step 4: Run tests**

Run: `py -m pytest tests/test_smoke.py -v && py -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/flowforge/harness.py tests/test_smoke.py
git commit -m "fix: consensus strategy uses continuation wire for new fan-out semantics"
```

---

### Task 10: Final verification — all boundary + existing tests pass

**Files:** None (verification only)

- [ ] **Step 1: Run all existing tests**

Run: `py -m pytest tests/ -v --tb=short`
Expected: All tests PASS (58 original + ~25 new + 21 boundary = ~100+)

- [ ] **Step 2: Run boundary contract tests specifically**

Run: `py -m pytest tests/test_boundary_contracts.py -v`
Expected: All 21 PASS (was 7 pass, 14 fail before implementation)

- [ ] **Step 3: Lint check**

Run: `py -m ruff check src/ tests/ && py -m ruff format --check src/ tests/`
Expected: Clean

- [ ] **Step 4: Fix any lint issues**

Run: `py -m ruff check src/ tests/ --fix && py -m ruff format src/ tests/`

- [ ] **Step 5: Final commit if lint fixes needed**

```bash
git add -u
git commit -m "style: lint and format fixes"
```

- [ ] **Step 6: Tag completion**

```bash
git log --oneline -12
```

Verify all group commits are present and clean.
