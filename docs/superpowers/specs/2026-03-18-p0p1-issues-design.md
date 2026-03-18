# FlowForge P0+P1 Issues Fix — Design Spec

**Date**: 2026-03-18
**Scope**: 10 issues (4 P0, 6 P1) from `docs/ISSUES.md`
**Approach**: Dependency-ordered groups — each group builds on the previous

---

## Items Covered

| # | ID | Summary | Priority | Group |
|---|-----|---------|----------|-------|
| 1 | MINOR-001 | Unreachable code in retry loop | P0 | 1 |
| 2 | DESIGN-005 | Wire validation | P0 | 1 |
| 3 | DESIGN-007 | Silent max_steps cutoff | P0 | 1 |
| 4 | FEAT-004 | Timing in trace | P1 | 1 |
| 5 | BUG-002 | _CountingWrapper breaks with AsyncUnit | P1 | 2 |
| 6 | DESIGN-003 | No async Team.run() | P1 | 2 |
| 7 | DESIGN-006 | Fan-out can't chain to downstream | P1 | 3 |
| 8 | FEAT-005 | Per-unit timeout support | P1 | 3 |
| 9 | FEAT-010 | Error routing (dead-letter wire) | P1 | 3 |
| 10 | BUG-001 | Hierarchical strategy not parallel | P0 | 4 |

---

## Group 1: Foundation

**Files**: `core.py`
**Dependencies**: None

### MINOR-001: Remove unreachable retry code

Delete `raise last_error` at the end of `_exec_with_retry` in both `Unit` (line 295) and `AsyncUnit` (line 333). The loop always either returns from `exec()` or returns from `exec_fallback()` — the line is dead code. The type checker doesn't need it because `exec_fallback` already has a return type annotation.

### DESIGN-005: Lazy wire validation

Add `_validate_graph()` private method on `Flow`:

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
                    errors.append(f"Wire target '{t}' (from '{src}') is not a registered unit")
    if not self._entry:
        errors.append("No entry point set")
    elif self._entry not in self._units:
        errors.append(f"Entry point '{self._entry}' is not a registered unit")
    if errors:
        raise ValueError(f"Invalid flow graph:\n  " + "\n  ".join(errors))
```

Called at the top of `run()` and `arun()`, before the execution loop. Construction-time methods (`wire()`, `add()`, `entry()`) remain permissive — users can wire before adding nodes.

### DESIGN-007: max_steps warning + opt-in raise

Add `FlowExhaustedError` exception class:

```python
class FlowExhaustedError(Exception):
    def __init__(self, steps: int, last_unit: str):
        self.steps = steps
        self.last_unit = last_unit
        super().__init__(f"Flow exhausted after {steps} steps at unit '{last_unit}'")
```

Add `raise_on_exhaust: bool = False` parameter to `run()` and `arun()`. After the while loop exits:

```python
if steps >= max_steps:
    logger.warning("Flow hit max_steps=%d at unit '%s'", max_steps, current)
    if raise_on_exhaust:
        raise FlowExhaustedError(steps, current)
```

Add `import logging` and `logger = logging.getLogger(__name__)` to `core.py`.

Export `FlowExhaustedError` from `__init__.py`.

### FEAT-004: Timing in trace entries

Wrap unit execution with `time.monotonic()` in both `run()` and `arun()`. The trace entry becomes:

```python
{
    "step": steps,
    "unit": current,
    "action": action,
    "duration_ms": round((end - start) * 1000, 2),
    "state_snapshot": store.model_dump(),
}
```

`time` is already imported. No new classes, no logging, no hooks — just data in the existing trace list. A future `Flow.stats()` method can aggregate from `_trace` without any changes to this structure.

---

## Group 2: Async Fixes

**Files**: `core.py`, `harness.py`
**Dependencies**: Group 1 (validated graph, trace timing)

### BUG-002: _CountingWrapper async support

Two wrapper classes instead of one:

```python
class _CountingWrapper(Unit):
    """Wraps a sync Unit to count executions for loop() termination."""
    def __init__(self, inner: Unit, counter: list[int]):
        self._inner = inner
        self._counter = counter
        self.max_retries = getattr(inner, "max_retries", 0)

    def prep(self, store):
        return self._inner.prep(store)

    def exec(self, prep_result):
        return self._inner.exec(prep_result)

    def post(self, store, exec_result):
        self._counter[0] += 1
        return self._inner.post(store, exec_result)


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

`Flow.loop()` selects the wrapper based on the evaluator type:

```python
if isinstance(original_unit, AsyncUnit):
    self._units[evaluator] = _AsyncCountingWrapper(original_unit, round_counter)
else:
    self._units[evaluator] = _CountingWrapper(original_unit, round_counter)
```

### DESIGN-003: Team.arun()

Add to `Team`:

```python
async def arun(self, task_desc: str, *, store: StoreBase = None) -> StoreBase:
    """Async execution. Mirrors run() but uses flow.arun()."""
    flow = self.compile(task_desc)
    if store is None:
        if self.store_class:
            store = self.store_class(task=task_desc)
        else:
            store = FlexStore(task=task_desc)
    return await flow.arun(store)
```

No streaming passthrough — streaming is a Flow-layer concern. Users who need `on_token` drop to `team.graph` and call `flow.arun(store, on_token=cb)` directly.

---

## Group 3: Flow Engine

**Files**: `core.py`
**Dependencies**: Group 2 (async wrappers for testing fan-out + async loops)

### DESIGN-006: Fan-out → continuation with boundary enforcement

**Current behavior**: `_resolve_next` hits a fan-out wire, calls `_run_fanout`, returns `None` — flow ends.

**New behavior**: After fan-out completes and results merge, the engine **continues iterating** the remaining wires for the same source node.

Changes to `_resolve_next`:

```python
def _resolve_next(self, current: str, action: str, store: S) -> str | None:
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
            continue  # <-- was: return None
        return w.target

    return None
```

Same change to `_aresolve_next` (with `await self._arun_fanout`).

**Boundary enforcement**: The fan-out itself (`_run_fanout` / `_arun_fanout`) is synchronous from the caller's perspective — it blocks until all targets complete and merge. The `continue` statement only fires after merge is done. This is the boundary: all fan-out work completes before the next wire is evaluated.

**Wire ordering convention**:
```python
flow.wire("dispatch", ["a", "b", "c"])           # fan-out, fires first
flow.wire("dispatch", "aggregate", on="default")  # continuation after fan-out
```

Fan-out wires (list target) are processed first in iteration order. After completion, the next matching string-target wire becomes the continuation.

**Edge case**: Multiple fan-out wires on the same node run sequentially (each blocks before the next), maintaining barrier semantics.

### FEAT-005: Per-unit timeout

Add `timeout` attribute:
- `Unit.timeout: float | None = None`
- `AsyncUnit.timeout: float | None = None`

Add `default_timeout: float | None = None` parameter to `Flow.run()` and `Flow.arun()`.

Effective timeout resolution:

```python
effective = getattr(unit, 'timeout', None) or default_timeout
```

**Sync path** (`Flow.run`):

```python
if effective is not None:
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(unit.run, store)
        try:
            action = future.result(timeout=effective)
        except TimeoutError:
            raise  # will be caught by error routing
else:
    action = unit.run(store)
```

**Async path** (`Flow.arun`):

```python
if effective is not None:
    action = await asyncio.wait_for(unit.arun(store), timeout=effective)
else:
    action = await unit.arun(store)
```

If no timeout is configured (both `None`), no wrapping — zero overhead on the happy path. `TimeoutError` feeds into the error routing system (FEAT-010).

### FEAT-010: Error routing

Two-layer error handling:

**Wire-level** (`on="error"`):
Wires with `on="error"` match when the source unit raises an exception.

**Flow-level** (`on_error="handler_node"`):
Add `on_error: str | None = None` parameter to `Flow.__init__()`.

**Implementation in `run()`** (same pattern in `arun()`):

```python
while current and steps < max_steps:
    steps += 1
    unit = self._units.get(current)
    if unit is None:
        raise RuntimeError(f"Unknown unit '{current}'")

    start = time.monotonic()
    try:
        # timeout wrapping here (FEAT-005)
        action = unit.run(store)
    except Exception as exc:
        end = time.monotonic()
        object.__setattr__(store, '_last_error', exc)
        self._trace.append({
            "step": steps,
            "unit": current,
            "action": "error",
            "duration_ms": round((end - start) * 1000, 2),
            "error": str(exc),
            "state_snapshot": store.model_dump(),
        })
        # Wire-level error routing
        error_target = self._resolve_error_target(current)
        if error_target:
            current = error_target
            continue
        # Flow-level fallback
        if self._on_error and self._on_error in self._units:
            current = self._on_error
            continue
        raise  # no error routing configured — preserve current behavior
    else:
        end = time.monotonic()
        self._trace.append({...})  # normal trace entry

    next_node = self._resolve_next(current, action, store)
    current = next_node
```

Helper method:

```python
def _resolve_error_target(self, current: str) -> str | None:
    """Find a wire with on='error' from the current node."""
    for w in self._wires.get(current, []):
        if w.on == "error":
            return w.target if isinstance(w.target, str) else None
    return None
```

The error handler node receives a store with `_last_error` set (via `object.__setattr__` to bypass Pydantic's `extra="forbid"`). Its `post()` return value drives normal wire routing — the flow can continue or end as usual.

**`_last_error` lifecycle**: Set on error, cleared when the error handler node runs successfully (at the start of the next iteration, before `unit.run()`). Add `object.__setattr__(store, '_last_error', None)` at the top of the loop.

---

## Group 4: Harness

**Files**: `harness.py`
**Dependencies**: Group 3 (fan-out continuation)

### BUG-001: Hierarchical strategy — parallel workers

Replace the sequential wiring in `_build_hierarchical`:

```python
def _build_hierarchical(self, task_desc: str = "") -> Flow:
    flow = Flow(reducers=self._reducers)

    mgr = self.manager or Agent(
        "Manager", "Decompose and delegate tasks",
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

    # Continuation: after fan-out completes and merges → synthesize
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

Workers run in parallel via fan-out, results merge via reducers, then continuation wire routes to `synthesize`. The individual `flow.wire(agent.name, "synthesize")` lines are removed — the continuation wire handles the transition.

---

## Testing Strategy

Each group gets its own test additions in `tests/test_flowforge.py` or dedicated test files:

**Group 1**:
- `test_wire_validation_catches_typo` — wire to nonexistent node, assert `ValueError` at `run()`
- `test_wire_validation_allows_late_add` — wire first, add node later, run succeeds
- `test_max_steps_warning` — flow that loops, assert warning logged
- `test_max_steps_raises_when_opted_in` — assert `FlowExhaustedError`
- `test_trace_has_duration` — run flow, check `duration_ms` in trace entries

**Group 2**:
- `test_async_loop_with_async_evaluator` — `flow.loop()` with `AsyncUnit` evaluator, run via `arun()`
- `test_team_arun` — `Team.arun()` basic execution

**Group 3**:
- `test_fanout_continuation` — fan-out → merge → next node runs
- `test_fanout_boundary` — verify downstream node sees merged state
- `test_unit_timeout_sync` — unit with timeout, exceeds it, `TimeoutError` raised
- `test_unit_timeout_async` — same, async path
- `test_default_timeout_flow` — flow-level default applies to units without explicit timeout
- `test_error_routing_wire_level` — unit raises, routes to error handler via `on="error"` wire
- `test_error_routing_flow_level` — no wire-level route, falls back to `on_error` handler
- `test_error_routing_preserves_reraise` — no routing configured, original exception propagates
- `test_last_error_on_store` — error handler can read `_last_error` from store

**Group 4**:
- `test_hierarchical_parallel_execution` — workers run in parallel (timing assertion), synthesize sees all outputs

---

## Exports

New public API additions to `__init__.py`:
- `FlowExhaustedError`

No other new public types. `_AsyncCountingWrapper`, `_resolve_error_target`, `_validate_graph` are all private.

---

## Files Changed

| File | Changes |
|------|---------|
| `src/flowforge/core.py` | Groups 1–3: validation, retry cleanup, timing, async wrappers, fan-out continuation, timeout, error routing |
| `src/flowforge/harness.py` | Groups 2+4: `Team.arun()`, hierarchical fix |
| `src/flowforge/__init__.py` | Export `FlowExhaustedError` |
| `tests/test_flowforge.py` | Group 1+4 tests |
| `tests/test_async.py` | Group 2+3 async tests |
