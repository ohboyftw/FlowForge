"""
FlowForge — Unified Agent Orchestration Framework
Core module: Pydantic-typed Store + graph primitives

Design: Store is a Pydantic BaseModel subclass, giving us:
  - Typed fields with validation on every mutation
  - JSON serialization for checkpointing/persistence
  - Schema introspection for auto-documentation
  - Immutable snapshots via .model_copy(deep=True)
  - Field-level defaults and Optional[] for progressive state building
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    ClassVar,
    Generic,
    TypeVar,
)

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STORE — Pydantic-typed shared state with checkpointing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class StoreBase(BaseModel):
    """
    Base class for all FlowForge state stores.

    Subclass this with your typed fields:

        class ResearchState(StoreBase):
            query: str = ""
            findings: list[str] = []
            confidence: float = 0.0
            approved: bool = False

    The Store validates on every field set, checkpoints as JSON,
    and provides schema introspection for documentation/debugging.
    """

    model_config = ConfigDict(
        validate_assignment=True,  # validate on every s.field = val
        arbitrary_types_allowed=True,
        extra="forbid",  # catch typos — no surprise fields
    )

    # ── Reducer declarations for parallel fan-in ──
    # Usage: __reducers__ = {"findings": "extend", "score": lambda old, new: max(old, new)}
    __reducers__: ClassVar[dict[str, str | Callable]] = {}

    # ── Internal metadata (excluded from schema/serialization) ──
    _checkpoints: dict[str, dict] = {}
    _history: list[dict] = []
    _created_at: datetime = None

    def model_post_init(self, __context: Any) -> None:
        object.__setattr__(self, "_checkpoints", {})
        object.__setattr__(self, "_history", [])
        object.__setattr__(self, "_created_at", datetime.now(timezone.utc))

    # ── Checkpointing ──

    def checkpoint(self, name: str) -> None:
        """Snapshot current state. Serializes via Pydantic for safety."""
        snap = self.model_dump(mode="json")
        self._checkpoints[name] = snap
        self._history.append(
            {
                "action": "checkpoint",
                "name": name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def rollback(self, name: str) -> None:
        """Restore state from a named checkpoint."""
        if name not in self._checkpoints:
            raise KeyError(
                f"No checkpoint named '{name}'. Available: {list(self._checkpoints.keys())}"
            )
        snap = self._checkpoints[name]
        for field_name, value in snap.items():
            object.__setattr__(self, field_name, value)
        # Re-validate after rollback
        self.model_validate(self.model_dump())
        self._history.append(
            {
                "action": "rollback",
                "name": name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def list_checkpoints(self) -> list[str]:
        return list(self._checkpoints.keys())

    # ── Serialization helpers ──

    def to_json(self) -> str:
        """Full state as JSON string."""
        return self.model_dump_json(indent=2)

    def to_dict(self) -> dict:
        """Full state as dict (validated)."""
        return self.model_dump()

    @classmethod
    def from_json(cls, data: str) -> StoreBase:
        """Reconstruct from JSON with full validation."""
        return cls.model_validate_json(data)

    @classmethod
    def from_dict(cls, data: dict) -> StoreBase:
        """Reconstruct from dict with full validation."""
        return cls.model_validate(data)

    # ── Schema introspection ──

    @classmethod
    def describe_fields(cls) -> dict[str, str]:
        """Returns {field_name: type_annotation} for documentation."""
        return {name: str(info.annotation) for name, info in cls.model_fields.items()}

    # ── Diff support (useful for logging what changed between units) ──

    def diff(self, other: StoreBase) -> dict[str, tuple[Any, Any]]:
        """Compare two store states. Returns {field: (old, new)} for changed fields."""
        changes = {}
        for field_name in self.__class__.model_fields:
            old_val = getattr(other, field_name)
            new_val = getattr(self, field_name)
            if old_val != new_val:
                changes[field_name] = (old_val, new_val)
        return changes

    # ── Immutable snapshot (for passing to LLMs without mutation risk) ──

    def frozen_copy(self) -> StoreBase:
        """Deep copy for read-only consumption."""
        return self.model_copy(deep=True)


# Convenience: a flexible store for quick prototyping (allows any fields)
class FlexStore(BaseModel):
    """
    Untyped store for rapid prototyping. Use StoreBase subclass for production.

    Trades type safety for flexibility — fields are validated as Any.
    """

    model_config = ConfigDict(
        extra="allow",  # allow any field
        validate_assignment=True,
    )

    _checkpoints: dict[str, dict] = {}

    def model_post_init(self, __context: Any) -> None:
        object.__setattr__(self, "_checkpoints", {})

    def checkpoint(self, name: str) -> None:
        self._checkpoints[name] = self.model_dump(mode="json")

    def rollback(self, name: str) -> None:
        if name not in self._checkpoints:
            raise KeyError(f"No checkpoint '{name}'")
        for k, v in self._checkpoints[name].items():
            setattr(self, k, v)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STORE REDUCERS — For parallel fan-in state merging
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ReducerRegistry:
    """
    Defines how parallel unit outputs merge into shared state.

    Built-in strategies:
      - "replace"  : last write wins (default)
      - "extend"   : append lists
      - "merge"    : shallow dict merge
      - "append"   : wrap in list and extend
      - custom     : any (old_val, new_val) -> merged_val callable
    """

    _builtins: ClassVar[dict[str, Callable]] = {
        "replace": lambda old, new: new,
        "extend": lambda old, new: (old or []) + (new if isinstance(new, list) else [new]),
        "merge": lambda old, new: {**(old or {}), **(new or {})},
        "append": lambda old, new: (old or []) + [new],
    }

    def __init__(self, field_reducers: dict[str, str | Callable] = None):
        self._reducers: dict[str, Callable] = {}
        for field_name, reducer in (field_reducers or {}).items():
            self._register(field_name, reducer)

    def _register(self, field_name: str, reducer: str | Callable) -> None:
        if isinstance(reducer, str):
            if reducer not in self._builtins:
                raise ValueError(
                    f"Unknown reducer '{reducer}'. Available: {list(self._builtins.keys())}"
                )
            self._reducers[field_name] = self._builtins[reducer]
        else:
            self._reducers[field_name] = reducer

    @classmethod
    def from_store_class(cls, store_cls: type[StoreBase]) -> ReducerRegistry:
        """Build a ReducerRegistry from a StoreBase subclass's __reducers__ ClassVar."""
        reducers = getattr(store_cls, "__reducers__", {})
        return cls(reducers)

    def merge(self, other: ReducerRegistry) -> ReducerRegistry:
        """Return a new registry combining self and other (other wins on conflicts)."""
        combined = ReducerRegistry()
        combined._reducers = {**self._reducers, **other._reducers}
        return combined

    def reduce(self, store: StoreBase, field_name: str, old_val: Any, new_val: Any) -> Any:
        reducer = self._reducers.get(field_name, self._builtins["replace"])
        return reducer(old_val, new_val)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UNIT — Atomic computation node
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

S = TypeVar("S", bound=StoreBase)


class FlowExhaustedError(Exception):
    """Raised when Flow.run() hits the max_steps limit."""

    def __init__(self, steps: int, last_unit: str):
        self.steps = steps
        self.last_unit = last_unit
        super().__init__(f"Flow exhausted after {steps} steps at unit '{last_unit}'")


class Unit(Generic[S]):
    """
    Atomic computation with prep → exec → post lifecycle.

    Generic over Store type S for type-safe state access.

        class ResearchUnit(Unit[ResearchState]):
            def prep(self, store: ResearchState):
                return store.query  # IDE autocompletes!

            def exec(self, query: str) -> list[str]:
                return web_search(query)

            def post(self, store: ResearchState, results: list[str]) -> str:
                store.findings = results
                store.confidence = 0.8
                return "default"
    """

    max_retries: int = 0
    timeout: float | None = None

    def prep(self, store: S) -> Any:
        """Extract what you need from state. Return prep context."""
        return None

    def exec(self, prep_result: Any) -> Any:
        """Pure computation. No store access — keeps it testable."""
        return prep_result

    def exec_fallback(self, prep_result: Any, error: Exception) -> Any:
        """Called after max_retries exhausted. Default: re-raise."""
        raise error

    def post(self, store: S, exec_result: Any) -> str:
        """Write results to store. Return action label for routing."""
        return "default"

    def run(self, store: S) -> str:
        """Execute the full lifecycle. Returns action label."""
        p = self.prep(store)
        e = self._exec_with_retry(p)
        return self.post(store, e)

    def _exec_with_retry(self, prep_result: Any) -> Any:
        for attempt in range(self.max_retries + 1):
            try:
                return self.exec(prep_result)
            except Exception as exc:
                if attempt == self.max_retries:
                    return self.exec_fallback(prep_result, exc)


class AsyncUnit(Generic[S]):
    """
    Async variant of Unit. prep/post stay sync, only exec is async.

    For LLM network I/O where async is essential for concurrency.
    """

    max_retries: int = 0
    timeout: float | None = None

    def prep(self, store: S) -> Any:
        return None

    async def exec(self, prep_result: Any) -> Any:
        return prep_result

    async def exec_fallback(self, prep_result: Any, error: Exception) -> Any:
        raise error

    def post(self, store: S, exec_result: Any) -> str:
        return "default"

    async def arun(self, store: S) -> str:
        p = self.prep(store)
        e = await self._exec_with_retry(p)
        return self.post(store, e)

    async def _exec_with_retry(self, prep_result: Any) -> Any:
        for attempt in range(self.max_retries + 1):
            try:
                return await self.exec(prep_result)
            except Exception as exc:
                if attempt == self.max_retries:
                    return await self.exec_fallback(prep_result, exc)


class FunctionUnit(Unit[S]):
    """
    Wrap a plain function as a Unit. For quick prototyping.

        unit = FunctionUnit(lambda store: do_stuff(store))
    """

    def __init__(self, fn: Callable[[S], str | None]):
        self._fn = fn

    def run(self, store: S) -> str:
        result = self._fn(store)
        return result or "default"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WIRE — Typed edge with conditions, interrupts, fan-out
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class Wire:
    """
    Connection between Units in a Flow.

    Attributes:
        target:     Name of destination Unit (or list for fan-out)
        on:         Action label to match (from Unit.post return value)
        when:       Optional guard condition evaluated against Store
        interrupt:  If True, pause execution for human approval
        metadata:   Arbitrary data for debugging/tracing
    """

    target: str | list[str]
    on: str = "default"
    when: Callable[[StoreBase], bool] | None = None
    interrupt: bool = False
    metadata: dict = field(default_factory=dict)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FLOW — Directed graph of Units + Wires
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _merge_fanout_results(
    store: StoreBase,
    results: list[dict],
    original_values: dict,
    registry: ReducerRegistry,
) -> None:
    """Merge parallel fan-out results back into the original store using reducers."""
    # Skip internal fields
    skip_fields = {"_checkpoints", "_history", "_created_at"}

    # Collect all fields that changed in any result
    all_fields = set()
    for r in results:
        all_fields.update(r.keys())
    all_fields -= skip_fields

    for field_name in all_fields:
        orig_val = original_values.get(field_name)
        merged = orig_val
        for result_dict in results:
            if field_name not in result_dict:
                continue
            new_val = result_dict[field_name]
            if new_val != orig_val:
                merged = registry.reduce(store, field_name, merged, new_val)
        if merged != orig_val:
            setattr(store, field_name, merged)


class _CountingWrapper(Unit):
    """Wraps a Unit to count executions for loop() termination."""

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


class InterruptSignal(Exception):  # noqa: N818 — Signal, not Error
    """Raised when a Wire with interrupt=True is traversed."""

    def __init__(self, wire: Wire, store: StoreBase, from_node: str):
        self.wire = wire
        self.store = store
        self.from_node = from_node
        super().__init__(f"Interrupt at wire {from_node} → {wire.target}")


class Flow(Generic[S]):
    """
    Directed graph of Units connected by Wires.

    Supports:
      - Linear chains
      - Conditional branching
      - Fan-out (parallel dispatch to multiple targets)
      - Human-in-the-loop interrupts
      - Checkpointing at any point

    Usage:
        flow = Flow[ResearchState]()
        flow.add("search", SearchUnit())
        flow.add("analyze", AnalyzeUnit())
        flow.wire("search", "analyze")
        flow.entry("search")

        state = ResearchState(query="LLM agents")
        result = flow.run(state)
    """

    def __init__(self, reducers: ReducerRegistry = None, on_error: str | None = None):
        self._units: dict[str, Unit] = {}
        self._wires: dict[str, list[Wire]] = {}
        self._entry: str | None = None
        self._reducers = reducers or ReducerRegistry()
        self._trace: list[dict] = []
        self._on_error = on_error

    # ── Graph construction (fluent API) ──

    def add(self, name: str, unit: Unit) -> Flow[S]:
        """Register a Unit node."""
        self._units[name] = unit
        return self

    def wire(
        self,
        src: str,
        tgt: str | list[str],
        *,
        on: str = "default",
        when: Callable[[S], bool] = None,
        interrupt: bool = False,
    ) -> Flow[S]:
        """Connect src → tgt with optional conditions."""
        w = Wire(target=tgt, on=on, when=when, interrupt=interrupt)
        self._wires.setdefault(src, []).append(w)
        return self

    def entry(self, name: str) -> Flow[S]:
        """Set the entry point node. Validated lazily at run time."""
        self._entry = name
        return self

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
            raise ValueError("Invalid flow graph:\n  " + "\n  ".join(errors))

    # ── Execution ──

    def run(
        self,
        store: S,
        *,
        max_steps: int = 100,
        raise_on_exhaust: bool = False,
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
                object.__setattr__(store, "_last_error", None)
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

    def resume(self, store: S, from_node: str, **kwargs) -> S:
        """Resume after an interrupt. Continues from the target of the interrupted wire."""
        self._entry = from_node
        return self.run(store)

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
            try:
                effective_timeout = getattr(unit, "timeout", None) or default_timeout
                if isinstance(unit, AsyncUnit):
                    if on_token:
                        unit._on_token = on_token
                    if effective_timeout is not None:
                        action = await asyncio.wait_for(unit.arun(store), timeout=effective_timeout)
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
                                    store,
                                    field_name,
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
                object.__setattr__(store, "_last_error", None)
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

    def _build_fanout_registry(self, store: S) -> ReducerRegistry:
        """Build a merged reducer registry from store class + flow-level reducers."""
        store_registry = ReducerRegistry.from_store_class(store.__class__)
        return store_registry.merge(self._reducers)

    async def _arun_fanout(
        self, targets: list[str], store: S, on_token: Callable | None = None
    ) -> None:
        """Async fan-out using asyncio.gather with reducer-based merge."""
        units = [(t, self._units[t]) for t in targets if t in self._units]
        if not units:
            return

        registry = self._build_fanout_registry(store)
        original_values = store.model_dump()

        async def _run_one(unit, store_copy):
            if isinstance(unit, AsyncUnit):
                if on_token:
                    unit._on_token = on_token
                await unit.arun(store_copy)
            else:
                unit.run(store_copy)
            return store_copy.model_dump()

        copies = [(u, store.model_copy(deep=True)) for _name, u in units]
        results = await asyncio.gather(*[_run_one(u, c) for u, c in copies])

        _merge_fanout_results(store, results, original_values, registry)

    def loop(
        self,
        generator: str,
        evaluator: str,
        *,
        until: Callable[[S], bool],
        max_rounds: int = 5,
    ) -> Flow[S]:
        """
        Sugar for evaluator-optimizer loop pattern.

        Wires: generator -> evaluator (always),
               evaluator -> generator (when not until),
               evaluator -> end (when until or max_rounds).

        Tracks rounds via a `_loop_rounds` counter on the store.
        """
        round_counter = [0]

        def _check_continue(store: S) -> bool:
            return not until(store) and round_counter[0] < max_rounds

        def _check_done(store: S) -> bool:
            return until(store) or round_counter[0] >= max_rounds

        original_unit = self._units.get(evaluator)
        if original_unit is None:
            raise ValueError(f"Unknown unit '{evaluator}'")
        if isinstance(original_unit, AsyncUnit):
            self._units[evaluator] = _AsyncCountingWrapper(original_unit, round_counter)
        else:
            self._units[evaluator] = _CountingWrapper(original_unit, round_counter)

        self.wire(generator, evaluator)
        self.wire(evaluator, generator, on="default", when=_check_continue)
        # Terminal wire — a no-op FunctionUnit as the end sentinel
        end_name = f"_loop_end_{generator}_{evaluator}"
        self.add(end_name, FunctionUnit(lambda s: "default"))
        self.wire(evaluator, end_name, on="default", when=_check_done)
        return self

    def _resolve_next(self, current: str, action: str, store: S) -> str | None:
        """Resolve the next node based on wires, action label, and conditions."""
        wires = self._wires.get(current, [])

        for w in wires:
            # Match on action label
            if w.on != action and w.on != "*":
                continue
            # Check guard condition
            if w.when is not None and not w.when(store):
                continue
            # Interrupt check
            if w.interrupt:
                raise InterruptSignal(w, store, current)
            # Fan-out (parallel) — real concurrent execution
            if isinstance(w.target, list):
                self._run_fanout(w.target, store)
                continue  # fan-out done, check next wire for continuation
            return w.target

        return None  # no matching wire = end of flow

    def _resolve_error_target(self, current: str) -> str | None:
        """Find a wire with on='error' from the current node."""
        for w in self._wires.get(current, []):
            if w.on == "error":
                return w.target if isinstance(w.target, str) else None
        return None

    def _run_fanout(self, targets: list[str], store: S) -> None:
        """Run fan-out targets concurrently using ThreadPoolExecutor with reducer-based merge."""
        units = [(t, self._units[t]) for t in targets if t in self._units]
        if not units:
            return

        registry = self._build_fanout_registry(store)
        original_values = store.model_dump()

        def _run_on_copy(unit: Unit, store_copy: StoreBase) -> dict:
            unit.run(store_copy)
            return store_copy.model_dump()

        copies = [(u, store.model_copy(deep=True)) for _name, u in units]

        with ThreadPoolExecutor(max_workers=len(copies)) as pool:
            futures = [pool.submit(_run_on_copy, u, c) for u, c in copies]
            results = [f.result() for f in futures]

        _merge_fanout_results(store, results, original_values, registry)

    # ── Introspection ──

    @property
    def trace(self) -> list[dict]:
        """Execution trace from the last run."""
        return self._trace

    @property
    def nodes(self) -> list[str]:
        return list(self._units.keys())

    @property
    def edges(self) -> list[tuple[str, str, str]]:
        """Returns [(src, tgt, action_label), ...]"""
        result = []
        for src, wires in self._wires.items():
            for w in wires:
                targets = w.target if isinstance(w.target, list) else [w.target]
                for t in targets:
                    result.append((src, t, w.on))
        return result

    def to_mermaid(self) -> str:
        """Export flow as a Mermaid graph definition."""
        lines = ["graph TD"]
        if self._entry:
            lines.append(f"    style {self._entry} fill:#4CAF50,color:#fff")
        for src, tgt, label in self.edges:
            if label == "default":
                lines.append(f"    {src} --> {tgt}")
            else:
                lines.append(f"    {src} -->|{label}| {tgt}")
        return "\n".join(lines)

    def describe(self) -> str:
        """Human-readable graph description."""
        lines = [f"Flow: {len(self._units)} units, {len(self.edges)} edges"]
        lines.append(f"Entry: {self._entry}")
        for src, tgt, label in self.edges:
            lines.append(f"  {src} --[{label}]--> {tgt}")
        return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example: Typed state for a research → code pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ResearchCodeState(StoreBase):
    """Example typed state for a research-then-code pipeline."""

    task: str = ""
    research_query: str = ""
    findings: list[str] = Field(default_factory=list)
    code: str = ""
    review_notes: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    approved: bool = False
    attempts: int = 0


# Quick usage demonstration:
if __name__ == "__main__":
    # Create typed state
    state = ResearchCodeState(task="Build a CLI for stock analysis")

    # Validation works
    state.confidence = 0.85  # ✓ float
    # state.confidence = "high" # ✗ ValidationError!

    # Checkpoint
    state.findings = ["Found Yahoo Finance API", "Found Alpha Vantage"]
    state.checkpoint("after_research")

    # Mutate
    state.findings = []
    assert state.findings == []

    # Rollback
    state.rollback("after_research")
    assert len(state.findings) == 2

    # Serialization
    json_str = state.to_json()
    restored = ResearchCodeState.from_json(json_str)
    assert restored.findings == state.findings

    # Schema introspection
    print(ResearchCodeState.describe_fields())
    # {'task': 'str', 'research_query': 'str', 'findings': 'list[str]', ...}

    print("✓ All Store tests pass")
