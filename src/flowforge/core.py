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

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    ClassVar,
    Generic,
    TypeVar,
)

from pydantic import BaseModel, ConfigDict, Field

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
            if isinstance(reducer, str):
                if reducer not in self._builtins:
                    raise ValueError(
                        f"Unknown reducer '{reducer}'. Available: {list(self._builtins.keys())}"
                    )
                self._reducers[field_name] = self._builtins[reducer]
            else:
                self._reducers[field_name] = reducer

    def reduce(self, store: StoreBase, field_name: str, old_val: Any, new_val: Any) -> Any:
        reducer = self._reducers.get(field_name, self._builtins["replace"])
        return reducer(old_val, new_val)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UNIT — Atomic computation node
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

S = TypeVar("S", bound=StoreBase)


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

    def prep(self, store: S) -> Any:
        """Extract what you need from state. Return prep context."""
        return None

    def exec(self, prep_result: Any) -> Any:
        """Pure computation. No store access — keeps it testable."""
        return prep_result

    def post(self, store: S, exec_result: Any) -> str:
        """Write results to store. Return action label for routing."""
        return "default"

    def run(self, store: S) -> str:
        """Execute the full lifecycle. Returns action label."""
        p = self.prep(store)
        e = self.exec(p)
        return self.post(store, e)


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

    def __init__(self, reducers: ReducerRegistry = None):
        self._units: dict[str, Unit] = {}
        self._wires: dict[str, list[Wire]] = {}
        self._entry: str | None = None
        self._reducers = reducers or ReducerRegistry()
        self._trace: list[dict] = []

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
        """Set the entry point node."""
        if name not in self._units:
            raise ValueError(f"Unknown unit '{name}'")
        self._entry = name
        return self

    # ── Execution ──

    def run(self, store: S, *, max_steps: int = 100) -> S:
        """
        Execute the flow. Returns the final store state.

        Raises InterruptSignal if a human-in-the-loop wire is hit.
        Caller can catch it, inspect store, and resume.
        """
        if not self._entry:
            raise RuntimeError("No entry point set. Call flow.entry('name')")

        self._trace = []
        current = self._entry
        steps = 0

        while current and steps < max_steps:
            steps += 1
            unit = self._units.get(current)
            if unit is None:
                raise RuntimeError(f"Unknown unit '{current}'")

            # Execute unit
            action = unit.run(store)
            self._trace.append(
                {
                    "step": steps,
                    "unit": current,
                    "action": action,
                    "state_snapshot": store.to_dict() if hasattr(store, "to_dict") else str(store),
                }
            )

            # Find next node
            next_node = self._resolve_next(current, action, store)
            current = next_node

        return store

    def resume(self, store: S, from_node: str, **kwargs) -> S:
        """Resume after an interrupt. Continues from the target of the interrupted wire."""
        # Find the interrupted wire and continue to its target
        self._entry = from_node
        return self.run(store)

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
            # Fan-out (parallel) — run all targets sequentially for now
            if isinstance(w.target, list):
                for t in w.target:
                    if t in self._units:
                        self._units[t].run(store)
                return None  # fan-out terminates this branch
            return w.target

        return None  # no matching wire = end of flow

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
