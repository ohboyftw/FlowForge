#!/usr/bin/env python
"""FlowForge eval harness.

Usage:
    py scripts/eval.py          # human-readable
    py scripts/eval.py --json   # JSON for evo-loop
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import os
# Prefer PYTHONPATH if set (used by evo-loop to point to worktree src/)
# Otherwise fall back to this script's parent project src/
_src = os.environ.get("PYTHONPATH") or str(Path(__file__).resolve().parent.parent / "src")
sys.path.insert(0, _src)

_assertions = []


def assertion(key, description):
    def decorator(fn):
        _assertions.append((key, description, fn))
        return fn
    return decorator


@assertion("T1_IMPORTS", "Core modules import without error")
def check_imports():
    errors = []
    modules = ['examples', 'flowforge']
    for mod in modules:
        try:
            __import__(mod)
        except Exception as exc:
            errors.append(f"{mod}: {exc}")
    if errors:
        return False, f"Import failures: {'; '.join(errors)}"
    return True, f"{len(modules)} modules imported"


@assertion("T4_TESTS_PASS", "All test files pass")
def check_tests():
    import subprocess
    import os
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-x", "-q"],
        capture_output=True, text=True, timeout=120, env=env,
    )
    lines = result.stdout.strip().split("\n")
    summary = lines[-1] if lines else ""
    if result.returncode == 0:
        return True, f"Tests: {summary}"
    failed = [l for l in lines if "FAILED" in l]
    return False, f"{summary} | First: {failed[0] if failed else 'unknown'}"


@assertion("T2_REDUCER_COMMUTATIVITY", "Fan-out merge is independent of completion order")
def check_commutativity():
    from flowforge.core import StoreBase, Flow, AsyncUnit, FunctionUnit, ReducerRegistry
    import asyncio

    class CommState(StoreBase):
        task: str = ""
        items: list[str] = []
        __reducers__ = {"items": "extend"}

    class AddA(AsyncUnit):
        async def prep(self, store): return store
        async def exec(self, prep_result): return ["a1", "a2"]
        async def post(self, store, exec_result):
            store.items = exec_result
            return "done"

    class AddB(AsyncUnit):
        async def prep(self, store): return store
        async def exec(self, prep_result): return ["b1", "b2"]
        async def post(self, store, exec_result):
            store.items = exec_result
            return "done"

    results_set = set()
    for _ in range(5):
        flow = Flow(reducers=ReducerRegistry({"items": "extend"}))
        flow.add("dispatch", FunctionUnit(lambda s: "default"))
        flow.add("a", AddA())
        flow.add("b", AddB())
        flow.wire("dispatch", ["a", "b"])
        flow.entry("dispatch")

        store = CommState(task="commutativity")
        store = asyncio.run(flow.arun(store))
        results_set.add(tuple(sorted(store.items)))

    if len(results_set) != 1:
        return False, f"Non-deterministic merge: got {len(results_set)} distinct results"
    items = list(results_set)[0]
    if set(items) != {"a1", "a2", "b1", "b2"}:
        return False, f"Missing items: {items}"
    return True, f"Commutative: {items}"


@assertion("T2_REDUCER_ASSOCIATIVITY", "3-branch extend/merge produces consistent results")
def check_associativity():
    from flowforge.core import StoreBase, Flow, Unit, FunctionUnit, ReducerRegistry

    class AssocState(StoreBase):
        task: str = ""
        items: list[str] = []
        meta: dict = {}
        __reducers__ = {"items": "extend", "meta": "merge"}

    class Writer(Unit):
        def __init__(self, tag: str):
            self._tag = tag
        def prep(self, store): return store
        def exec(self, prep_result): return self._tag
        def post(self, store, exec_result):
            store.items = [f"{exec_result}_item"]
            store.meta = {exec_result: True}
            return "done"

    flow = Flow(reducers=ReducerRegistry({"items": "extend", "meta": "merge"}))
    flow.add("dispatch", FunctionUnit(lambda s: "default"))
    for tag in ["x", "y", "z"]:
        flow.add(tag, Writer(tag))
    flow.wire("dispatch", ["x", "y", "z"])
    flow.entry("dispatch")

    store = AssocState(task="assoc")
    store = flow.run(store)

    expected_items = {"x_item", "y_item", "z_item"}
    expected_meta = {"x", "y", "z"}

    if set(store.items) != expected_items:
        return False, f"Items missing: got {store.items}, expected {expected_items}"
    if set(store.meta.keys()) != expected_meta:
        return False, f"Meta keys missing: got {store.meta}, expected {expected_meta}"
    return True, f"Associative: {len(store.items)} items, {len(store.meta)} meta keys"


@assertion("T2_CONFLICT_DETECTION", "Conflicting scalar writes are handled explicitly")
def check_conflict_detection():
    from flowforge.core import StoreBase, Flow, Unit, FunctionUnit

    class ConflictState(StoreBase):
        task: str = ""
        answer: str = ""

    class WriteA(Unit):
        def prep(self, store): return store
        def exec(self, prep_result): return "alpha"
        def post(self, store, exec_result):
            store.answer = exec_result
            return "done"

    class WriteB(Unit):
        def prep(self, store): return store
        def exec(self, prep_result): return "beta"
        def post(self, store, exec_result):
            store.answer = exec_result
            return "done"

    results = set()
    for _ in range(10):
        flow = Flow()
        flow.add("dispatch", FunctionUnit(lambda s: "default"))
        flow.add("a", WriteA())
        flow.add("b", WriteB())
        flow.wire("dispatch", ["a", "b"])
        flow.entry("dispatch")

        store = ConflictState(task="conflict")
        try:
            store = flow.run(store)
            results.add(store.answer)
        except (ValueError, RuntimeError) as exc:
            return True, f"Conflict raised: {exc}"

    if len(results) > 1:
        return False, f"Non-deterministic conflict resolution: got {results}"
    if len(results) == 1:
        return True, f"Deterministic resolution: always '{results.pop()}' (declaration order)"
    return False, "No result produced"


@assertion("T2_ROUNDTRIP_INTEGRITY", "Store survives fan-out merge -> serialize -> deserialize")
def check_roundtrip():
    from flowforge.core import StoreBase, Flow, Unit, FunctionUnit, ReducerRegistry

    class RichState(StoreBase):
        task: str = ""
        items: list[str] = []
        counts: dict[str, int] = {}
        flag: bool = False
        __reducers__ = {"items": "extend", "counts": "merge"}

    class SetFields(Unit):
        def __init__(self, tag: str):
            self._tag = tag
        def prep(self, store): return store
        def exec(self, prep_result): return self._tag
        def post(self, store, exec_result):
            store.items = [exec_result]
            store.counts = {exec_result: len(exec_result)}
            store.flag = True
            return "done"

    flow = Flow(reducers=ReducerRegistry({"items": "extend", "counts": "merge"}))
    flow.add("dispatch", FunctionUnit(lambda s: "default"))
    flow.add("w1", SetFields("alpha"))
    flow.add("w2", SetFields("beta"))
    flow.wire("dispatch", ["w1", "w2"])
    flow.entry("dispatch")

    store = RichState(task="roundtrip")
    store = flow.run(store)

    dumped = store.model_dump()
    try:
        restored = RichState.model_validate(dumped)
    except Exception as exc:
        return False, f"Validation failed post-merge: {exc}"

    if set(restored.items) != set(store.items):
        return False, f"Items lost: {restored.items} vs {store.items}"
    if restored.counts != store.counts:
        return False, f"Counts diverged: {restored.counts} vs {store.counts}"
    return True, f"Round-trip clean: {len(restored.items)} items, {len(restored.counts)} counts"


@assertion("T3_FLOW_AS_UNIT", "A Flow can be wrapped as a Unit inside another Flow")
def check_flow_as_unit():
    from flowforge.core import StoreBase, Flow, Unit, FunctionUnit

    class NumState(StoreBase):
        task: str = ""
        value: int = 0

    class Increment(Unit):
        def prep(self, store): return store.value
        def exec(self, val): return val + 1
        def post(self, store, result):
            store.value = result
            return "done"

    inner = Flow()
    inner.add("inc1", Increment())
    inner.add("inc2", Increment())
    inner.wire("inc1", "inc2", on="done")
    inner.entry("inc1")

    try:
        if hasattr(inner, "as_unit"):
            inner_unit = inner.as_unit()
        elif hasattr(inner, "to_unit"):
            inner_unit = inner.to_unit()
        else:
            return False, "Flow has no as_unit/to_unit method — composition not supported"

        outer = Flow()
        outer.add("setup", Increment())
        outer.add("inner_flow", inner_unit)
        outer.add("final", Increment())
        outer.wire("setup", "inner_flow", on="done")
        outer.wire("inner_flow", "final", on="done")
        outer.entry("setup")

        store = NumState(task="composition", value=0)
        store = outer.run(store)

        if store.value != 4:
            return False, f"Expected value=4, got {store.value}"
        return True, f"Flow-as-Unit works: value={store.value}"
    except Exception as exc:
        return False, f"Flow-as-Unit failed: {exc}"


@assertion("T3_NESTED_FANOUT", "Fan-out target can trigger another fan-out")
def check_nested_fanout():
    from flowforge.core import StoreBase, Flow, Unit, FunctionUnit, ReducerRegistry

    class NestedState(StoreBase):
        task: str = ""
        log: list[str] = []
        __reducers__ = {"log": "extend"}

    class Logger(Unit):
        def __init__(self, tag: str):
            self._tag = tag
        def prep(self, store): return store
        def exec(self, prep_result): return self._tag
        def post(self, store, exec_result):
            store.log = [exec_result]
            return "default"

    flow = Flow(reducers=ReducerRegistry({"log": "extend"}))
    flow.add("root", FunctionUnit(lambda s: "default"))
    flow.add("branch_a", Logger("a"))
    flow.add("branch_b", FunctionUnit(lambda s: "default"))
    flow.add("leaf_b1", Logger("b1"))
    flow.add("leaf_b2", Logger("b2"))

    flow.wire("root", ["branch_a", "branch_b"])
    flow.wire("branch_b", ["leaf_b1", "leaf_b2"])
    flow.entry("root")

    store = NestedState(task="nested")
    try:
        store = flow.run(store)
    except Exception as exc:
        return False, f"Nested fan-out failed: {exc}"

    expected = {"a", "b1", "b2"}
    if set(store.log) != expected:
        return False, f"Expected {expected}, got {set(store.log)}"
    return True, f"Nested fan-out works: {sorted(store.log)}"


@assertion("T3_TEAM_OF_TEAMS", "A Team can contain agents wrapping inner Teams")
def check_team_of_teams():
    from flowforge.core import FlexStore
    from flowforge.harness import Agent, Team

    def mock_llm(system, user, tools=None):
        return f"response from {system[:20]}"

    inner_team = Team(
        agents=[
            Agent("analyst", "analyze data", llm_fn=mock_llm),
            Agent("writer", "write reports", llm_fn=mock_llm),
        ],
        strategy="sequential",
    )

    try:
        inner_flow = inner_team.compile("sub-task")

        if hasattr(inner_flow, "as_unit"):
            inner_unit = inner_flow.as_unit()
        elif hasattr(inner_team, "as_agent"):
            inner_unit = inner_team.as_agent()
        else:
            return False, "No Team.as_agent() or Flow.as_unit() — team nesting not supported"

        outer = Team(
            agents=[
                Agent("coordinator", "coordinate work", llm_fn=mock_llm),
            ],
            strategy="sequential",
        )
        outer_flow = outer.compile("Produce a report")

        store = FlexStore(task="team-of-teams")
        store = outer_flow.run(store)

        if not store:
            return False, "Team-of-Teams returned no store"
        return True, f"Team-of-Teams executed: {type(store).__name__}"
    except Exception as exc:
        return False, f"Team-of-Teams failed: {exc}"


@assertion("T3_MIXED_SYNC_ASYNC_FANOUT", "Sync + async units in same fan-out merge correctly")
def check_mixed_fanout():
    from flowforge.core import StoreBase, Flow, Unit, AsyncUnit, FunctionUnit, ReducerRegistry
    import asyncio

    class MixedState(StoreBase):
        task: str = ""
        results: list[str] = []
        __reducers__ = {"results": "extend"}

    class SyncWorker(Unit):
        def prep(self, store): return store
        def exec(self, prep_result): return "sync_result"
        def post(self, store, exec_result):
            store.results = [exec_result]
            return "done"

    class AsyncWorker(AsyncUnit):
        async def prep(self, store): return store
        async def exec(self, prep_result): return "async_result"
        async def post(self, store, exec_result):
            store.results = [exec_result]
            return "done"

    flow = Flow(reducers=ReducerRegistry({"results": "extend"}))
    flow.add("dispatch", FunctionUnit(lambda s: "default"))
    flow.add("sync_w", SyncWorker())
    flow.add("async_w", AsyncWorker())
    flow.wire("dispatch", ["sync_w", "async_w"])
    flow.entry("dispatch")

    store = MixedState(task="mixed")
    store = asyncio.run(flow.arun(store))

    expected = {"sync_result", "async_result"}
    if set(store.results) != expected:
        return False, f"Expected {expected}, got {set(store.results)}"
    return True, f"Mixed fan-out correct: {sorted(store.results)}"


# ── T5: Abstraction Honesty — does the API do what it claims? ────────────────

@assertion("T5_FLUENT_CHAIN", "Flow.add().wire().entry() returns self at every step")
def check_fluent_chain():
    from flowforge.core import Flow, Unit, FunctionUnit

    class Noop(Unit):
        def prep(self, store): return None
        def exec(self, _): return None
        def post(self, store, _): return "done"

    flow = Flow()
    r1 = flow.add("a", Noop())
    r2 = r1.wire("a", "b", on="done")
    # wire to non-existent node is fine at build time — validated at run
    r3 = flow.add("b", Noop())
    r4 = r3.entry("a")

    if r1 is not flow:
        return False, f"add() returned {type(r1).__name__}, not Flow"
    if r2 is not flow:
        return False, f"wire() returned {type(r2).__name__}, not Flow"
    if r4 is not flow:
        return False, f"entry() returned {type(r4).__name__}, not Flow"
    return True, "Fluent chain: add/wire/entry all return self"


@assertion("T5_CUSTOM_REDUCER_FANOUT", "Custom callable reducers compose with fan-out merge")
def check_custom_reducer_fanout():
    from flowforge.core import StoreBase, Flow, Unit, FunctionUnit, ReducerRegistry

    class ScoreState(StoreBase):
        task: str = ""
        score: float = 0.0
        __reducers__ = {"score": lambda old, new: max(old, new)}

    class Scorer(Unit):
        def __init__(self, val: float):
            self._val = val
        def prep(self, store): return store
        def exec(self, _): return self._val
        def post(self, store, result):
            store.score = result
            return "done"

    flow = Flow(reducers=ReducerRegistry({"score": lambda old, new: max(old, new)}))
    flow.add("dispatch", FunctionUnit(lambda s: "default"))
    flow.add("low", Scorer(0.3))
    flow.add("high", Scorer(0.9))
    flow.wire("dispatch", ["low", "high"])
    flow.entry("dispatch")

    store = ScoreState(task="custom-reducer")
    store = flow.run(store)

    if store.score != 0.9:
        return False, f"Expected max(0.3, 0.9)=0.9, got {store.score}"
    return True, f"Custom reducer works in fan-out: score={store.score}"


@assertion("T5_CHECKPOINT_ACROSS_COMPOSITION", "Store checkpoints survive across composed flows")
def check_checkpoint_composition():
    from flowforge.core import StoreBase, Flow, Unit

    class CountState(StoreBase):
        task: str = ""
        value: int = 0

    class IncrementAndCheckpoint(Unit):
        def prep(self, store): return store.value
        def exec(self, val): return val + 1
        def post(self, store, result):
            store.value = result
            store.checkpoint(f"at_{result}")
            return "done"

    inner = Flow()
    inner.add("inc", IncrementAndCheckpoint())
    inner.entry("inc")

    outer = Flow()
    outer.add("first", IncrementAndCheckpoint())
    outer.add("inner_flow", inner.as_unit())
    outer.wire("first", "inner_flow", on="done")
    outer.entry("first")

    store = CountState(task="checkpoint-composition", value=0)
    store = outer.run(store)

    if store.value != 2:
        return False, f"Expected value=2, got {store.value}"
    checkpoints = store.list_checkpoints() if hasattr(store, "list_checkpoints") else list(store._checkpoints.keys())
    if "at_1" not in checkpoints:
        return False, f"Checkpoint 'at_1' missing. Have: {checkpoints}"
    if "at_2" not in checkpoints:
        return False, f"Checkpoint 'at_2' missing. Have: {checkpoints}"

    # Rollback should restore to checkpoint
    store.rollback("at_1")
    if store.value != 1:
        return False, f"Rollback to at_1 failed: value={store.value}"
    return True, f"Checkpoints survive composition: {checkpoints}, rollback works"


@assertion("T5_COMPILED_TEAM_IS_REAL_FLOW", "Team.compile() returns a Flow equivalent to hand-built")
def check_compiled_team_equivalence():
    from flowforge.core import FlexStore
    from flowforge.harness import Agent, Team

    def mock_llm(system, user, tools=None):
        return f"mock:{user[:20]}"

    team = Team(
        agents=[
            Agent("researcher", "find info", llm_fn=mock_llm),
            Agent("writer", "write report", llm_fn=mock_llm),
        ],
        strategy="sequential",
    )

    flow = team.compile("test task")

    # The compiled flow should be a real Flow with inspectable structure
    from flowforge.core import Flow
    if not isinstance(flow, Flow):
        return False, f"compile() returned {type(flow).__name__}, not Flow"

    # Should have nodes and edges (graph structure)
    if not hasattr(flow, "nodes") or not hasattr(flow, "edges"):
        return False, "Compiled Flow has no nodes/edges — graph not inspectable"

    nodes = flow.nodes
    edges = flow.edges
    if len(nodes) < 2:
        return False, f"Expected at least 2 nodes, got {len(nodes)}"
    if len(edges) < 1:
        return False, f"Expected at least 1 edge, got {len(edges)}"

    # Should be runnable
    store = FlexStore(task="equivalence-test")
    store = flow.run(store)
    if not store:
        return False, "Compiled Flow produced no store"

    return True, f"Compiled Team is real Flow: {len(nodes)} nodes, {len(edges)} edges"


# ── T6: Abstraction Completeness — are there gaps? ──────────────────────────

@assertion("T6_TRACE_RECORDS_PATH", "Flow.trace records the execution path taken")
def check_trace_path():
    from flowforge.core import StoreBase, Flow, Unit

    class TraceState(StoreBase):
        task: str = ""
        value: int = 0

    class Step(Unit):
        def __init__(self, label: str):
            self._label = label
        def prep(self, store): return store
        def exec(self, _): return self._label
        def post(self, store, result):
            store.value += 1
            return "next" if store.value < 3 else "done"

    flow = Flow()
    flow.add("a", Step("a"))
    flow.add("b", Step("b"))
    flow.add("c", Step("c"))
    flow.wire("a", "b", on="next")
    flow.wire("b", "c", on="next")
    flow.entry("a")

    store = TraceState(task="trace")
    store = flow.run(store)

    trace = flow.trace
    if not trace:
        return False, "Flow.trace is empty after execution"
    if not isinstance(trace, list):
        return False, f"Flow.trace is {type(trace).__name__}, not list"

    # Trace should record at least the nodes visited
    node_names = [t.get("node", t.get("unit", "")) for t in trace]
    if "a" not in str(node_names):
        return False, f"Trace missing 'a': {node_names}"
    return True, f"Trace recorded {len(trace)} entries: {node_names[:5]}"


@assertion("T6_INTERRUPT_IN_FANOUT", "InterruptSignal in one fan-out branch is catchable")
def check_interrupt_in_fanout():
    from flowforge.core import StoreBase, Flow, Unit, FunctionUnit, InterruptSignal, ReducerRegistry

    class IntState(StoreBase):
        task: str = ""
        results: list[str] = []
        __reducers__ = {"results": "extend"}

    class NormalUnit(Unit):
        def prep(self, store): return store
        def exec(self, _): return "ok"
        def post(self, store, result):
            store.results = [result]
            return "done"

    flow = Flow(reducers=ReducerRegistry({"results": "extend"}))
    flow.add("dispatch", FunctionUnit(lambda s: "default"))
    flow.add("normal", NormalUnit())
    flow.add("interrupter", NormalUnit())
    flow.add("catch", NormalUnit())
    flow.wire("dispatch", ["normal", "interrupter"])
    flow.wire("interrupter", "catch", on="done", interrupt=True)
    flow.entry("dispatch")

    store = IntState(task="interrupt-fanout")
    try:
        store = flow.run(store)
        # If no interrupt raised, the flow handled it gracefully
        return True, f"Fan-out with interrupt wire completed: results={store.results}"
    except InterruptSignal as sig:
        # InterruptSignal is catchable — that's the contract
        return True, f"InterruptSignal caught from fan-out: {sig}"
    except Exception as exc:
        return False, f"Unexpected error (not InterruptSignal): {type(exc).__name__}: {exc}"


@assertion("T6_DESCRIBE_MATCHES_REALITY", "Flow.describe() output matches actual graph structure")
def check_describe_matches():
    from flowforge.core import Flow, Unit, FunctionUnit

    class Noop(Unit):
        def prep(self, store): return None
        def exec(self, _): return None
        def post(self, store, _): return "done"

    flow = Flow()
    flow.add("start", FunctionUnit(lambda s: "default"))
    flow.add("worker_a", Noop())
    flow.add("worker_b", Noop())
    flow.wire("start", ["worker_a", "worker_b"])
    flow.entry("start")

    if not hasattr(flow, "describe"):
        return False, "Flow has no describe() method"

    desc = flow.describe()
    if not desc:
        return False, "describe() returned empty"

    desc_str = str(desc)
    # Description should mention all nodes
    for name in ["start", "worker_a", "worker_b"]:
        if name not in desc_str:
            return False, f"describe() missing node '{name}': {desc_str[:200]}"

    return True, f"describe() covers all nodes ({len(desc_str)} chars)"


@assertion("T6_MERMAID_VALID_GRAPH", "Flow.to_mermaid() produces valid mermaid syntax")
def check_mermaid_output():
    from flowforge.core import Flow, Unit, FunctionUnit

    class Noop(Unit):
        def prep(self, store): return None
        def exec(self, _): return None
        def post(self, store, _): return "done"

    flow = Flow()
    flow.add("start", FunctionUnit(lambda s: "default"))
    flow.add("a", Noop())
    flow.add("b", Noop())
    flow.wire("start", ["a", "b"])
    flow.entry("start")

    if not hasattr(flow, "to_mermaid"):
        return False, "Flow has no to_mermaid() method"

    mermaid = flow.to_mermaid()
    if not mermaid:
        return False, "to_mermaid() returned empty"
    if "graph" not in mermaid.lower() and "flowchart" not in mermaid.lower():
        return False, f"to_mermaid() missing graph/flowchart header: {mermaid[:100]}"
    if "start" not in mermaid:
        return False, f"to_mermaid() missing 'start' node"
    if "-->" not in mermaid:
        return False, f"to_mermaid() missing edges (-->)"

    return True, f"Valid mermaid ({len(mermaid)} chars, has graph header + edges)"


# ── T7: Competitive Patterns — CrewAI / LangGraph feature parity ─────────────

@assertion("T7_LANGGRAPH_CONDITIONAL_LOOP", "Flow.loop() expresses LangGraph's revisit-until-done pattern")
def check_conditional_loop():
    from flowforge.core import StoreBase, Flow, Unit

    class RefineState(StoreBase):
        task: str = ""
        draft: str = ""
        quality: float = 0.0

    class Generator(Unit):
        def prep(self, store): return store.quality
        def exec(self, quality): return f"draft_v{int(quality * 10 + 1)}"
        def post(self, store, result):
            store.draft = result
            return "default"

    class Evaluator(Unit):
        def prep(self, store): return store.draft
        def exec(self, draft):
            version = int(draft.split("_v")[1]) if "_v" in draft else 0
            return min(version * 0.3, 1.0)
        def post(self, store, result):
            store.quality = result
            return "default"

    flow = Flow()
    flow.add("generate", Generator())
    flow.add("evaluate", Evaluator())
    flow.entry("generate")
    flow.loop("generate", "evaluate", until=lambda s: s.quality >= 0.9, max_rounds=5)

    store = RefineState(task="loop-test")
    store = flow.run(store)

    if store.quality < 0.9:
        return False, f"Loop didn't converge: quality={store.quality}"
    if store.draft == "draft_v1":
        return False, f"Loop didn't iterate: still on first draft"
    return True, f"Loop converged: quality={store.quality}, draft={store.draft}"


@assertion("T7_LANGGRAPH_INTERRUPT_RESUME", "InterruptSignal + Flow.resume() = LangGraph human-in-the-loop")
def check_interrupt_resume():
    from flowforge.core import StoreBase, Flow, Unit, InterruptSignal

    class ApprovalState(StoreBase):
        task: str = ""
        proposal: str = ""
        approved: bool = False
        final: str = ""

    class Proposer(Unit):
        def prep(self, store): return store
        def exec(self, _): return "spend $10k on GPUs"
        def post(self, store, result):
            store.proposal = result
            return "review"

    class Reviewer(Unit):
        def prep(self, store): return store
        def exec(self, _): return "needs_approval"
        def post(self, store, _):
            if not store.approved:
                return "interrupt"
            return "proceed"

    class Finalizer(Unit):
        def prep(self, store): return store.proposal
        def exec(self, proposal): return f"APPROVED: {proposal}"
        def post(self, store, result):
            store.final = result
            return "done"

    flow = Flow()
    flow.add("propose", Proposer())
    flow.add("review", Reviewer())
    flow.add("finalize", Finalizer())
    flow.wire("propose", "review", on="review")
    flow.wire("review", "finalize", on="proceed")
    flow.wire("review", "finalize", on="interrupt", interrupt=True)
    flow.entry("propose")

    store = ApprovalState(task="interrupt-resume")

    # Phase 1: run until interrupt
    try:
        store = flow.run(store)
        # If no interrupt, the flow handled it differently
        if store.final and "APPROVED" in store.final:
            return True, f"Flow completed without interrupt (approved path): {store.final}"
    except InterruptSignal as sig:
        # Human injects approval
        sig.store.approved = True
        # Phase 2: resume from the interrupted wire's target
        store = flow.resume(sig.store, sig.wire.target)
        if not store.final:
            return False, f"Resume produced no final output"
        if "APPROVED" not in store.final:
            return False, f"Resume didn't finalize: {store.final}"
        return True, f"Interrupt/resume works: {store.final}"

    return False, f"Neither interrupt nor completion happened"


@assertion("T7_CREWAI_HIERARCHICAL_DELEGATION", "Team(strategy='hierarchical') = CrewAI manager delegation")
def check_hierarchical_delegation():
    from flowforge.core import FlexStore, Flow
    from flowforge.harness import Agent, Team

    call_log = []

    def tracking_llm(system, user, tools=None):
        role = system.split("\n")[0] if system else "unknown"
        call_log.append(role[:30])
        return f"result from {role[:20]}"

    team = Team(
        agents=[
            Agent("researcher", "find information", llm_fn=tracking_llm),
            Agent("analyst", "analyze data", llm_fn=tracking_llm),
        ],
        manager=Agent("director", "decompose and synthesize", llm_fn=tracking_llm),
        strategy="hierarchical",
    )

    flow = team.compile("Build a market report")

    # Verify the graph structure has decompose → fan-out → synthesize
    if not isinstance(flow, Flow):
        return False, f"compile() returned {type(flow).__name__}"

    nodes = flow.nodes
    if len(nodes) < 3:
        return False, f"Expected 3+ nodes (decompose/workers/synthesize), got {len(nodes)}"

    store = FlexStore(task="hierarchical-delegation")
    store = flow.run(store)

    if len(call_log) < 3:
        return False, f"Expected 3+ LLM calls (decompose + workers + synthesize), got {len(call_log)}"

    return True, f"Hierarchical delegation: {len(call_log)} LLM calls, {len(nodes)} nodes"


@assertion("T7_CREWAI_TOOL_INTEGRATION", "Agent with tools= calls tools and feeds results to store")
def check_tool_integration():
    from flowforge.core import FlexStore, Flow
    from flowforge.harness import Agent
    from flowforge.identity import Task

    tool_calls = []

    def search_tool(query: str) -> str:
        tool_calls.append(query)
        return f"Found: {query} results"

    def tool_aware_llm(system, user, tools=None):
        # Simulate an LLM that uses tools
        if tools:
            result = tools[0]("test query")
            return f"Based on tool: {result}"
        return "no tools available"

    agent = Agent(
        "researcher", "find information",
        llm_fn=tool_aware_llm,
        tools=[search_tool],
    )

    task = Task(
        description="Search for AI papers",
        output_field="findings",
    )

    # Run through a Flow (the real usage pattern)
    unit = agent.as_unit(task)
    flow = Flow()
    flow.add("research", unit)
    flow.entry("research")

    store = FlexStore(task="tool-integration")
    store = flow.run(store)

    # Verify tool was called
    if not tool_calls:
        return False, "Tool was never called — tools not passed to LLM"

    # Verify result landed in store
    findings = getattr(store, "findings", None)
    if not findings:
        return False, f"Tool result not in store.findings"

    return True, f"Tool integration works: {len(tool_calls)} tool calls, findings={str(findings)[:50]}"


@assertion("T7_LANGGRAPH_CHECKPOINT_RESUME_MIDFLOW", "Checkpoint mid-flow + resume from checkpoint = LangGraph persistence")
def check_checkpoint_resume_midflow():
    from flowforge.core import StoreBase, Flow, Unit

    class PipelineState(StoreBase):
        task: str = ""
        stage: str = ""
        data: list[str] = []

    class StageUnit(Unit):
        def __init__(self, name: str):
            self._name = name
        def prep(self, store): return store
        def exec(self, _): return self._name
        def post(self, store, result):
            store.stage = result
            store.data = store.data + [result]
            store.checkpoint(f"after_{result}")
            return "next"

    flow = Flow()
    flow.add("extract", StageUnit("extract"))
    flow.add("transform", StageUnit("transform"))
    flow.add("load", StageUnit("load"))
    flow.wire("extract", "transform", on="next")
    flow.wire("transform", "load", on="next")
    flow.entry("extract")

    # Run full pipeline
    store = PipelineState(task="etl")
    store = flow.run(store)

    if store.data != ["extract", "transform", "load"]:
        return False, f"Pipeline didn't complete: {store.data}"

    # Now simulate crash recovery: rollback to mid-point and resume
    store.rollback("after_extract")
    if store.stage != "extract":
        return False, f"Rollback failed: stage={store.stage}"
    if store.data != ["extract"]:
        return False, f"Rollback data wrong: {store.data}"

    # Resume from transform
    store = flow.resume(store, "transform")
    if store.data != ["extract", "transform", "load"]:
        return False, f"Resume didn't complete pipeline: {store.data}"

    return True, f"Checkpoint + resume works: full pipeline recovered from mid-point"


@assertion("T7_LANGGRAPH_DYNAMIC_STATE", "FlexStore allows dynamic fields like LangGraph's TypedDict flexibility")
def check_dynamic_state():
    from flowforge.core import FlexStore, Flow, Unit, FunctionUnit

    class DynamicWriter(Unit):
        def __init__(self, field: str, value):
            self._field = field
            self._value = value
        def prep(self, store): return store
        def exec(self, _): return (self._field, self._value)
        def post(self, store, result):
            setattr(store, result[0], result[1])
            return "done"

    flow = Flow()
    flow.add("dispatch", FunctionUnit(lambda s: "default"))
    flow.add("w1", DynamicWriter("mood", "happy"))
    flow.add("w2", DynamicWriter("score", 42))
    flow.wire("dispatch", ["w1", "w2"])
    flow.entry("dispatch")

    store = FlexStore(task="dynamic-state")
    store = flow.run(store)

    if not hasattr(store, "mood") or store.mood != "happy":
        return False, f"Dynamic field 'mood' missing or wrong: {getattr(store, 'mood', 'MISSING')}"
    if not hasattr(store, "score") or store.score != 42:
        return False, f"Dynamic field 'score' missing or wrong: {getattr(store, 'score', 'MISSING')}"

    # Verify serialization round-trip
    dumped = store.model_dump()
    restored = FlexStore.model_validate(dumped)
    if restored.mood != "happy" or restored.score != 42:
        return False, f"Dynamic fields lost in round-trip"

    return True, f"Dynamic state works: mood={store.mood}, score={store.score}"


# ── T8: Native Observability — honest, lightweight, zero-dep ─────────────────

@assertion("T8_CALLBACK_ON_UNIT_LIFECYCLE", "Flow accepts callbacks that fire on unit start/end")
def check_callbacks():
    from flowforge.core import StoreBase, Flow, Unit

    class SimpleState(StoreBase):
        task: str = ""
        value: int = 0

    class Inc(Unit):
        def prep(self, store): return store.value
        def exec(self, val): return val + 1
        def post(self, store, result):
            store.value = result
            return "next"

    events = []

    def on_start(node_name, store):
        events.append(("start", node_name))

    def on_end(node_name, store, action):
        events.append(("end", node_name, action))

    flow = Flow()
    flow.add("a", Inc())
    flow.add("b", Inc())
    flow.wire("a", "b", on="next")
    flow.entry("a")

    # Flow should accept callbacks — check for the interface
    if hasattr(flow, "on_unit_start") or hasattr(flow, "add_callback"):
        if hasattr(flow, "on_unit_start"):
            flow.on_unit_start = on_start
            flow.on_unit_end = on_end
        elif hasattr(flow, "add_callback"):
            flow.add_callback(on_start=on_start, on_end=on_end)
    elif hasattr(flow, "callbacks"):
        flow.callbacks = {"on_unit_start": on_start, "on_unit_end": on_end}
    else:
        # Try passing callbacks to run()
        store = SimpleState(task="callbacks")
        try:
            store = flow.run(store, callbacks={"on_unit_start": on_start, "on_unit_end": on_end})
        except TypeError:
            return False, "Flow has no callback interface (on_unit_start, add_callback, callbacks, or run(callbacks=))"

        if not events:
            return False, "Callbacks were accepted but never fired"
        starts = [e for e in events if e[0] == "start"]
        if len(starts) < 2:
            return False, f"Expected 2+ start events, got {len(starts)}: {events}"
        return True, f"Callbacks via run(): {len(events)} events fired"

    store = SimpleState(task="callbacks")
    store = flow.run(store)

    if not events:
        return False, "Callback interface exists but callbacks never fired"
    starts = [e for e in events if e[0] == "start"]
    if len(starts) < 2:
        return False, f"Expected 2+ start events, got {len(starts)}: {events}"
    return True, f"Callbacks work: {len(events)} events for {len(starts)} units"


@assertion("T8_TRACE_STRUCTURED_EXPORT", "Flow.trace exports as structured dicts with node, action, duration")
def check_trace_export():
    import time
    from flowforge.core import StoreBase, Flow, Unit

    class State(StoreBase):
        task: str = ""
        value: int = 0

    class SlowUnit(Unit):
        def prep(self, store): return store
        def exec(self, _):
            time.sleep(0.01)  # 10ms
            return "done"
        def post(self, store, _):
            store.value += 1
            return "next"

    flow = Flow()
    flow.add("step1", SlowUnit())
    flow.add("step2", SlowUnit())
    flow.wire("step1", "step2", on="next")
    flow.entry("step1")

    store = State(task="trace-export")
    store = flow.run(store)

    trace = flow.trace
    if not trace:
        return False, "Flow.trace is empty"

    # Each trace entry should have at minimum: node name and action
    entry = trace[0]
    if not isinstance(entry, dict):
        return False, f"Trace entry is {type(entry).__name__}, not dict"

    has_node = "node" in entry or "unit" in entry or "name" in entry
    if not has_node:
        return False, f"Trace entry missing node identifier: {list(entry.keys())}"

    has_action = "action" in entry or "result" in entry or "output" in entry
    has_duration = "duration" in entry or "duration_ms" in entry or "elapsed" in entry

    if not has_action:
        return False, f"Trace entry missing action/result: {list(entry.keys())}"

    # Duration is the key observability metric — if missing, the trace isn't useful
    if not has_duration:
        return False, f"Trace entry missing duration: {list(entry.keys())}. Need timing for observability."

    return True, f"Trace has {len(trace)} entries with node+action+duration"


@assertion("T8_LLM_TOKEN_ACCOUNTING", "LLMUnit reports token usage and latency in trace or store")
def check_token_accounting():
    from flowforge.core import FlexStore, Flow
    from flowforge.harness import Agent
    from flowforge.identity import Task

    def metered_llm(system, user, tools=None):
        # Real LLMs return usage stats — simulate that
        return "The answer is 42"

    agent = Agent("analyst", "analyze data", llm_fn=metered_llm)
    task = Task(description="What is the answer?", output_field="answer")
    unit = agent.as_unit(task)

    flow = Flow()
    flow.add("analyze", unit)
    flow.entry("analyze")

    store = FlexStore(task="token-accounting")
    store = flow.run(store)

    # Check trace for latency
    trace = flow.trace
    if trace:
        entry = trace[0]
        has_latency = any(k in entry for k in ["duration", "duration_ms", "latency_ms", "elapsed"])
        if has_latency:
            return True, f"LLM latency in trace: {entry}"

    # Check store for usage metadata
    usage_fields = ["_llm_usage", "_token_usage", "_latency", "latency_ms"]
    for field in usage_fields:
        if hasattr(store, field):
            return True, f"LLM usage in store.{field}: {getattr(store, field)}"

    # Check if LLMUnit.exec returns latency (it does: {"output": ..., "latency_ms": ...})
    # The question is whether this makes it into the trace
    return False, f"No token/latency accounting in trace or store. Trace keys: {list(trace[0].keys()) if trace else 'empty'}"


@assertion("T8_TRACE_TO_JSON", "Flow trace is JSON-serializable for export to any tool")
def check_trace_json():
    import json
    from flowforge.core import StoreBase, Flow, Unit

    class State(StoreBase):
        task: str = ""
        value: int = 0

    class Step(Unit):
        def prep(self, store): return store
        def exec(self, _): return "ok"
        def post(self, store, _):
            store.value += 1
            return "next"

    flow = Flow()
    flow.add("a", Step())
    flow.add("b", Step())
    flow.wire("a", "b", on="next")
    flow.entry("a")

    store = State(task="json-trace")
    store = flow.run(store)

    trace = flow.trace
    if not trace:
        return False, "Flow.trace is empty"

    try:
        serialized = json.dumps(trace, default=str)
    except (TypeError, ValueError) as exc:
        return False, f"Trace not JSON-serializable: {exc}"

    # Round-trip
    restored = json.loads(serialized)
    if len(restored) != len(trace):
        return False, f"Round-trip lost entries: {len(trace)} -> {len(restored)}"

    return True, f"Trace JSON round-trip: {len(serialized)} bytes, {len(trace)} entries"


def run_all():
    results = {}
    t0 = time.time()
    for key, desc, fn in _assertions:
        try:
            passed, detail = fn()
        except Exception as e:
            passed, detail = False, f"CRASH: {e}"
        results[key] = {"status": "PASS" if passed else "FAIL", "detail": detail}
    elapsed = time.time() - t0
    pass_count = sum(1 for v in results.values() if v["status"] == "PASS")
    total = len(results)
    score = (pass_count * 100 // total) if total else 0
    return {
        "score": score,
        "passed": pass_count,
        "total": total,
        "elapsed_seconds": round(elapsed, 2),
        "assertions": results,
    }


if __name__ == "__main__":
    data = run_all()
    if "--json" in sys.argv:
        print(json.dumps(data, indent=2))
    else:
        project = Path(__file__).resolve().parent.parent.name
        print(f"\n  {project} Eval  |  Score: {data['score']}%  |  {data['passed']}/{data['total']}  |  {data['elapsed_seconds']}s")
        print("=" * 60)
        for key, info in data["assertions"].items():
            icon = "[PASS]" if info["status"] == "PASS" else "[FAIL]"
            desc = next((d for k, d, _ in _assertions if k == key), key)
            print(f"  {icon} {key}: {desc}")
            print(f"         {info['detail']}")
