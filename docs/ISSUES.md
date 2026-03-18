# FlowForge Issue & Feature Tracker

**Source:** Deep evaluation (2026-03-18) + Canopy integration analysis
**Priority tiers:** P0 (fix now), P1 (fix soon), P2 (backlog)

---

## Bugs

### BUG-001: Hierarchical strategy doesn't parallelize workers [P0]
**File:** `src/flowforge/harness.py` — `_build_hierarchical`
**Impact:** "Hierarchical" team is actually sequential with a manager
**Fix:** Wire workers as a list (fan-out), not individually
```python
# Current (sequential):
flow.wire("decompose", agent.name)
# Fix (parallel):
flow.wire("decompose", worker_names)  # list = parallel fan-out
```
**Canopy relevance:** Blocks future graph-level parallel spawning (decompose → [spawn_a, spawn_b] → aggregate)

---

### BUG-002: `_CountingWrapper` breaks with AsyncUnit [P1]
**File:** `src/flowforge/core.py`
**Impact:** Loops with async evaluators silently produce wrong behavior
**Fix:** Detect `AsyncUnit`, delegate to `arun()` instead of `run()`
**Canopy relevance:** Not used today, but would bite if retry-until-pass is modeled as a graph cycle

---

### BUG-003: `Task.output_field` defined but never used [P2]
**File:** `src/flowforge/identity.py`
**Impact:** Dead field — `LLMUnit.post()` reads `self.output_field` (Unit), never `task.output_field`
**Fix:** Remove from Task, or wire into LLMUnit
**Canopy relevance:** None (Canopy doesn't use identity layer)

---

### BUG-004: `Task.max_retries` defined but never wired [P2]
**File:** `src/flowforge/identity.py`
**Impact:** Users must set retries on Unit directly, making the Task field misleading
**Fix:** `LLMUnit.__init__` should copy `task.max_retries` to `self.max_retries`
**Canopy relevance:** None (Canopy doesn't use identity layer)

---

## Design Issues

### DESIGN-001: No tool invocation system [P2]
**Impact:** Critical for FlowForge as standalone — agents without tools are chat wrappers
**Scope:** `ToolDef` model, tool-call parsing in `LLMUnit.post()`, automatic re-prompting loop
**Canopy relevance:** None (Canopy agents are external processes, not FlowForge LLM agents)

---

### DESIGN-002: `_llm_post` always returns `"default"` [P2]
**File:** `src/flowforge/harness.py:47-52`
**Impact:** LLM output can never drive conditional wire routing
**Fix:** Parse action labels from LLM output (e.g., `[ACTION: escalate]`) or let users override `_llm_post`
**Canopy relevance:** None (Canopy Units return `"continue"` explicitly)

---

### DESIGN-003: No async `Team.run()` [P1]
**Impact:** Forces `asyncio.run()` wrapper or dropping to Flow layer for async
**Fix:** Add `Team.arun()` that calls `flow.arun()` — trivial
**Canopy relevance:** Low (Canopy uses Flow directly)

---

### DESIGN-004: Consensus strategy is misleadingly named [P2]
**File:** `src/flowforge/harness.py` — `_build_consensus`
**Impact:** Just "parallel + collect," not actual consensus
**Fix:** Rename to `parallel_collect`, or implement real voting/agreement
**Canopy relevance:** None

---

### DESIGN-005: No wire validation at construction time [P0]
**Impact:** `flow.wire("typo_node", "target")` silently succeeds, fails at runtime
**Fix:** Validate source/target exist in `flow.wire()`, raise `ValueError` immediately
**Canopy relevance:** Direct — `build_canopy_flow` uses 13 string stage names, typo would be silent breakage

---

### DESIGN-006: Fan-out can't chain to downstream node [P1]
**File:** `src/flowforge/core.py` — `_run_fanout`
**Impact:** `_run_fanout` always returns `None`, ending the branch. No fan-out → merge → continue.
**Fix:** Allow wiring a "join" node after parallel targets (map-reduce pattern)
**Canopy relevance:** Would enable modeling decompose → parallel spawns → aggregate as graph structure

---

### DESIGN-007: Silent `max_steps` cutoff [P0]
**Impact:** Flow hitting 100-step limit silently returns — no warning, no exception
**Fix:** `logger.warning()` or raise `FlowExhaustedError` when limit is hit
**Canopy relevance:** Direct — 13 stages per iteration is safe, but silent failure would be a debugging nightmare if pipeline grows

---

### DESIGN-008: `on_token` is a fragile dynamic attribute [P2]
**File:** `src/flowforge/core.py`
**Impact:** Set via `unit._on_token = callback` with no `__init__`, no type hint
**Fix:** Constructor parameter or property
**Canopy relevance:** None (Canopy doesn't stream tokens through FlowForge)

---

### DESIGN-009: `resume()` is underdeveloped [P2]
**File:** `src/flowforge/core.py:535`
**Impact:** Just resets entry + calls `run()`. `**kwargs` ignored. No state injection or wire skipping.
**Canopy relevance:** Would matter if `InterruptSignal` for trust gates is implemented (currently deferred)

---

### MINOR-001: Unreachable code in retry loop [P0]
**File:** `src/flowforge/core.py:295`
**Impact:** `raise last_error` after exhaustive loop is dead code
**Fix:** Remove
**Canopy relevance:** In `AsyncUnit._exec_with_retry` which every Canopy Unit inherits

---

### MINOR-002: Synthetic loop-end nodes pollute introspection [P2]
**Impact:** `_loop_end_gen_eval` nodes appear in `to_mermaid()` and `describe()`
**Canopy relevance:** None currently

---

### MINOR-003: FlexStore is second-class [P2]
**Impact:** Doesn't inherit StoreBase — missing `_history`, `to_dict()`, etc. Two parallel hierarchies.
**Canopy relevance:** None (Canopy uses StoreBase)

---

## Missing Features

| ID | Feature | Priority | Canopy Relevance |
|----|---------|----------|------------------|
| FEAT-001 | Tool system (definition, calling, execution loop) | P2 | None |
| FEAT-002 | Streaming for sync path | P2 | None |
| FEAT-003 | Memory / conversation history | P2 | None |
| FEAT-004 | Observability (logging, timing in trace) | P1 | Medium — trace timing would help debug slow Canopy stages |
| FEAT-005 | Timeout support (per-unit, global) | P1 | Medium — LLM calls in refine/decompose can hang |
| FEAT-006 | Team nesting (agent can be a team) | P2 | None |
| FEAT-007 | Cost/token tracking | P2 | Low — Canopy has its own MetricsCollector |
| FEAT-008 | Prompt templating | P2 | None |
| FEAT-009 | Caching | P2 | None |
| FEAT-010 | Error routing (dead-letter wire) | P1 | Medium — would replace try/except in every Unit.exec() |

---

## Test Gaps

| ID | Gap | Priority |
|----|-----|----------|
| TEST-001 | Empty flow, single-node flow, cyclic flow | P1 |
| TEST-002 | `Unit.prep()` or `Unit.post()` raising exceptions | P0 |
| TEST-003 | Rollback with non-existent checkpoint name | P1 |
| TEST-004 | Async loop (AsyncUnit as evaluator) | P0 (linked to BUG-002) |
| TEST-005 | FlexStore with parallel reducers | P2 |
| TEST-006 | `max_steps` limit being hit | P0 (linked to DESIGN-007) |
| TEST-007 | `exec_fallback` for AsyncUnit | P1 |

---

## Priority Summary

### P0 — Fix now (4 items)
- [ ] DESIGN-005: Wire validation at construction time
- [ ] DESIGN-007: Warn/raise on `max_steps` exhaustion
- [ ] MINOR-001: Remove unreachable code in retry loop
- [ ] BUG-001: Hierarchical parallel wiring fix

### P1 — Fix soon (6 items)
- [ ] BUG-002: `_CountingWrapper` async support
- [ ] DESIGN-003: Add `Team.arun()`
- [ ] DESIGN-006: Fan-out → fan-in chaining
- [ ] FEAT-004: Observability (timing in trace)
- [ ] FEAT-005: Per-unit timeout support
- [ ] FEAT-010: Error routing (dead-letter wire)

### P2 — Backlog (12 items)
- [ ] BUG-003, BUG-004: Dead Task fields
- [ ] DESIGN-001: Tool system
- [ ] DESIGN-002: LLM-driven routing
- [ ] DESIGN-004: Consensus rename
- [ ] DESIGN-008: `on_token` cleanup
- [ ] DESIGN-009: `resume()` development
- [ ] MINOR-002, MINOR-003: Introspection + FlexStore
- [ ] FEAT-001–003, FEAT-006–009: Features
