# FlowForge — Claude Code Instructions

## Current State

Freshly extracted from archive. Source files exist, needs git init and test validation.

**Active plan**: `docs/plans/2026-03-09-scaffold-plan.md` — execute Phase A first, then Phase B.

## Project Overview

FlowForge is a unified agent orchestration framework that combines PocketFlow (graph primitives), LangGraph (state machines), CrewAI (role-based agents), and Agno (ergonomic API) into a layered architecture where each layer compiles down to the one below.

**Core philosophy**: Progressive disclosure — start with `Agent("role").run("task")` and reach for graph primitives only when needed.

## Architecture

```
src/flowforge/
├── core.py       # Primitive layer: StoreBase(Pydantic), Unit, Wire, Flow
├── identity.py   # Identity layer: Persona, Task, TaskResult
├── harness.py    # Ergonomic layer: Agent, Team, LLMUnit
└── __init__.py   # Public API
```

### Layer Cake (bottom → top)

1. **Primitive** (`core.py`): `Unit` (prep/exec/post lifecycle), `Flow` (directed graph), `StoreBase` (Pydantic-typed state with checkpointing)
2. **Identity** (`identity.py`): `Persona` (role/goal/backstory → system prompt), `Task` (description + context injection), `TaskResult`
3. **Harness** (`harness.py`): `Agent` (single-agent wrapper), `Team` (multi-agent with strategies: sequential/parallel/hierarchical/consensus)

### Key Design Rules

- **Store is always a Pydantic BaseModel** with `validate_assignment=True` and `extra="forbid"`
- **Unit.exec()** is pure — no store access, keeps it testable
- **Wire conditions** use `when=lambda s: bool` for conditional routing
- **InterruptSignal** is raised (not returned) for human-in-the-loop
- **Team.compile()** returns the Flow before execution — escape hatch for custom wiring
- **LLM function signature**: `(system: str, user: str, tools: list = None) -> str`

## Code Style

- Python 3.10+ with type hints everywhere
- Pydantic v2 models for all data structures
- Docstrings on all public classes and methods
- Use `Field(default_factory=list)` for mutable defaults in Pydantic
- Fluent API for Flow construction: `flow.add(...).wire(...).entry(...)`
- Action labels as strings returned from `Unit.post()` for routing

## Testing

```bash
python -m pytest tests/ -v
```

Tests cover all layers independently. Mock LLMs return deterministic strings for testing.

## Common Tasks

### Adding a new example
1. Create `examples/NN_name.py`
2. Define a typed `StoreBase` subclass for the domain
3. Show both Team API (simple) and Flow API (full control)
4. Use mock LLMs so it runs without API keys
5. Add to README examples table

### Adding a new Team strategy
1. Add strategy name to `Team.STRATEGIES` list in `harness.py`
2. Implement `_build_{strategy}()` method
3. Test in `tests/test_flowforge.py`

### Adding a new Unit type
1. Subclass `Unit` in the appropriate layer
2. Implement `prep()`, `exec()`, `post()`
3. `post()` must return an action label string

## Dependencies

- **Required**: `pydantic>=2.0` (only dependency)
- **Optional**: `litellm`, `anthropic`, `openai` (for real LLM calls)
