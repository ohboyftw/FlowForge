# FlowForge — Scaffold & Advance Plan

Created: 2026-03-09
Status: Ready for execution

## Context

FlowForge is a unified agent orchestration framework (~250 lines core) combining PocketFlow + LangGraph + CrewAI + Agno into a layered architecture with progressive disclosure.

Source has been extracted from `flowforge-project.tar.gz` into the project root. The core source files exist at `src/flowforge/` (core.py, identity.py, harness.py). Tests exist at `tests/test_flowforge.py`. Five examples exist in `examples/`.

## Phase A — Scaffold (get to runnable state)

- [ ] **A1**: Initialize git repo — `git init && git add -A && git commit -m "feat: initial FlowForge scaffold from archive"`
- [ ] **A2**: Create `.env.example` with placeholder LLM API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY)
- [ ] **A3**: Install dev dependencies — `pip install -e ".[dev]"` or `uv pip install -e ".[dev]"`
- [ ] **A4**: Run `py -m pytest tests/ -v` and confirm all tests pass
- [ ] **A5**: Run `ruff check src/ tests/` and `ruff format --check src/ tests/` — fix any lint issues
- [ ] **A6**: Verify `py -c "from flowforge import Unit, Flow, Agent, Team; print('OK')"` works
- [ ] **A7**: Clean up duplicate files from pre-extraction (`flowforge-CLAUDE.md`, `flowforge-README.md`, `flowforge-pyproject.toml`) — remove if content matches extracted versions
- [ ] **A8**: Commit clean state — `git add -A && git commit -m "chore: clean scaffold, all tests passing"`

### Definition of Done (Phase A)
- Git repo initialized with clean history
- `pytest` passes with 0 failures
- `ruff` passes with 0 errors
- Core imports work
- No duplicate/stale files

## Phase B — Advance (one meaningful step)

- [ ] **B1**: Read all 5 examples in `examples/` — verify each runs without error (mock LLM mode)
- [ ] **B2**: Add a real LLM integration smoke test using `litellm` — test with a cheap model (e.g., `claude-haiku-4-5-20251001`) if API key available, otherwise mock
- [ ] **B3**: Add a `Makefile` target `make smoke` that runs the smoke test
- [ ] **B4**: Verify `Team` with `strategy="parallel"` works correctly (concurrent agent execution)
- [ ] **B5**: Add `py.typed` marker file to `src/flowforge/` for PEP 561 compliance
- [ ] **B6**: Commit — `git add -A && git commit -m "feat: verified examples, added smoke test"`

### Definition of Done (Phase B)
- All examples run without error
- At least one integration test exercises real or mocked LLM call through the harness layer
- Parallel team strategy verified working

## Key Files

| File | Purpose |
|------|---------|
| `src/flowforge/core.py` | Primitive layer: Unit, Flow, StoreBase, Wire |
| `src/flowforge/identity.py` | Identity layer: Persona, Task, TaskResult |
| `src/flowforge/harness.py` | Ergonomic layer: Agent, Team, LLMUnit |
| `tests/test_flowforge.py` | Test suite |
| `examples/` | 5 example scripts |
| `pyproject.toml` | Build config (hatchling) |

## Commands Reference

```bash
# Install
pip install -e ".[dev]"

# Test
py -m pytest tests/ -v

# Lint
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/flowforge/
```
