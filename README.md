```
    _____ _               _____
   |  ___| | _____      _|  ___|__  _ __ __ _  ___
   | |_  | |/ _ \ \ /\ / / |_ / _ \| '__/ _` |/ _ \
   |  _| | | (_) \ V  V /|  _| (_) | | | (_| |  __/
   |_|   |_|\___/ \_/\_/ |_|  \___/|_|  \__, |\___|
                                         |___/
```

**Unified agent orchestration in ~1600 lines of Python.**

PocketFlow's graph primitives + LangGraph's state machines + CrewAI's roles + Agno's ergonomics — one framework, progressive disclosure.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Pydantic v2](https://img.shields.io/badge/pydantic-v2-E92063.svg)](https://docs.pydantic.dev/)
[![Tests](https://img.shields.io/badge/tests-58%20passing-brightgreen.svg)](#testing)

---

```python
from flowforge import Agent, Team, Flow, StoreBase

# One-liner
result = Agent("Researcher", "Find papers on LLM agents").run("What's new?")

# Team of agents
team = Team([researcher, writer, editor], strategy="sequential")
team.run("Write a report on AI agents")

# Full graph control
flow = Flow()
flow.add("research", ResearchUnit()).add("write", WriteUnit())
flow.wire("research", "write", when=lambda s: s.has_data)
flow.entry("research").run(MyState())
```

## Why FlowForge?

Every agent framework converges on the same computation model — a directed graph with shared state. They just disagree on where to put the abstraction boundary.

| Framework | Abstraction | Strength | Pain Point |
|-----------|------------|----------|------------|
| PocketFlow | Graph primitives | 100 lines, zero deps | Rebuild everything |
| LangGraph | State machines | Conditional routing, checkpoints | Steep learning curve |
| CrewAI | Roles & tasks | Intuitive team metaphor | Limited flow control |
| Agno | Pure Python | Blazing fast, ergonomic | No graph visualization |
| **FlowForge** | **Layer cake** | **All of the above** | **Pick your level** |

FlowForge's answer: **progressive disclosure**. Start with `Agent("role").run("task")` and only reach for graph primitives when you need conditional branching, checkpointing, or custom state.

## The Layer Cake

```
  ╔═══════════════════════════════════════════════════════╗
  ║  HARNESS LAYER         Agent(), Team(), .run()        ║  ← start here
  ║  · Agno-style ergonomics                              ║
  ╠═══════════════════════════════════════════════════════╣
  ║  IDENTITY LAYER        Persona, Task, TaskResult      ║  ← when you need roles
  ║  · CrewAI-style semantics                             ║
  ╠═══════════════════════════════════════════════════════╣
  ║  PRIMITIVE LAYER       Unit, Wire, Flow, StoreBase    ║  ← when you need control
  ║  · PocketFlow + LangGraph power                       ║
  ╚═══════════════════════════════════════════════════════╝
         ▲ each layer compiles down to the one below
```

## Installation

```bash
pip install flowforge
```

Or from source:

```bash
git clone https://github.com/ohboyftw/FlowForge.git
cd FlowForge
pip install -e ".[dev]"
```

**Only hard dependency**: `pydantic>=2.0`

**Optional** (for real LLM calls): `pip install flowforge[litellm]` or `flowforge[anthropic]` or `flowforge[openai]`

## Quick Start

### 1. Define Your State

```python
from flowforge import StoreBase
from pydantic import Field

class ResearchState(StoreBase):
    query: str = ""
    findings: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    approved: bool = False
```

Validates on every mutation. Catches typos (`extra="forbid"`). Serializes to JSON. Supports checkpoints and rollback.

### 2. Single Agent

```python
from flowforge import Agent

researcher = Agent(
    "Senior Researcher",
    "Find accurate technical information",
    backstory="PhD in ML, 10 years at DeepMind",
    model="claude-sonnet-4-20250514",
)
result = researcher.run("What are the best agent frameworks?")
```

### 3. Team of Agents

```python
from flowforge import Agent, Team

researcher = Agent("Researcher", "Find information", llm_fn=my_llm)
writer = Agent("Writer", "Create reports", llm_fn=my_llm)
editor = Agent("Editor", "Polish content", llm_fn=my_llm)

team = Team([researcher, writer, editor], strategy="sequential")
result = team.run("Write a report on AI agents")
```

**Strategies**: `sequential` | `parallel` | `hierarchical` | `consensus`

### 4. Custom Graph

```python
from flowforge import Flow, Unit, StoreBase

class MyState(StoreBase):
    count: int = 0
    done: bool = False

class IncrementUnit(Unit):
    def prep(self, store):
        return store.count
    def exec(self, count):
        return count + 1
    def post(self, store, result):
        store.count = result
        return "continue" if result < 5 else "done"

flow = Flow()
flow.add("inc", IncrementUnit())
flow.add("finish", FunctionUnit(lambda s: setattr(s, 'done', True) or "default"))
flow.wire("inc", "inc", on="continue")   # self-loop
flow.wire("inc", "finish", on="done")
flow.entry("inc")

state = MyState()
flow.run(state)
assert state.count == 5
```

### 5. Human-in-the-Loop

```python
from flowforge import Flow, InterruptSignal

flow.wire("analyze", "execute", interrupt=True)  # pauses here

try:
    flow.run(state)
except InterruptSignal as sig:
    print(f"Pending: {sig.store.proposed_action}")
    flow.resume(sig.store, from_node="execute")  # human approves
```

### 6. Async

```python
from flowforge import AsyncUnit, Flow

class FetchUnit(AsyncUnit):
    async def exec(self, url):
        async with aiohttp.ClientSession() as s:
            return await (await s.get(url)).text()

await flow.arun(state)
```

## Examples

All examples run standalone with mock LLMs — no API keys needed:

```bash
python examples/01_research_report.py
```

| # | Example | Pattern | What It Shows |
|---|---------|---------|---------------|
| 01 | [Research & Report](examples/01_research_report.py) | Pipeline | 3 abstraction levels side-by-side |
| 02 | [Customer Support](examples/02_customer_support.py) | Router | Triage → specialists, human-in-loop |
| 03 | [Content Pipeline](examples/03_content_pipeline.py) | Pipeline + Loop | Research → draft → edit → SEO with revision loop |
| 04 | [Stock Analysis](examples/04_stock_analysis.py) | Fan-out / Fan-in | Parallel specialists → synthesis |
| 05 | [Tango Code Review](examples/05_tango_review.py) | Cross-pollination | Dual reviewers with consensus |
| 06 | [Prompt Chaining](examples/06_prompt_chaining.py) | Chain | Outline → draft → edit pipeline |
| 07 | [Parallelization](examples/07_parallelization.py) | Parallel | 3 analysts with reducer-merged results |
| 08 | [Orchestrator-Workers](examples/08_orchestrator_workers.py) | Hierarchical | Manager decomposes, workers execute |
| 09 | [Evaluator-Optimizer](examples/09_evaluator_optimizer.py) | Loop | Generate → evaluate → refine cycle |
| 10 | [YoExecute Orchestrator](examples/10_yoexecute_orchestrator.py) | Real-world | Autonomous PM workflow |

## Using with Real LLMs

FlowForge is model-agnostic. Pass any `(system, user, tools?) -> str` callable:

```python
import litellm  # or anthropic, openai — any provider works

def my_llm(system, user, tools=None):
    resp = litellm.completion(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content

agent = Agent("Researcher", "Find data", llm_fn=my_llm)
```

The pattern is the same for any provider — wrap the client call in a `(system, user, tools?) -> str` function. See the [examples](examples/) for Anthropic and OpenAI variants.

## Escape Hatches

Every upper layer exposes the layer below — no black boxes:

```python
# Team → compiled Flow
team = Team([agent1, agent2], strategy="sequential")
graph = team.graph                         # access the Flow
graph.wire("agent2", "agent1",             # add custom wiring
           on="retry", when=lambda s: s.score < 0.8)

# Agent → underlying Unit
unit = agent.as_unit(task)                 # get the LLMUnit
flow.add("custom_node", unit)              # use in manual Flow

# Persona → system prompt
print(agent.persona.to_prompt())           # see what the LLM gets

# Store → JSON / dict
state.checkpoint("before_risky_op")        # save state
state.rollback("before_risky_op")          # restore
json_str = state.to_json()                 # serialize
```

## Architecture

```
src/flowforge/
├── __init__.py      Public API exports
├── core.py          StoreBase, Unit, AsyncUnit, Wire, Flow    (812 lines)
├── identity.py      Persona, Task, TaskResult                 (228 lines)
└── harness.py       Agent, Team, LLMUnit, AsyncLLMUnit        (576 lines)

examples/            10 runnable patterns (mock LLMs, no keys)
tests/               58 tests across 5 test files
```

## Design Principles

1. **Progressive disclosure** — start simple, add complexity only when needed
2. **Pydantic all the way down** — Store, Persona, Task, TaskResult are all BaseModel
3. **Compilation model** — upper layers compile to lower layers, never magic
4. **Intentional leakiness** — escape hatches at every boundary, not accidental leaks
5. **Model agnostic** — `llm_fn` is just a callable, use any provider
6. **Zero framework lock-in** — only hard dependency is Pydantic

## Testing

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

58 tests covering all layers: primitives, identity, harness, async, strategies, and smoke tests.

## License

MIT

## Credits

FlowForge stands on the shoulders of:

- [PocketFlow](https://github.com/The-Pocket/PocketFlow) — the 100-line graph insight
- [LangGraph](https://github.com/langchain-ai/langgraph) — state machines for agents
- [CrewAI](https://github.com/crewAIInc/crewAI) — role-based collaboration
- [Agno](https://github.com/agno-agi/agno) — ergonomic, fast agent framework

Built by [Aravind](https://github.com/ohboyftw)
