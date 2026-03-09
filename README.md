# FlowForge

**Unified agent orchestration in ~250 lines of core Python.**

FlowForge combines the best ideas from PocketFlow, LangGraph, CrewAI, and Agno into a single, layered framework where each layer is independently useful and compiles down to the one below.

```python
# One-liner — Agno-grade simplicity
agent = Agent("Researcher", "Find papers on LLM agents", model="claude-sonnet-4-20250514")
result = agent.run("What's new in agentic AI?")

# Team — CrewAI-grade roles
team = Team([researcher, coder, reviewer], strategy="sequential")
team.run("Build a CLI tool")

# Custom graph — LangGraph-grade control
flow = Flow[MyState]()
flow.add("research", ResearchUnit())
flow.add("code", CodeUnit())
flow.wire("research", "code", when=lambda s: s.has_data)
flow.entry("research")
flow.run(MyState(query="test"))
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
┌─────────────────────────────────────────────────────┐
│  ERGONOMIC LAYER — Agent(), Team(), .run()           │  ← Start here
│  Developer-facing API (Agno philosophy)              │
├─────────────────────────────────────────────────────┤
│  ORCHESTRATION LAYER — Flow, Wire, conditional edges │  ← When you need control
│  Graph engine (LangGraph power)                      │
├─────────────────────────────────────────────────────┤
│  IDENTITY LAYER — Persona, Task, TaskResult          │  ← When you need roles
│  Role system (CrewAI semantics)                      │
├─────────────────────────────────────────────────────┤
│  PRIMITIVE LAYER — Unit, StoreBase(Pydantic), Flow   │  ← When you need everything
│  Core graph + typed state (PocketFlow minimal)       │
└─────────────────────────────────────────────────────┘
```

## 5 Primitives, Zero More

| Primitive | From | Purpose |
|-----------|------|---------|
| **Unit** | PocketFlow Node | Atomic computation: `prep → exec → post` lifecycle |
| **Wire** | LangGraph Edge | Typed connection with conditions, interrupts, fan-out |
| **Persona** | CrewAI Agent | Role/goal/backstory identity → compiles to system prompt |
| **StoreBase** | Pydantic BaseModel | Typed shared state with validation, checkpoints, rollback |
| **Harness** | Agno Agent/Team | Ergonomic entry point that auto-wires everything |

## Installation

```bash
# From source (recommended during early development)
git clone https://github.com/ohboyconsultancy/flowforge.git
cd flowforge
pip install -e ".[dev]"

# Or just pip
pip install flowforge
```

**Only dependency**: `pydantic>=2.0`

## Quick Start

### 1. Define Your State (Pydantic all the way down)

```python
from flowforge import StoreBase
from pydantic import Field

class ResearchState(StoreBase):
    query: str = ""
    findings: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    approved: bool = False
```

The Store validates on every mutation, catches typos (`extra="forbid"`), serializes to JSON for checkpoints, and provides schema introspection.

### 2. One Agent (simplest)

```python
from flowforge import Agent

researcher = Agent(
    "Senior Researcher",
    "Find accurate technical information",
    backstory="PhD in ML, 10 years at DeepMind",
    model="claude-sonnet-4-20250514",
)
result = researcher.run("What are the best agent frameworks in 2025?")
```

### 3. Team of Agents

```python
from flowforge import Agent, Team

researcher = Agent("Researcher", "Find information", llm_fn=my_llm)
writer = Agent("Writer", "Create reports", llm_fn=my_llm)
editor = Agent("Editor", "Polish content", llm_fn=my_llm)

# Sequential pipeline
team = Team([researcher, writer, editor], strategy="sequential")
result = team.run("Write a report on AI agents")

# Or hierarchical with a manager
team = Team([researcher, writer], strategy="hierarchical", manager=manager)
```

**Strategies**: `sequential`, `parallel`, `hierarchical`, `consensus`

### 4. Custom Graph (full control)

```python
from flowforge import Flow, Unit, StoreBase

class MyState(StoreBase):
    count: int = 0
    done: bool = False

class IncrementUnit(Unit):
    def prep(self, store: MyState):
        return store.count
    def exec(self, count: int) -> int:
        return count + 1
    def post(self, store: MyState, result: int) -> str:
        store.count = result
        return "continue" if result < 5 else "done"

flow = Flow()
flow.add("increment", IncrementUnit())
flow.add("finish", FunctionUnit(lambda s: setattr(s, 'done', True) or "default"))
flow.wire("increment", "increment", on="continue")  # loop!
flow.wire("increment", "finish", on="done")
flow.entry("increment")

state = MyState()
flow.run(state)
assert state.count == 5
```

### 5. Human-in-the-Loop

```python
from flowforge import Flow, InterruptSignal

flow.wire("analyze", "execute",
          interrupt=True)  # ← Pauses here

try:
    flow.run(state)
except InterruptSignal as sig:
    # Show state to human for review
    print(f"Pending action: {sig.store.proposed_action}")
    # Human approves...
    flow.resume(sig.store, from_node="execute")
```

## Examples

Each example runs standalone with mock LLMs (no API keys needed):

| # | Example | Pattern | Shows |
|---|---------|---------|-------|
| 01 | [Research & Report](examples/01_research_report.py) | Pipeline | 3 abstraction levels side-by-side |
| 02 | [Customer Support](examples/02_customer_support.py) | Router | Triage → specialists, human-in-loop |
| 03 | [Content Pipeline](examples/03_content_pipeline.py) | Pipeline + Loop | Research → draft → edit → SEO with revision |
| 04 | [Stock Analysis](examples/04_stock_analysis.py) | Fan-out/Fan-in | Parallel specialists → synthesis |
| 05 | [Tango Code Review](examples/05_tango_review.py) | Cross-pollination | Dual reviewers with consensus |

```bash
# Run any example
python examples/01_research_report.py
python examples/02_customer_support.py
```

## Using with Real LLMs

FlowForge is model-agnostic. Pass any `(system, user, tools) -> str` callable:

```python
# With LiteLLM (any provider)
import litellm
def my_llm(system, user, tools=None):
    resp = litellm.completion(
        model="claude-sonnet-4-20250514",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content

agent = Agent("Researcher", "Find data", llm_fn=my_llm)

# With OpenAI directly
from openai import OpenAI
client = OpenAI()
def openai_llm(system, user, tools=None):
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content

# With Anthropic directly
import anthropic
client = anthropic.Anthropic()
def claude_llm(system, user, tools=None):
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text
```

## Escape Hatches

Every upper layer exposes the layer below:

```python
# Team → compiled Flow
team = Team([agent1, agent2], strategy="sequential")
graph = team.graph                    # Access the Flow
graph.wire("agent2", "agent1",        # Add custom wire
           on="retry",
           when=lambda s: s.score < 0.8)

# Agent → underlying Unit
unit = agent.as_unit(task)            # Get the LLMUnit
flow.add("custom_node", unit)         # Use in manual Flow

# Persona → system prompt
print(agent.persona.to_prompt())      # See what the LLM receives

# Store → JSON / dict
state.checkpoint("before_risky_op")   # Save state
json_str = state.to_json()            # Serialize
state.rollback("before_risky_op")     # Restore
```

## Design Principles

1. **Progressive disclosure** — start simple, add complexity only when needed
2. **Pydantic all the way down** — Store, Persona, Task, TaskResult are all BaseModel
3. **Compilation model** — upper layers compile to lower layers, never magic
4. **Intentional leakiness** — escape hatches at every boundary, not accidental leaks
5. **Model agnostic** — `llm_fn` is just a callable, use any provider
6. **Zero framework lock-in** — only dependency is Pydantic

## Development

```bash
# Clone and install
git clone https://github.com/ohboyconsultancy/flowforge.git
cd flowforge
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Run all examples
python examples/01_research_report.py
python examples/02_customer_support.py
python examples/03_content_pipeline.py
python examples/04_stock_analysis.py
python examples/05_tango_review.py
```

## Architecture

```
src/flowforge/
├── __init__.py          # Public API exports
├── core.py              # StoreBase, Unit, Wire, Flow (~250 lines)
├── identity.py          # Persona, Task, TaskResult (~120 lines)
└── harness.py           # Agent, Team, LLMUnit (~200 lines)

examples/                # 5 canonical patterns, all runnable
tests/                   # 28 tests covering all layers
```

Total core: **~570 lines**. Only dependency: **Pydantic**.

## License

MIT

## Credits

FlowForge stands on the shoulders of:
- [PocketFlow](https://github.com/The-Pocket/PocketFlow) — the 100-line graph insight
- [LangGraph](https://github.com/langchain-ai/langgraph) — state machines for agents
- [CrewAI](https://github.com/crewAIInc/crewAI) — role-based collaboration
- [Agno](https://github.com/agno-agi/agno) — ergonomic, fast agent framework

Built by [OhboyConsultancy FZ LLC](https://ohboyconsultancy.com) — AI consulting for the Gulf region and India.
