# FlowForge MVP — Design Document

Created: 2026-03-09
Status: Approved (pending implementation)

## Positioning

"Learn agent orchestration patterns with FlowForge, then ship with whatever framework your team picks."

Rapid prototyping of agent topologies with real LLM calls. NOT a production framework — graduate to LangGraph/CrewAI/Agno for deployment.

## Gap Analysis Summary

Source: Crawled PocketFlow, Agno, LangGraph, CrewAI docs + Anthropic's "Building Effective Agents" guide.

| Gap | Severity | Resolution |
|-----|----------|------------|
| Fake parallelism (sequential fan-out) | P0 | ThreadPoolExecutor + asyncio.gather |
| Parallel requires FlexStore (breaks typed state) | P0 | __reducers__ ClassVar on StoreBase |
| No evaluator-optimizer loop | P0 | flow.loop() primitive |
| No async support | P1 | AsyncUnit, Flow.arun() |
| No streaming | P1 | on_token callback |
| Team strategies hardcoded | P2 | Strategy registry |
| No mermaid export | P2 | flow.to_mermaid() |

## Architecture Decisions

### Async Strategy
- prep/post stay sync (fast store access, no I/O)
- Only exec goes async (where LLM calls happen)
- Flow auto-detects sync vs async via inspect.iscoroutinefunction
- Sync Flow.run() unchanged for backward compat

### Typed Reducers
- `__reducers__: ClassVar[dict]` on StoreBase (not json_schema_extra)
- ClassVar excluded from Pydantic schema — clean separation
- Auto-detected by Flow and Team during parallel execution
- Existing ReducerRegistry still works for explicit config

### LLM Integration
- LiteLLM as optional convenience (pip install flowforge[litellm])
- Bring-your-own llm_fn always works: (system, user, tools) -> str
- Docs show copy-paste snippets for OpenAI and Anthropic direct APIs

### Patterns (4 of 5 Anthropic patterns)
1. Prompt chaining (sequential) — already works
2. Parallelization (fan-out + merge) — fix to be real
3. Orchestrator-workers (hierarchical) — refine
4. Evaluator-optimizer (loop) — new flow.loop() primitive
- Routing omitted: emerges from eval-opt feeding orchestrator-workers

### Extensibility
- Team.register_strategy() for custom strategies
- Unit subclassing for custom computation
- flow.loop() composes over existing wire mechanics
- flow.to_mermaid() for documentation export

## What's Deliberately Out of Scope
- Durable execution / persistence across restarts
- Tool execution framework
- Deployment / serving infrastructure
- Observability / tracing integrations
- Long-term memory
- Interactive visualization
- Routing as a dedicated strategy
