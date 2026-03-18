# Proposal: FlowForge HTTP Server for n8n Integration

**Date:** 2026-03-10
**Status:** Draft

## Problem

FlowForge is an in-process Python orchestration library. It has no persistence, no scheduling, and no way to trigger Flows externally. If the process dies mid-Flow, all state is lost.

n8n solves all three: cron/webhook triggers, per-step execution persistence in Postgres, and automatic retry. But n8n can't call FlowForge — there's no HTTP interface.

## Goal

Add a thin FastAPI server (`flowforge.server`) that exposes Flow execution over HTTP — enabling n8n (or any HTTP client) to start, poll, resume, and inspect Flows.

## Why n8n, Not Paperclip

Paperclip (v0.3.0) is an AI agent org-chart orchestrator. It treats agents as black boxes between heartbeats. If a Flow crashes mid-execution, Paperclip doesn't know — the next heartbeat restarts from scratch unless you manually build checkpoint/restore plumbing.

n8n operates at the right granularity for FlowForge:

| Concern | n8n | Paperclip |
|---|---|---|
| Persistence grain | Per-node (matches Unit) | Per-heartbeat (too coarse) |
| Crash recovery | Resume from last saved node | Re-run entire agent |
| Retry | Per-node, configurable backoff | Next heartbeat |
| Human-in-loop | Wait node (pause until webhook) | Approval API (more ceremony) |
| Triggers | Cron, webhook, app event, manual | Heartbeat schedule only |
| State handoff | JSON payload between nodes | Manual serialize/deserialize |

## Design

### New Module: `flowforge.server`

One new file: `src/flowforge/server.py` (~150 lines)

Dependencies added: `fastapi`, `uvicorn` (optional extra in pyproject.toml)

### Architecture

```
n8n                              FlowForge Server
────                             ────────────────
Cron/Webhook
  │
  ├─ POST /flows/run ──────────→ Start Flow, return run_id
  │                              Background: flow.run(store)
  │                              State saved to disk on complete/interrupt/error
  ├─ GET  /flows/{id}/status ──→ Poll: running | complete | interrupted | error
  │
  ├─ GET  /flows/{id}/result ──→ Full result: state JSON + trace + metadata
  │
  ├─ POST /flows/{id}/resume ──→ Resume from interrupt (after human approval)
  │                              Calls flow.resume(store, from_node)
  │
  └─ GET  /flows/{id}/trace ───→ Execution trace: step-by-step unit history
```

### State Directory

All run state persisted to `{data_dir}/runs/{run_id}/`:

```
runs/
  abc123/
    meta.json        # {"status": "complete", "started_at": ..., "flow_name": ...}
    state.json       # StoreBase.to_json() — latest state snapshot
    trace.json       # Flow execution trace (step, unit, action, timestamp)
    error.txt        # Stack trace if errored (optional)
```

File-based, not Postgres — keeps FlowForge dependency-free. n8n provides the durable Postgres layer on its side.

### API Specification

#### `POST /flows/run`

Start a Flow execution. Returns immediately with `run_id`.

**Request:**
```json
{
  "flow": "content_pipeline",
  "task": "Write a report on MCP protocol adoption",
  "store": {},
  "config": {
    "max_steps": 100,
    "timeout_s": 300
  }
}
```

- `flow` — registered Flow name (see Flow Registry below)
- `task` — task description passed to Agent/Team
- `store` — optional initial state (merged into StoreBase)
- `config` — optional execution config

**Response:**
```json
{
  "run_id": "abc123",
  "status": "running"
}
```

#### `GET /flows/{run_id}/status`

Poll for completion. Designed for n8n's loop-until-done pattern.

**Response:**
```json
{
  "run_id": "abc123",
  "status": "complete",
  "started_at": "2026-03-10T14:30:00Z",
  "elapsed_ms": 45200,
  "steps_executed": 4
}
```

Status values: `running`, `complete`, `interrupted`, `error`

#### `GET /flows/{run_id}/result`

Full result after completion.

**Response:**
```json
{
  "run_id": "abc123",
  "status": "complete",
  "state": { "research": "...", "draft": "...", "score": 0.92 },
  "trace": [
    {"step": 1, "unit": "researcher", "action": "default", "ms": 12000},
    {"step": 2, "unit": "writer", "action": "default", "ms": 8000},
    {"step": 3, "unit": "reviewer", "action": "accept", "ms": 5000}
  ],
  "metadata": {
    "total_ms": 25000,
    "steps": 3
  }
}
```

#### `POST /flows/{run_id}/resume`

Resume an interrupted Flow after human approval.

**Request:**
```json
{
  "approved": true,
  "context": "Approved by board — proceed with deploy"
}
```

**Response:**
```json
{
  "run_id": "abc123",
  "status": "running"
}
```

#### `GET /flows/{run_id}/trace`

Execution trace for debugging.

**Response:**
```json
{
  "run_id": "abc123",
  "trace": [
    {"step": 1, "unit": "researcher", "action": "default", "ms": 12000, "state_snapshot": {...}}
  ]
}
```

#### `GET /flows`

List registered Flows.

**Response:**
```json
{
  "flows": [
    {"name": "content_pipeline", "description": "Research → Draft → Edit", "nodes": 3},
    {"name": "ops_health_check", "description": "VPS health monitoring", "nodes": 2}
  ]
}
```

### Flow Registry

Flows are registered via Python — no YAML/JSON config language.

```python
# my_flows.py
from flowforge import Agent, Team, Flow
from flowforge.server import FlowRegistry

registry = FlowRegistry()

@registry.register("content_pipeline", description="Research → Draft → Edit")
def content_pipeline(task: str, store_data: dict) -> tuple[Flow, StoreBase]:
    researcher = Agent("Researcher", "Find info", llm_fn=my_llm)
    writer = Agent("Writer", "Draft content", llm_fn=my_llm)
    editor = Agent("Editor", "Polish", llm_fn=my_llm)

    team = Team([researcher, writer, editor], strategy="sequential")
    flow = team.compile(task)
    store = FlexStore(task=task, **store_data)
    return flow, store

@registry.register("ops_check", description="VPS health check")
def ops_check(task: str, store_data: dict) -> tuple[Flow, StoreBase]:
    # Custom Flow with interrupt before destructive action
    flow = Flow()
    flow.add("check", HealthCheckUnit())
    flow.add("fix", FixUnit())
    flow.wire("check", "fix", on="anomaly", interrupt=True)  # Human approval required
    flow.entry("check")
    store = OpsState(**store_data)
    return flow, store
```

### Server Startup

```python
# run_server.py
from flowforge.server import create_app
from my_flows import registry

app = create_app(registry=registry, data_dir="./flow_data")

# Or CLI:
# py -m flowforge.server --flows my_flows:registry --port 8100
```

### InterruptSignal → n8n Wait Node Bridge

This is the key integration. When a Flow hits an interrupt wire:

1. FlowForge catches `InterruptSignal`, saves `sig.store` to `state.json`
2. Sets run status to `interrupted` with `resume_from` node name
3. n8n polls `/status`, sees `interrupted`
4. n8n branches to "Send approval email" or "Wait for webhook"
5. Human approves → webhook fires → n8n calls `POST /flows/{id}/resume`
6. Server loads state from disk, calls `flow.resume(store, from_node)`

```
n8n Workflow:
                                          ┌─────────────┐
POST /flows/run ─→ Poll /status ─→ IF ──→ │ interrupted  │
                       ↑            │      │ Send email   │
                       │            │      │ Wait webhook │
                       └── running ─┘      │ POST /resume │
                                           └──────┬──────┘
                                    complete ←─────┘
                                       │
                                       ▼
                                  GET /result → Notify
```

### What This Does NOT Include

- **Authentication** — add middleware per deployment (API key, OAuth)
- **WebSocket streaming** — poll-based is simpler and sufficient for n8n
- **Multi-tenant isolation** — single-user server, n8n handles multi-flow
- **Database backend** — file-based state; n8n's Postgres is the durable store
- **Auto-scaling** — single process, one Flow at a time per `run_id`
- **Metrics/monitoring** — n8n execution history handles this

## Implementation Plan

### Task 1: FlowRegistry class

**Files:** Create `src/flowforge/server.py`

- `FlowRegistry` class with `register()` decorator
- Stores flow builder functions by name
- `list()` returns registered flows with metadata
- `build(name, task, store_data)` calls builder, returns `(Flow, StoreBase)`

### Task 2: Run manager

**Files:** Extend `src/flowforge/server.py`

- `RunManager` class — tracks active/completed runs
- `start(flow_name, task, store_data, config)` → spawns in background thread, returns `run_id`
- `get_status(run_id)` → status + timing
- `get_result(run_id)` → full state + trace
- Catches `InterruptSignal` → saves state, sets status to `interrupted`
- Catches exceptions → saves traceback, sets status to `error`
- On completion → saves final state to `{data_dir}/runs/{run_id}/`

### Task 3: Resume logic

**Files:** Extend `src/flowforge/server.py`

- `resume(run_id, approved, context)` → loads state from disk, calls `flow.resume()`
- Re-registers flow builder (needed to reconstruct Flow graph)
- Spawns in background thread, same completion handling as Task 2

### Task 4: FastAPI endpoints

**Files:** Extend `src/flowforge/server.py`

- 6 endpoints: `POST /flows/run`, `GET /flows/{id}/status`, `GET /flows/{id}/result`, `POST /flows/{id}/resume`, `GET /flows/{id}/trace`, `GET /flows`
- `create_app(registry, data_dir)` factory function
- Optional CLI entry: `py -m flowforge.server --flows module:registry --port 8100`

### Task 5: Tests

**Files:** Create `tests/test_server.py`

- Test FlowRegistry: register, list, build
- Test RunManager: start, poll, complete, interrupt, error, resume
- Test API endpoints via `httpx.AsyncClient` + FastAPI TestClient
- Test interrupt → resume lifecycle end-to-end
- Test state persistence to disk and reload

### Task 6: Example Flow + n8n workflow JSON

**Files:** Create `examples/11_n8n_integration.py`, `examples/n8n_workflow.json`

- Example: health check Flow with interrupt before fix
- Importable n8n workflow JSON (n8n supports JSON import)
- README section documenting the integration

## Dependencies

```toml
[project.optional-dependencies]
server = ["fastapi>=0.100", "uvicorn>=0.20"]
```

Server is an optional extra — `pip install flowforge[server]`. Core FlowForge remains Pydantic-only.

## Example n8n Workflow (Importable)

```json
{
  "name": "FlowForge Health Check",
  "nodes": [
    {
      "type": "n8n-nodes-base.scheduleTrigger",
      "parameters": { "rule": { "interval": [{ "field": "hours", "hoursInterval": 4 }] } },
      "name": "Every 4 Hours"
    },
    {
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "POST",
        "url": "http://localhost:8100/flows/run",
        "body": { "flow": "ops_check", "task": "Check VPS health" }
      },
      "name": "Start Flow"
    },
    {
      "type": "n8n-nodes-base.wait",
      "parameters": { "amount": 10, "unit": "seconds" },
      "name": "Wait"
    },
    {
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "GET",
        "url": "=http://localhost:8100/flows/{{ $('Start Flow').item.json.run_id }}/status"
      },
      "name": "Poll Status"
    },
    {
      "type": "n8n-nodes-base.if",
      "parameters": {
        "conditions": { "string": [{ "value1": "={{ $json.status }}", "value2": "running" }] }
      },
      "name": "Still Running?"
    },
    {
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "GET",
        "url": "=http://localhost:8100/flows/{{ $('Start Flow').item.json.run_id }}/result"
      },
      "name": "Get Result"
    }
  ],
  "connections": {
    "Every 4 Hours": { "main": [[ { "node": "Start Flow" } ]] },
    "Start Flow": { "main": [[ { "node": "Wait" } ]] },
    "Wait": { "main": [[ { "node": "Poll Status" } ]] },
    "Poll Status": { "main": [[ { "node": "Still Running?" } ]] },
    "Still Running?": {
      "main": [
        [ { "node": "Wait" } ],
        [ { "node": "Get Result" } ]
      ]
    }
  }
}
```

## Success Criteria

1. `py -m flowforge.server --flows examples.11_n8n_integration:registry` starts and serves
2. `POST /flows/run` starts a Flow and returns `run_id` within 100ms
3. `GET /flows/{id}/status` reflects real-time execution state
4. InterruptSignal → `interrupted` status → `POST /resume` → Flow completes
5. State survives server restart (file-based persistence)
6. n8n workflow JSON imports and runs end-to-end against the server
7. No new required dependencies (fastapi/uvicorn are optional extras)
