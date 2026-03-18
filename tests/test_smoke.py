"""
FlowForge Smoke Test — exercises the full harness layer with mock LLM.

Verifies Agent.run(), Team strategies, and Flow compilation
work end-to-end without requiring real API keys.
"""

from pydantic import Field

from flowforge import Agent, FlexStore, StoreBase, Team


def _mock_llm(system: str, user: str, tools: list = None) -> str:
    """Deterministic mock LLM for testing."""
    return f"[mock] role={system[:30]}... task={user[:50]}..."


class SmokeState(StoreBase):
    task: str = ""
    results: list[str] = Field(default_factory=list)
    step1: str = ""
    step2: str = ""
    step3: str = ""


# ── Single agent ──


def test_single_agent_smoke():
    agent = Agent("Smoke Tester", "Verify things work", llm_fn=_mock_llm)
    result = agent.run("Is FlowForge working?")
    assert result is not None
    assert "[mock]" in result


# ── Sequential team ──


def test_sequential_team_smoke():
    agents = [
        Agent("Step1", "First step", llm_fn=_mock_llm),
        Agent("Step2", "Second step", llm_fn=_mock_llm),
        Agent("Step3", "Third step", llm_fn=_mock_llm),
    ]
    team = Team(agents, strategy="sequential")
    store = team.run("Run a 3-step pipeline", store=SmokeState(task="smoke"))

    assert hasattr(store, "step1")
    assert hasattr(store, "step2")
    assert hasattr(store, "step3")


# ── Parallel team ──


def test_parallel_team_smoke():
    agents = [
        Agent("Worker1", "Do task A", llm_fn=_mock_llm),
        Agent("Worker2", "Do task B", llm_fn=_mock_llm),
        Agent("Worker3", "Do task C", llm_fn=_mock_llm),
    ]
    team = Team(agents, strategy="parallel")
    flow = team.compile("parallel smoke test")

    assert "dispatch" in flow.nodes
    assert "worker1" in flow.nodes
    assert "worker2" in flow.nodes
    assert "worker3" in flow.nodes

    # Actually run it — use FlexStore since parallel writes dynamic agent-named fields
    store = team.run("parallel smoke test", store=FlexStore(task="parallel"))
    # All workers should have written their output fields
    assert hasattr(store, "worker1")
    assert hasattr(store, "worker2")
    assert hasattr(store, "worker3")


# ── Hierarchical team ──


def test_hierarchical_team_smoke():
    mgr = Agent("Manager", "Coordinate team", llm_fn=_mock_llm)
    workers = [
        Agent("Dev1", "Write code", llm_fn=_mock_llm),
        Agent("Dev2", "Write tests", llm_fn=_mock_llm),
    ]
    team = Team(workers, strategy="hierarchical", manager=mgr)
    flow = team.compile("build something")

    assert "decompose" in flow.nodes
    assert "synthesize" in flow.nodes
    assert "dev1" in flow.nodes
    assert "dev2" in flow.nodes


def test_hierarchical_parallel_workers():
    """Hierarchical team wires workers as fan-out, not sequential."""
    mgr = Agent("Manager", "Manage", llm_fn=_mock_llm)
    w1 = Agent("Dev1", "Code", llm_fn=_mock_llm)
    w2 = Agent("Dev2", "Test", llm_fn=_mock_llm)
    t = Team([w1, w2], strategy="hierarchical", manager=mgr)

    flow = t.compile("build")

    # Workers should appear in a fan-out from decompose
    decompose_edges = [(s, tgt, lbl) for s, tgt, lbl in flow.edges if s == "decompose"]
    targets = [tgt for _, tgt, _ in decompose_edges]

    # Both workers and synthesize should be targets of decompose
    assert "dev1" in targets
    assert "dev2" in targets
    assert "synthesize" in targets

    # No direct worker → synthesize edges (continuation handles it)
    worker_synth = [(s, t) for s, t, _ in flow.edges if s in ("dev1", "dev2") and t == "synthesize"]
    assert worker_synth == []

    # End-to-end: should run to completion
    store = FlexStore(task="hierarchical test")
    result = t.run("build something", store=store)
    assert hasattr(result, "final_result")


# ── Consensus team ──


def test_consensus_team_smoke():
    agents = [
        Agent("ReviewerA", "Review from angle A", llm_fn=_mock_llm),
        Agent("ReviewerB", "Review from angle B", llm_fn=_mock_llm),
    ]
    team = Team(agents, strategy="consensus")
    flow = team.compile("review code")

    assert "consensus" in flow.nodes
    assert "dispatch" in flow.nodes


def test_consensus_reaches_consensus_node():
    """Consensus strategy reaches the consensus node with new fan-out semantics."""
    agents = [
        Agent("ReviewerA", "Review A", llm_fn=_mock_llm),
        Agent("ReviewerB", "Review B", llm_fn=_mock_llm),
    ]
    team = Team(agents, strategy="consensus")

    store = FlexStore(task="consensus test")
    result = team.run("review code", store=store)

    # The flow should complete — consensus node ran
    assert result is not None
    # Verify consensus node actually executed by checking the trace
    flow = team.graph
    node_names = [entry["unit"] for entry in flow.trace]
    assert "consensus" in node_names, f"consensus node not in trace: {node_names}"


# ── Flow compilation escape hatch ──


def test_team_compile_and_modify():
    a1 = Agent("Alpha", "Do alpha", llm_fn=_mock_llm)
    a2 = Agent("Beta", "Do beta", llm_fn=_mock_llm)
    team = Team([a1, a2], strategy="sequential")

    graph = team.graph
    graph.wire("beta", "alpha", on="retry", when=lambda s: True)

    edges = graph.edges
    retry_edge = [e for e in edges if e[0] == "beta" and e[1] == "alpha"]
    assert len(retry_edge) == 1
    assert retry_edge[0][2] == "retry"
