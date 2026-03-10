"""
Example 08 — Orchestrator-Workers (Anthropic Pattern)

A manager decomposes the task, workers execute in parallel,
manager synthesizes results. Uses hierarchical Team strategy.

Works with mock LLM by default. Set OPENAI_API_KEY or ANTHROPIC_API_KEY
for real LLM calls via litellm.
"""

from flowforge import Agent, Team


# ── Mock LLM ──
def _mock_llm(system: str, user: str, **kwargs) -> str:
    if "decompose" in system.lower() or "decompose" in user.lower():
        return "Subtasks: 1) Research API design patterns 2) Implement core endpoints 3) Write tests"
    if "synthesize" in system.lower():
        return "Final synthesis: All subtasks completed. API is designed, implemented, and tested."
    if "research" in system.lower() or "architect" in system.lower():
        return "Research complete: RESTful patterns with Pydantic validation recommended."
    if "implement" in system.lower() or "engineer" in system.lower():
        return "Implementation complete: FastAPI endpoints with typed request/response models."
    if "test" in system.lower() or "qa" in system.lower():
        return "Testing complete: 95% coverage, all edge cases handled."
    return f"Worker output: {user[:50]}"


def main():
    # Manager agent
    manager = Agent(
        "Tech_Lead",
        "Decompose tasks and synthesize team outputs",
        llm_fn=_mock_llm,
    )

    # Worker agents
    researcher = Agent("API_Architect", "Research API design patterns", llm_fn=_mock_llm)
    engineer = Agent("Backend_Engineer", "Implement API endpoints", llm_fn=_mock_llm)
    tester = Agent("QA_Engineer", "Write comprehensive tests", llm_fn=_mock_llm)

    # Hierarchical team
    team = Team(
        [researcher, engineer, tester],
        strategy="hierarchical",
        manager=manager,
    )

    store = team.run("Build a REST API for task management")

    print("=== Orchestrator-Workers Example ===")
    print(f"\nDecomposition: {getattr(store, 'subtasks', 'N/A')}")
    print(f"\nArchitect: {getattr(store, 'api_architect', 'N/A')}")
    print(f"Engineer: {getattr(store, 'backend_engineer', 'N/A')}")
    print(f"QA: {getattr(store, 'qa_engineer', 'N/A')}")
    print(f"\nFinal: {getattr(store, 'final_result', 'N/A')}")
    print(f"\n--- Mermaid ---\n{team.graph.to_mermaid()}")


if __name__ == "__main__":
    main()
