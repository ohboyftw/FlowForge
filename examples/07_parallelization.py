"""
Example 07 — Parallelization (Anthropic Pattern)

3 analysts run concurrently on typed state with reducers.
Results merge via "extend" reducer into a shared findings list.

Works with mock LLM by default. Set OPENAI_API_KEY or ANTHROPIC_API_KEY
for real LLM calls via litellm.
"""

from pydantic import Field

from flowforge import Agent, StoreBase, Team


# ── Typed State with Reducers ──
class AnalysisState(StoreBase):
    __reducers__ = {
        "findings": "extend",
        "market_analyst": "replace",
        "tech_analyst": "replace",
        "risk_analyst": "replace",
    }

    task: str = ""
    findings: list[str] = Field(default_factory=list)
    market_analyst: str = ""
    tech_analyst: str = ""
    risk_analyst: str = ""


# ── Mock LLM ──
def _mock_llm(system: str, user: str, **kwargs) -> str:
    if "market" in system.lower():
        return "Market analysis: Strong growth potential in AI orchestration space"
    if "tech" in system.lower():
        return "Technical analysis: Python ecosystem well-suited, Pydantic adoption growing"
    if "risk" in system.lower():
        return "Risk analysis: Competition from LangChain/CrewAI, but niche is underserved"
    return f"Analysis: {user[:50]}"


def main():
    # Create 3 analyst agents
    market = Agent("Market_Analyst", "Analyze market opportunity", llm_fn=_mock_llm)
    tech = Agent("Tech_Analyst", "Analyze technical feasibility", llm_fn=_mock_llm)
    risk = Agent("Risk_Analyst", "Analyze risks and mitigation", llm_fn=_mock_llm)

    # Parallel team with typed reducers
    team = Team(
        [market, tech, risk],
        strategy="parallel",
        store_class=AnalysisState,
    )

    state = AnalysisState(task="Evaluate FlowForge as a product")
    result = team.run("Analyze FlowForge framework", store=state)

    print("=== Parallelization Example ===")
    print(f"\nMarket: {result.market_analyst}")
    print(f"Tech: {result.tech_analyst}")
    print(f"Risk: {result.risk_analyst}")
    print(f"\n--- Mermaid ---\n{team.graph.to_mermaid()}")


if __name__ == "__main__":
    main()
