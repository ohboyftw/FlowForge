"""
FlowForge Example 1: Research & Report Pipeline
═══════════════════════════════════════════════════

THE canonical multi-agent example. Every framework shows this:
  - CrewAI: "researcher + reporting_analyst" (their quickstart)
  - LangGraph: "research node → write node" (their tutorial)
  - Agno: "web_agent + finance_agent team"

This example shows THREE levels of FlowForge usage:
  Level 1: One-liner (Agno-grade simplicity)
  Level 2: Team composition (CrewAI-grade roles)
  Level 3: Custom graph (LangGraph-grade control)

All three produce the same output — a researched report on a topic.
"""


from flowforge import (
    Agent, Team, StoreBase, Flow, Unit, FunctionUnit,
    Persona, Task, LLMUnit, Personas,
)
from pydantic import Field


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared: Mock LLM for demonstration (swap for real LLM)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def mock_llm(system: str, user: str, tools: list = None) -> str:
    """Mock LLM that simulates responses based on role."""
    if "Researcher" in system:
        return (
            "Key findings on AI agents in 2025:\n"
            "1. PocketFlow proves 100-line frameworks are sufficient\n"
            "2. CrewAI hit $3.2M revenue with role-based agents\n"
            "3. LangGraph is the recommended successor to LangChain\n"
            "4. OpenAI Agents SDK replaced experimental Swarm\n"
            "5. Agno claims 10,000x faster agent instantiation"
        )
    elif "Writer" in system or "Report" in system:
        return (
            "# AI Agent Frameworks: 2025 Landscape Report\n\n"
            "## Executive Summary\n"
            "The agent framework ecosystem has matured significantly...\n\n"
            "## Key Findings\n"
            "The research reveals five major trends shaping the space...\n\n"
            "## Conclusion\n"
            "The convergence toward graph-based primitives suggests..."
        )
    return f"[Response to: {user[:80]}...]"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LEVEL 1: One-liner — Agno-grade simplicity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def level_1_oneliner():
    """
    Simplest possible usage. One agent, one call.
    Equivalent to Agno's:
        agent = Agent(model=Groq(...), tools=[DuckDuckGoTools()])
        agent.print_response("Tell me about AI agents")
    """
    print("\n═══ Level 1: One-liner ═══")

    researcher = Agent("Researcher", "Find information about AI agents",
                       llm_fn=mock_llm)
    result = researcher.run("Research the AI agent framework landscape in 2025")
    print(result)
    print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LEVEL 2: Team composition — CrewAI-grade roles
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def level_2_team():
    """
    Two agents collaborate sequentially.
    Equivalent to CrewAI's:
        crew = Crew(
            agents=[researcher, reporting_analyst],
            tasks=[research_task, reporting_task],
            process=Process.sequential
        )
        crew.kickoff(inputs={"topic": "AI agents"})
    """
    print("═══ Level 2: Team (sequential) ═══")

    researcher = Agent(
        "Senior Researcher",
        "Find comprehensive, accurate information",
        backstory="PhD in computer science, expert at synthesizing technical trends",
        constraints=["Always cite sources", "Focus on 2024-2025 developments"],
        llm_fn=mock_llm,
    )

    writer = Agent(
        "Report Writer",
        "Create polished, detailed reports from research",
        backstory="Technical writer with 10 years at top tech publications",
        constraints=["Use markdown formatting", "Include executive summary"],
        llm_fn=mock_llm,
    )

    team = Team(
        [researcher, writer],
        strategy="sequential",
    )

    # Run the team
    store = team.run("AI agent frameworks landscape in 2025")

    # Inspect what happened
    print(f"Graph structure:\n{team.describe()}")
    print(f"\nResearcher output: {getattr(store, 'senior_researcher', 'N/A')[:100]}...")
    print(f"Writer output: {getattr(store, 'report_writer', 'N/A')[:100]}...")
    print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LEVEL 3: Custom graph — LangGraph-grade control
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ReportState(StoreBase):
    """Typed state for the research → report pipeline."""
    topic: str = ""
    research_findings: list[str] = Field(default_factory=list)
    draft_report: str = ""
    review_notes: str = ""
    quality_score: float = 0.0
    is_approved: bool = False
    revision_count: int = 0


class ResearchUnit(Unit):
    """Searches for information and populates findings."""
    def __init__(self, llm_fn):
        self.llm_fn = llm_fn
        self.persona = Personas.researcher()
        self.persona.constraints = ["Focus on technical details", "Cite specific frameworks"]

    def prep(self, store: ReportState):
        return store.topic

    def exec(self, topic: str) -> list[str]:
        result = self.llm_fn(
            system=self.persona.to_prompt(),
            user=f"Research this topic thoroughly: {topic}",
        )
        return result.split("\n")

    def post(self, store: ReportState, findings: list[str]) -> str:
        store.research_findings = [f.strip() for f in findings if f.strip()]
        return "default"


class WriteUnit(Unit):
    """Writes a report from research findings."""
    def __init__(self, llm_fn):
        self.llm_fn = llm_fn

    def prep(self, store: ReportState):
        return {
            "topic": store.topic,
            "findings": store.research_findings,
        }

    def exec(self, context: dict) -> str:
        return self.llm_fn(
            system="You are a Report Writer. Create polished markdown reports.",
            user=f"Write a report on '{context['topic']}' using these findings:\n"
                 + "\n".join(context["findings"]),
        )

    def post(self, store: ReportState, report: str) -> str:
        store.draft_report = report
        store.revision_count += 1
        return "default"


class ReviewUnit(Unit):
    """Reviews the draft and scores quality."""
    def __init__(self, llm_fn):
        self.llm_fn = llm_fn

    def prep(self, store: ReportState):
        return store.draft_report

    def exec(self, draft: str) -> dict:
        # In production, this would be an LLM call
        # Here we simulate a quality check
        has_sections = "##" in draft
        has_summary = "Summary" in draft
        score = 0.5 + (0.25 if has_sections else 0) + (0.25 if has_summary else 0)
        return {"score": score, "notes": "Needs more data citations" if score < 0.9 else "Approved"}

    def post(self, store: ReportState, review: dict) -> str:
        store.quality_score = review["score"]
        store.review_notes = review["notes"]

        if store.quality_score >= 0.8 or store.revision_count >= 3:
            store.is_approved = True
            return "approved"
        return "needs_revision"


def level_3_custom_graph():
    """
    Full graph control with conditional loops, checkpointing, and typed state.
    Equivalent to LangGraph's:
        graph = StateGraph(State)
        graph.add_node("research", research_node)
        graph.add_node("write", write_node)
        graph.add_node("review", review_node)
        graph.add_conditional_edges("review", route_review)
    """
    print("═══ Level 3: Custom Graph (with review loop) ═══")

    # Build the graph
    flow = Flow()
    flow.add("research", ResearchUnit(mock_llm))
    flow.add("write", WriteUnit(mock_llm))
    flow.add("review", ReviewUnit(mock_llm))

    # Wiring: research → write → review
    flow.wire("research", "write")
    flow.wire("write", "review")

    # Conditional: review loops back to write OR exits
    flow.wire("review", "write",
              on="needs_revision",
              when=lambda s: s.revision_count < 3)
    # (If "approved" or max revisions, flow ends — no wire = termination)

    flow.entry("research")

    # Execute with typed state
    state = ReportState(topic="AI agent frameworks landscape in 2025")
    state.checkpoint("initial")  # save starting state

    result = flow.run(state)

    # Inspect results
    print(f"Topic: {result.topic}")
    print(f"Findings: {len(result.research_findings)} items")
    print(f"Report preview: {result.draft_report[:120]}...")
    print(f"Quality score: {result.quality_score}")
    print(f"Approved: {result.is_approved}")
    print(f"Revisions: {result.revision_count}")
    print(f"\nExecution trace:")
    for step in flow.trace:
        print(f"  Step {step['step']}: {step['unit']} → action='{step['action']}'")
    print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run all levels
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════╗")
    print("║  FlowForge Example 1: Research & Report      ║")
    print("║  Three levels of abstraction, same result    ║")
    print("╚══════════════════════════════════════════════╝")

    level_1_oneliner()
    level_2_team()
    level_3_custom_graph()

    print("═══ All levels complete ═══")
