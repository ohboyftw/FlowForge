"""
FlowForge Example 3: Content Creation Pipeline
═══════════════════════════════════════════════════

THE content team example. Every framework shows a variation:
  - CrewAI: "Content Creator Flow" (their official example)
  - LangGraph: "Blog writing with research, outline, draft, edit"
  - Agno: "web_agent + writer team"

Shows: Team strategy="hierarchical" where a manager decomposes
work and delegates to specialized writers, then synthesizes.
"""


from flowforge import (
    Agent, Team, StoreBase, Flow, Unit, FunctionUnit,
    Persona, Task, LLMUnit,
)
from pydantic import Field


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Typed State
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ContentState(StoreBase):
    """State for the content creation pipeline."""
    topic: str = ""
    target_audience: str = ""
    content_type: str = "blog_post"  # blog_post, linkedin, newsletter

    # Pipeline stages
    research: str = ""
    outline: list[str] = Field(default_factory=list)
    draft: str = ""
    edited: str = ""
    seo_optimized: str = ""

    # Quality
    word_count: int = 0
    readability_score: float = 0.0
    seo_score: float = 0.0
    revision_round: int = 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pipeline Units
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ResearchUnit(Unit):
    """Trend researcher — finds angles and data points."""
    def prep(self, store: ContentState):
        return {"topic": store.topic, "audience": store.target_audience}

    def exec(self, ctx: dict) -> str:
        return (
            f"Research on '{ctx['topic']}' for {ctx['audience']}:\n"
            "• Current market size: $4.2B (2025)\n"
            "• Key trend: shift from single agents to orchestrated teams\n"
            "• 72% of enterprise AI projects now involve multi-agent systems\n"
            "• Main frameworks: LangGraph, CrewAI, OpenAI SDK, Agno\n"
            "• Gap: no lightweight unified abstraction exists yet"
        )

    def post(self, store: ContentState, research: str) -> str:
        store.research = research
        return "default"


class OutlineUnit(Unit):
    """Creates a structured outline from research."""
    def prep(self, store: ContentState):
        return store.research

    def exec(self, research: str) -> list[str]:
        return [
            "1. Hook: The agent framework explosion of 2025",
            "2. Problem: choosing between 15+ frameworks",
            "3. The hidden pattern: they all converge on graphs",
            "4. Framework comparison: what actually differs",
            "5. Solution: the layer cake approach",
            "6. CTA: start building with progressive disclosure",
        ]

    def post(self, store: ContentState, outline: list[str]) -> str:
        store.outline = outline
        return "default"


class DraftUnit(Unit):
    """Writes the first draft from outline + research."""
    def prep(self, store: ContentState):
        return {"outline": store.outline, "research": store.research}

    def exec(self, ctx: dict) -> str:
        sections = "\n\n".join([
            f"## {item.split('. ', 1)[1] if '. ' in item else item}\n\n"
            f"Lorem ipsum content for section: {item}..."
            for item in ctx["outline"]
        ])
        return f"# AI Agent Frameworks: A Developer's Guide\n\n{sections}"

    def post(self, store: ContentState, draft: str) -> str:
        store.draft = draft
        store.word_count = len(draft.split())
        return "default"


class EditUnit(Unit):
    """Style editor — improves tone, clarity, flow."""
    def prep(self, store: ContentState):
        return {"draft": store.draft, "audience": store.target_audience}

    def exec(self, ctx: dict) -> dict:
        # Simulate editing
        edited = ctx["draft"].replace("Lorem ipsum", "In practice,")
        score = 0.75 + (0.1 if "practice" in edited else 0)
        return {"text": edited, "readability": score}

    def post(self, store: ContentState, result: dict) -> str:
        store.edited = result["text"]
        store.readability_score = result["readability"]
        store.revision_round += 1

        if result["readability"] < 0.8 and store.revision_round < 3:
            return "needs_revision"
        return "default"


class SEOUnit(Unit):
    """SEO optimizer — adds keywords, meta, internal links."""
    def prep(self, store: ContentState):
        return store.edited

    def exec(self, content: str) -> dict:
        optimized = content + (
            "\n\n---\n"
            "*Keywords: AI agents, multi-agent framework, LangGraph, CrewAI*\n"
            "*Meta: A practical guide to choosing the right AI agent framework*"
        )
        return {"text": optimized, "seo_score": 0.88}

    def post(self, store: ContentState, result: dict) -> str:
        store.seo_optimized = result["text"]
        store.seo_score = result["seo_score"]
        return "default"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Build the pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_content_pipeline() -> Flow:
    """
    Content pipeline with revision loop:
    
        research → outline → draft → edit ──→ seo
                                      ↑       │
                                      └───────┘
                                    (if readability < 0.8)
    """
    flow = Flow()

    flow.add("research", ResearchUnit())
    flow.add("outline", OutlineUnit())
    flow.add("draft", DraftUnit())
    flow.add("edit", EditUnit())
    flow.add("seo", SEOUnit())

    flow.wire("research", "outline")
    flow.wire("outline", "draft")
    flow.wire("draft", "edit")
    flow.wire("edit", "seo", on="default")
    flow.wire("edit", "draft", on="needs_revision")  # revision loop

    flow.entry("research")
    return flow


def demo_pipeline():
    """Run the full content pipeline."""
    print("\n── Full Pipeline Run ──")

    flow = build_content_pipeline()

    state = ContentState(
        topic="AI Agent Frameworks in 2025",
        target_audience="senior developers and tech leads",
        content_type="blog_post",
    )

    state.checkpoint("initial")
    result = flow.run(state)

    print(f"Topic: {result.topic}")
    print(f"Audience: {result.target_audience}")
    print(f"Research preview: {result.research[:80]}...")
    print(f"Outline: {len(result.outline)} sections")
    print(f"Draft words: {result.word_count}")
    print(f"Readability: {result.readability_score:.2f}")
    print(f"SEO score: {result.seo_score:.2f}")
    print(f"Revision rounds: {result.revision_round}")
    print(f"\nFinal content preview:")
    print(result.seo_optimized[:300] + "...")
    print(f"\nExecution trace ({len(flow.trace)} steps):")
    for step in flow.trace:
        print(f"  {step['step']}. {step['unit']} → '{step['action']}'")


def demo_team_approach():
    """
    Same pipeline but using Team API (CrewAI-style).
    Shows the "I just want roles, not graphs" approach.
    """
    print("\n── Team Approach (CrewAI-style) ──")

    def mock_llm(system, user, tools=None):
        if "Researcher" in system:
            return "Key trends in AI agents for 2025..."
        elif "Writer" in system:
            return "# Great Blog Post\n\nContent goes here..."
        elif "Editor" in system:
            return "Polished and improved version of the content..."
        return f"[{system[:30]}]: {user[:50]}..."

    researcher = Agent("Trend Researcher",
        "Find trending topics and data points",
        backstory="Former tech journalist, data-driven",
        llm_fn=mock_llm)

    writer = Agent("Content Writer",
        "Write engaging blog posts",
        backstory="10 years writing for TechCrunch and Wired",
        constraints=["Use active voice", "Include code examples"],
        llm_fn=mock_llm)

    editor = Agent("Style Editor",
        "Polish content for clarity and engagement",
        backstory="Former NYT editor",
        constraints=["Maintain author voice", "Cut unnecessary words"],
        llm_fn=mock_llm)

    team = Team(
        [researcher, writer, editor],
        strategy="sequential",
    )

    store = team.run("Write a blog post about FlowForge")
    print(f"Graph: {team.describe()}")
    print(f"Researcher output: {getattr(store, 'trend_researcher', 'N/A')[:80]}...")
    print(f"Writer output: {getattr(store, 'content_writer', 'N/A')[:80]}...")
    print(f"Editor output: {getattr(store, 'style_editor', 'N/A')[:80]}...")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║  FlowForge Example 3: Content Creation Pipeline  ║")
    print("║  Research → Outline → Draft → Edit → SEO         ║")
    print("╚══════════════════════════════════════════════════╝")

    demo_pipeline()
    demo_team_approach()

    print("\n═══ All demos complete ═══")
