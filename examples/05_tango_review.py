"""
FlowForge Example 5: Tango Code Review
═══════════════════════════════════════════════════

YOUR pattern, natively expressed in FlowForge.
The Tango pattern: dual-agent code review pairing two different
LLM reviewers for cross-pollination.

Shows: conditional loops, cross-wiring between agents,
checkpointing at each review round, and typed review state.
"""


from flowforge import (
    Agent, Team, StoreBase, Flow, Unit, FunctionUnit,
    Persona, Task, LLMUnit,
)
from pydantic import Field
from typing import Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Typed State
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ReviewFinding(StoreBase):
    """Individual review finding."""
    severity: str = "info"       # info / warning / error / critical
    category: str = ""           # bug / design / style / performance / security
    description: str = ""
    suggestion: str = ""
    file_path: str = ""
    line_number: int = 0


class ReviewState(StoreBase):
    """Typed state for the Tango review pipeline."""
    # Input
    code: str = ""
    language: str = "python"
    pr_description: str = ""

    # Review A (e.g., Claude)
    reviewer_a_findings: list[str] = Field(default_factory=list)
    reviewer_a_summary: str = ""
    reviewer_a_score: float = 0.0

    # Review B (e.g., GPT/Copilot)
    reviewer_b_findings: list[str] = Field(default_factory=list)
    reviewer_b_summary: str = ""
    reviewer_b_score: float = 0.0

    # Cross-review
    cross_review_notes: list[str] = Field(default_factory=list)
    consensus_items: list[str] = Field(default_factory=list)
    disputed_items: list[str] = Field(default_factory=list)

    # Final
    final_verdict: str = ""      # approve / request_changes / needs_discussion
    combined_score: float = 0.0
    review_rounds: int = 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Review Units
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ReviewerAUnit(Unit):
    """
    Reviewer A — focuses on correctness, bugs, and architecture.
    In production: Claude via Anthropic API.
    """
    def prep(self, store: ReviewState):
        return {
            "code": store.code,
            "language": store.language,
            "pr_desc": store.pr_description,
            # On second pass, include B's findings for cross-pollination
            "other_findings": store.reviewer_b_findings,
        }

    def exec(self, ctx: dict) -> dict:
        findings = [
            "[BUG] Missing null check on line 42 — could raise AttributeError",
            "[DESIGN] Function too long (>50 lines) — extract helper methods",
            "[SECURITY] User input not sanitized before SQL query",
        ]
        if ctx["other_findings"]:
            findings.append("[CROSS] Agree with Reviewer B on naming inconsistency")

        return {
            "findings": findings,
            "summary": "Found 3 issues: 1 critical (SQL injection), 1 bug, 1 design",
            "score": 0.6,  # 0-1, higher = better code
        }

    def post(self, store: ReviewState, result: dict) -> str:
        store.reviewer_a_findings = result["findings"]
        store.reviewer_a_summary = result["summary"]
        store.reviewer_a_score = result["score"]
        return "default"


class ReviewerBUnit(Unit):
    """
    Reviewer B — focuses on style, performance, and best practices.
    In production: GPT-4 / GitHub Copilot.
    """
    def prep(self, store: ReviewState):
        return {
            "code": store.code,
            "language": store.language,
            # On second pass, include A's findings
            "other_findings": store.reviewer_a_findings,
        }

    def exec(self, ctx: dict) -> dict:
        findings = [
            "[STYLE] Inconsistent naming: mix of camelCase and snake_case",
            "[PERF] List comprehension would be 3x faster than for-loop on line 28",
            "[PRACTICE] No docstring on public function `process_data`",
        ]
        if ctx["other_findings"]:
            findings.append("[CROSS] Confirm A's SQL injection finding — critical fix needed")

        return {
            "findings": findings,
            "summary": "Found 3 issues: style, performance, and documentation",
            "score": 0.7,
        }

    def post(self, store: ReviewState, result: dict) -> str:
        store.reviewer_b_findings = result["findings"]
        store.reviewer_b_summary = result["summary"]
        store.reviewer_b_score = result["score"]
        return "default"


class CrossReviewUnit(Unit):
    """
    Compares both reviews, finds consensus and disputes.
    The "Tango" step — where the two reviewers dance.
    """
    def prep(self, store: ReviewState):
        return {
            "a_findings": store.reviewer_a_findings,
            "b_findings": store.reviewer_b_findings,
            "a_score": store.reviewer_a_score,
            "b_score": store.reviewer_b_score,
        }

    def exec(self, ctx: dict) -> dict:
        # Find consensus (items both flagged or one confirmed)
        consensus = [f for f in ctx["a_findings"] + ctx["b_findings"]
                     if "[CROSS]" in f or "[SECURITY]" in f or "[BUG]" in f]

        # Items only one reviewer found
        all_findings = set(ctx["a_findings"] + ctx["b_findings"])
        disputed = [f for f in all_findings if "[STYLE]" in f]

        avg_score = (ctx["a_score"] + ctx["b_score"]) / 2
        has_critical = any("[SECURITY]" in f or "[BUG]" in f
                         for f in ctx["a_findings"] + ctx["b_findings"])

        if has_critical:
            verdict = "request_changes"
        elif avg_score >= 0.8:
            verdict = "approve"
        else:
            verdict = "needs_discussion"

        return {
            "consensus": consensus,
            "disputed": disputed,
            "verdict": verdict,
            "score": avg_score,
        }

    def post(self, store: ReviewState, result: dict) -> str:
        store.consensus_items = result["consensus"]
        store.disputed_items = result["disputed"]
        store.final_verdict = result["verdict"]
        store.combined_score = result["score"]
        store.review_rounds += 1

        # Checkpoint after each round
        store.checkpoint(f"round_{store.review_rounds}")

        if result["verdict"] == "needs_discussion" and store.review_rounds < 2:
            return "second_pass"
        return "done"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Build the Tango flow
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_tango_flow() -> Flow:
    """
    Tango Review Flow:
    
        reviewer_a → reviewer_b → cross_review
                                      │
                             ┌────────┤
                    (done)   │  (second_pass)
                             │        │
                             ▼        ▼
                           [end]   reviewer_a (with B's context)
                                     → reviewer_b (with A's context)
                                       → cross_review (final)
    """
    flow = Flow()

    flow.add("reviewer_a", ReviewerAUnit())
    flow.add("reviewer_b", ReviewerBUnit())
    flow.add("cross_review", CrossReviewUnit())

    # First pass: A → B → cross-review
    flow.wire("reviewer_a", "reviewer_b")
    flow.wire("reviewer_b", "cross_review")

    # Second pass (cross-pollination): if needs_discussion,
    # loop back so each reviewer can see the other's findings
    flow.wire("cross_review", "reviewer_a", on="second_pass")

    flow.entry("reviewer_a")
    return flow


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Demo
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def demo_tango():
    print("\n── Tango Code Review ──")

    sample_code = '''
def process_data(user_input):
    db = get_database()
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    results = db.execute(query)
    processedItems = []
    for item in results:
        if item.status == "active":
            processedItems.append(item)
    return processedItems
'''.strip()

    flow = build_tango_flow()

    state = ReviewState(
        code=sample_code,
        language="python",
        pr_description="Add user data processing function",
    )

    result = flow.run(state)

    print(f"\n🔍 Code Review Report")
    print(f"{'═' * 55}")

    print(f"\n📋 Reviewer A (Architecture/Bugs):")
    for f in result.reviewer_a_findings:
        print(f"  {f}")
    print(f"  Score: {result.reviewer_a_score:.0%}")

    print(f"\n📋 Reviewer B (Style/Performance):")
    for f in result.reviewer_b_findings:
        print(f"  {f}")
    print(f"  Score: {result.reviewer_b_score:.0%}")

    print(f"\n🤝 Cross-Review Consensus:")
    for item in result.consensus_items:
        print(f"  ✓ {item}")

    if result.disputed_items:
        print(f"\n⚡ Disputed (style preference):")
        for item in result.disputed_items:
            print(f"  ? {item}")

    print(f"\n{'═' * 55}")
    print(f"Verdict: {result.final_verdict.upper()}")
    print(f"Combined Score: {result.combined_score:.0%}")
    print(f"Review Rounds: {result.review_rounds}")
    print(f"Checkpoints: {result.list_checkpoints()}")

    print(f"\nExecution trace ({len(flow.trace)} steps):")
    for step in flow.trace:
        print(f"  {step['step']}. {step['unit']} → '{step['action']}'")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║  FlowForge Example 5: Tango Code Review          ║")
    print("║  Dual-agent cross-pollination review              ║")
    print("╚══════════════════════════════════════════════════╝")

    demo_tango()

    print("\n═══ Demo complete ═══")
