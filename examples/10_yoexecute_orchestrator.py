"""
Example 10 — YoExecute PM Orchestrator (Real-World Use Case)

Models YoExecute's autonomous PM workflow using FlowForge patterns:

1. **Prompt Chaining**: Triage → Skill Selection → Prompt Rendering
2. **Orchestrator-Workers**: Architect decomposes issue → parallel subtask agents
3. **Evaluator-Optimizer**: Quality gate loop on deliverables
4. **Parallelization**: Multiple subtasks execute concurrently with typed reducers

This example shows how FlowForge can prototype the same topology that
YoExecute implements in production with Elixir/OTP + Claude Code.

Works with mock LLM by default.
"""

from pydantic import Field

from flowforge import Flow, FunctionUnit, StoreBase, Unit

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Typed State — mirrors YoExecute's issue lifecycle
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class IssueState(StoreBase):
    """State for a single issue flowing through the PM pipeline."""

    __reducers__ = {
        "subtask_outputs": "extend",
        "review_notes": "extend",
    }

    # Issue metadata (from tracker)
    issue_id: str = ""
    title: str = ""
    description: str = ""
    labels: list[str] = Field(default_factory=list)

    # Triage output
    execution_mode: str = ""  # "direct" or "dispatch-bare"
    matched_skills: list[str] = Field(default_factory=list)

    # Prompt rendering
    rendered_prompt: str = ""

    # Decomposition (Mode C only)
    subtasks: list[dict] = Field(default_factory=list)
    subtask_outputs: list[str] = Field(default_factory=list)

    # Agent output
    deliverable: str = ""
    quality_score: float = 0.0
    review_notes: list[str] = Field(default_factory=list)
    attempt: int = 0

    # Terminal
    final_comment: str = ""
    terminal_state: str = ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mock LLM — simulates Claude Code responses
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _mock_llm(system: str, user: str, **kwargs) -> str:
    """Deterministic mock that simulates different agent personas."""
    u = user.lower()
    if "decompose" in u or "subtask" in u:
        return (
            "Subtask 1: Research competitive landscape\n"
            "Subtask 2: Define target personas\n"
            "Subtask 3: Draft value propositions"
        )
    if "research" in u or "competitive" in u:
        return "Competitive analysis: 3 direct competitors identified. Key differentiator: unified API."
    if "persona" in u:
        return (
            "Target personas: (1) Solo dev prototyping, (2) Team lead evaluating, (3) Platform eng."
        )
    if "value prop" in u:
        return "Value prop: Learn agent patterns with FlowForge, ship with any framework."
    if "review" in u or "quality" in u:
        return "Review: Comprehensive coverage. Score: 0.85. Minor gap: missing pricing analysis."
    if "synthesize" in u or "final" in u:
        return (
            "# PRD: FlowForge\n\n"
            "## Problem\nAgent orchestration is fragmented.\n\n"
            "## Solution\nUnified learning framework.\n\n"
            "## Personas\n1. Solo devs  2. Team leads  3. Platform engineers\n\n"
            "## Competitive Edge\nOnly framework focused on pattern education."
        )
    return f"[agent] {user[:80]}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Units — each maps to a YoExecute pipeline stage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# ── Stage 1: Triage (label → mode + skills) ──


class TriageUnit(Unit):
    """Maps issue labels to execution mode and skills (like YoExecute's ModeSelector)."""

    SKILL_MAP = {
        "pm-prd": ["pm-execution/write-prd"],
        "pm-gtm": ["pm-go-to-market/launch-plan"],
        "pm-discovery": ["pm-product-discovery/user-research", "pm-product-discovery/competitive"],
        "eng-feature": ["eng-execution/implement"],
    }

    def prep(self, store):
        return store.labels

    def exec(self, labels):
        skills = []
        for label in labels:
            skills.extend(self.SKILL_MAP.get(label, []))

        # Mode selection: multi-skill issues → dispatch-bare, single → direct
        mode = "dispatch-bare" if len(skills) > 1 else "direct"
        return {"mode": mode, "skills": skills}

    def post(self, store, result):
        store.execution_mode = result["mode"]
        store.matched_skills = result["skills"]
        return store.execution_mode  # route to different branches


# ── Stage 2: Prompt Rendering (like YoExecute's Liquid templates) ──


class RenderPromptUnit(Unit):
    """Renders the agent prompt from issue context + skills."""

    def prep(self, store):
        return {
            "issue_id": store.issue_id,
            "title": store.title,
            "description": store.description,
            "skills": store.matched_skills,
            "attempt": store.attempt,
        }

    def exec(self, ctx):
        skills_section = "\n".join(f"  - {s}" for s in ctx["skills"])
        prompt = (
            f"Issue: {ctx['issue_id']} — {ctx['title']}\n"
            f"Description: {ctx['description']}\n"
            f"Skills loaded:\n{skills_section}\n"
        )
        if ctx["attempt"] > 0:
            prompt += (
                f"\nThis is retry attempt #{ctx['attempt']}. Address previous review feedback.\n"
            )
        return prompt

    def post(self, store, result):
        store.rendered_prompt = result
        return "default"


# ── Stage 3a: Direct Execution (Mode A — single agent) ──


class DirectAgentUnit(Unit):
    """Single agent executes the full task (YoExecute Mode A)."""

    def __init__(self, llm_fn):
        self.llm_fn = llm_fn

    def prep(self, store):
        return store.rendered_prompt

    def exec(self, prompt):
        return self.llm_fn(
            system="You are a PM agent. Execute the task described below.",
            user=prompt,
        )

    def post(self, store, result):
        store.deliverable = result
        store.attempt += 1
        return "default"


# ── Stage 3b: Decompose (Mode C — architect splits into subtasks) ──


class DecomposeUnit(Unit):
    """Architect agent decomposes issue into subtasks (YoExecute Mode C)."""

    def __init__(self, llm_fn):
        self.llm_fn = llm_fn

    def prep(self, store):
        return {"prompt": store.rendered_prompt, "skills": store.matched_skills}

    def exec(self, ctx):
        response = self.llm_fn(
            system="You are a PM architect. Decompose this into subtasks.",
            user=f"Decompose into subtasks:\n{ctx['prompt']}",
        )
        # Parse response into subtask list
        subtasks = []
        for i, line in enumerate(response.strip().split("\n")):
            if line.strip():
                subtasks.append(
                    {
                        "id": f"subtask-{i + 1}",
                        "description": line.strip(),
                        "skill_ref": ctx["skills"][i] if i < len(ctx["skills"]) else None,
                    }
                )
        return subtasks

    def post(self, store, result):
        store.subtasks = result
        return "default"


# ── Stage 3c: Parallel subtask workers ──


class SubtaskWorkerUnit(Unit):
    """Executes a single subtask (runs in parallel during fan-out)."""

    def __init__(self, subtask_index: int, llm_fn):
        self.subtask_index = subtask_index
        self.llm_fn = llm_fn

    def prep(self, store):
        if self.subtask_index < len(store.subtasks):
            return store.subtasks[self.subtask_index]
        return {"description": "No subtask assigned"}

    def exec(self, subtask):
        return self.llm_fn(
            system="You are a specialist PM agent. Complete your assigned subtask.",
            user=f"Execute: {subtask['description']}",
        )

    def post(self, store, result):
        store.subtask_outputs = [f"[{self.subtask_index}] {result}"]
        return "default"


# ── Stage 3d: Synthesize subtask outputs ──


class SynthesizeUnit(Unit):
    """Aggregates parallel subtask outputs into final deliverable."""

    def __init__(self, llm_fn):
        self.llm_fn = llm_fn

    def prep(self, store):
        return {
            "title": store.title,
            "outputs": store.subtask_outputs,
        }

    def exec(self, ctx):
        outputs_text = "\n\n".join(ctx["outputs"])
        return self.llm_fn(
            system="You are a PM lead. Synthesize subtask outputs into a final deliverable.",
            user=f"Synthesize outputs for '{ctx['title']}':\n\n{outputs_text}",
        )

    def post(self, store, result):
        store.deliverable = result
        store.attempt += 1
        return "default"


# ── Stage 4: Quality Review (evaluator in eval-optimizer loop) ──


class QualityReviewUnit(Unit):
    """Reviews deliverable quality (evaluator in the loop)."""

    def __init__(self, llm_fn):
        self.llm_fn = llm_fn

    def prep(self, store):
        return {"deliverable": store.deliverable, "title": store.title}

    def exec(self, ctx):
        response = self.llm_fn(
            system="You are a quality reviewer. Score 0-1 and give feedback.",
            user=f"Review this deliverable for '{ctx['title']}':\n{ctx['deliverable']}",
        )
        # Parse score from response
        score = 0.85 if "0.85" in response else 0.5 + (0.15 * len(response) / 100)
        return {"feedback": response, "score": min(score, 1.0)}

    def post(self, store, result):
        store.quality_score = result["score"]
        store.review_notes = [result["feedback"]]
        return "default"


# ── Stage 5: Post results back to tracker ──


class PostResultUnit(Unit):
    """Posts deliverable as comment and updates issue state (like YoExecute's tracker update)."""

    def prep(self, store):
        return {
            "issue_id": store.issue_id,
            "deliverable": store.deliverable,
            "quality": store.quality_score,
            "attempts": store.attempt,
        }

    def exec(self, ctx):
        comment = (
            f"## Deliverable for {ctx['issue_id']}\n\n"
            f"{ctx['deliverable']}\n\n"
            f"---\n"
            f"Quality: {ctx['quality']:.0%} | Attempts: {ctx['attempts']}"
        )
        return {"comment": comment, "state": "Done"}

    def post(self, store, result):
        store.final_comment = result["comment"]
        store.terminal_state = result["state"]
        return "default"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Flow Assembly — wires up the full pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_direct_flow(llm_fn) -> Flow:
    """Mode A: triage → render → agent → review loop → post."""
    flow = Flow()
    flow.add("triage", TriageUnit())
    flow.add("render", RenderPromptUnit())
    flow.add("agent", DirectAgentUnit(llm_fn))
    flow.add("review", QualityReviewUnit(llm_fn))
    flow.add("post", PostResultUnit())

    flow.wire("triage", "render", on="direct")
    flow.wire("render", "agent")
    flow.entry("triage")

    # Eval-optimizer loop: agent → review, loop until quality >= 0.8
    flow.loop("agent", "review", until=lambda s: s.quality_score >= 0.8, max_rounds=3)

    # After loop exits, post results
    loop_end = [n for n in flow.nodes if n.startswith("_loop_end_")][0]
    flow.wire(loop_end, "post")

    return flow


def build_dispatch_flow(llm_fn, num_workers: int = 3) -> Flow:
    """Mode C: triage → render → decompose → parallel workers → synthesize → review → post.

    Fan-out terminates the branch in FlowForge, so we use a two-phase approach:
    Phase 1 (fan-out flow): dispatch → parallel workers (concurrent, merged via reducers)
    Phase 2 (main flow): decompose runs the fan-out internally, then continues to synthesize.
    """
    flow = Flow()
    flow.add("triage", TriageUnit())
    flow.add("render", RenderPromptUnit())
    flow.add("decompose", DecomposeUnit(llm_fn))

    # Parallel dispatch: a unit that internally fans out to workers
    class ParallelDispatchUnit(Unit):
        """Runs subtask workers in parallel using a nested Flow fan-out."""

        def __init__(self, llm_fn, num_workers):
            self._llm_fn = llm_fn
            self._num_workers = num_workers

        def prep(self, store):
            return store.subtasks

        def exec(self, subtasks):
            return subtasks  # pass through

        def post(self, store, subtasks):
            # Build a mini fan-out flow for parallel execution
            fanout = Flow()
            fanout.add("go", FunctionUnit(lambda s: "default"))
            worker_names = []
            for i in range(min(self._num_workers, len(subtasks))):
                name = f"w{i}"
                fanout.add(name, SubtaskWorkerUnit(i, self._llm_fn))
                worker_names.append(name)
            fanout.wire("go", worker_names)
            fanout.entry("go")
            fanout.run(store)
            return "default"

    flow.add("parallel", ParallelDispatchUnit(llm_fn, num_workers))
    flow.add("synthesize", SynthesizeUnit(llm_fn))
    flow.add("review", QualityReviewUnit(llm_fn))
    flow.add("post", PostResultUnit())

    flow.wire("triage", "render", on="dispatch-bare")
    flow.wire("render", "decompose")
    flow.wire("decompose", "parallel")
    flow.wire("parallel", "synthesize")
    flow.wire("synthesize", "review")
    flow.wire("review", "post")

    flow.entry("triage")
    return flow


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main — run both modes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_direct_mode():
    """Test Mode A: single-agent direct execution."""
    print("=" * 60)
    print("MODE A: Direct Execution (single agent)")
    print("=" * 60)

    flow = build_direct_flow(_mock_llm)
    state = IssueState(
        issue_id="OHBOY-42",
        title="Write launch blog post",
        description="Draft a blog post announcing FlowForge v0.1",
        labels=["eng-feature"],
    )

    result = flow.run(state)

    print(f"\nIssue: {result.issue_id} — {result.title}")
    print(f"Mode: {result.execution_mode}")
    print(f"Skills: {result.matched_skills}")
    print(f"Attempts: {result.attempt}")
    print(f"Quality: {result.quality_score:.0%}")
    print(f"\nDeliverable:\n{result.deliverable[:200]}...")
    print(f"\n--- Mermaid ---\n{flow.to_mermaid()}")


def run_dispatch_mode():
    """Test Mode C: architect decompose → parallel workers → synthesize."""
    print("\n" + "=" * 60)
    print("MODE C: Dispatch-Bare (architect + parallel workers)")
    print("=" * 60)

    flow = build_dispatch_flow(_mock_llm, num_workers=3)
    state = IssueState(
        issue_id="OHBOY-99",
        title="Write PRD for FlowForge",
        description="Create a comprehensive PRD covering market, personas, and positioning",
        labels=["pm-discovery"],
    )

    result = flow.run(state)

    print(f"\nIssue: {result.issue_id} — {result.title}")
    print(f"Mode: {result.execution_mode}")
    print(f"Skills: {result.matched_skills}")
    print(f"Subtasks: {len(result.subtasks)}")
    print(f"Subtask outputs: {len(result.subtask_outputs)}")
    for out in result.subtask_outputs:
        print(f"  {out[:80]}")
    print(f"Quality: {result.quality_score:.0%}")
    print(f"\nFinal deliverable:\n{result.deliverable[:300]}...")
    print(f"\nTracker comment:\n{result.final_comment[:200]}...")
    print(f"Terminal state: {result.terminal_state}")
    print(f"\n--- Mermaid ---\n{flow.to_mermaid()}")


def main():
    run_direct_mode()
    run_dispatch_mode()


if __name__ == "__main__":
    main()
