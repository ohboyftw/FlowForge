"""
FlowForge — Identity Layer
CrewAI-grade role semantics as Pydantic models.

Persona and Task are themselves Pydantic BaseModels, so they:
  - Validate on construction
  - Serialize to JSON (for logging, persistence, sharing)
  - Have schema introspection
  - Can be composed into typed collections
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Persona(BaseModel):
    """
    Agent identity — who this agent is and what it does.

    Maps to CrewAI's Agent(role, goal, backstory) but as a
    clean Pydantic model that compiles to a system prompt.

    Usage:
        researcher = Persona(
            role="Senior ML Researcher",
            goal="Find state-of-the-art papers on agentic AI",
            backstory="PhD from MIT, 10 years at DeepMind",
            constraints=["Always cite sources", "Prefer 2024+ papers"],
            delegation=True,
        )

        # Use as system prompt
        prompt = researcher.to_prompt()

        # Serialize for logging
        researcher.model_dump_json()
    """

    model_config = ConfigDict(frozen=False)

    role: str = Field(..., description="Agent's role title")
    goal: str = Field(..., description="Primary objective")
    backstory: str = Field(default="", description="Background context that shapes behavior")
    constraints: list[str] = Field(
        default_factory=list, description="Hard constraints the agent must follow"
    )
    tools_description: list[str] = Field(
        default_factory=list, description="Human-readable tool descriptions for prompt context"
    )
    delegation: bool = Field(default=True, description="Whether this agent can delegate to others")
    verbose: bool = Field(
        default=False, description="Whether to include chain-of-thought instructions"
    )

    def to_prompt(self) -> str:
        """Compile Persona to a system prompt string."""
        parts = [f"You are a {self.role}."]
        parts.append(f"Your goal: {self.goal}")

        if self.backstory:
            parts.append(f"Background: {self.backstory}")

        if self.constraints:
            parts.append("Constraints:")
            for c in self.constraints:
                parts.append(f"  - {c}")

        if self.tools_description:
            parts.append("Available tools:")
            for t in self.tools_description:
                parts.append(f"  - {t}")

        if self.delegation:
            parts.append("You may delegate sub-tasks to other team members when appropriate.")

        if self.verbose:
            parts.append("Think step by step. Show your reasoning before giving a final answer.")

        return "\n".join(parts)

    def with_overrides(self, **kwargs) -> Persona:
        """Create a new Persona with some fields overridden."""
        data = self.model_dump()
        data.update(kwargs)
        return Persona(**data)


class Task(BaseModel):
    """
    A unit of work with clear inputs, outputs, and context dependencies.

    Maps to CrewAI's Task but adds typed context_from for
    explicit data flow between tasks/agents.

    Usage:
        task = Task(
            description="Research recent papers on LLM agents",
            expected_output="List of 5 papers with summaries",
            context_from=["user_query", "previous_findings"],
        )

        # Compile to user prompt with context injection
        prompt = task.to_prompt(context={
            "user_query": "agentic AI frameworks",
            "previous_findings": ["PocketFlow", "LangGraph"],
        })
    """

    model_config = ConfigDict(frozen=False)

    description: str = Field(..., description="What needs to be done")
    expected_output: str = Field(default="", description="What the output should look like")
    context_from: list[str] = Field(
        default_factory=list, description="Store field names to inject as context"
    )
    output_field: str = Field(default="", description="Store field name to write results to")
    max_retries: int = Field(default=3, description="Max retry attempts on failure")

    def to_prompt(self, context: dict = None) -> str:
        """Compile Task to a user prompt string with injected context."""
        parts = [f"Task: {self.description}"]

        if self.expected_output:
            parts.append(f"Expected output: {self.expected_output}")

        if context:
            injected = {
                k: v for k, v in context.items() if k in self.context_from and v is not None
            }
            if injected:
                parts.append("\nContext:")
                for k, v in injected.items():
                    # Truncate long context for prompt efficiency
                    v_str = str(v)
                    if len(v_str) > 2000:
                        v_str = v_str[:2000] + "... [truncated]"
                    parts.append(f"  [{k}]: {v_str}")

        return "\n".join(parts)


class TaskResult(BaseModel):
    """
    Structured output from a Task execution.

    Captures the result, metadata, and quality signals
    for downstream consumption and evaluation.
    """

    task_description: str
    output: Any
    persona_role: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    tokens_used: int = 0
    latency_ms: float = 0.0
    attempt: int = 1
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Persona presets for common roles
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class Personas:
    """Factory for common agent personas. Customize via .with_overrides()."""

    @staticmethod
    def researcher(**overrides) -> Persona:
        return Persona(
            role="Senior Researcher",
            goal="Find accurate, comprehensive information",
            backstory="Expert at web research, paper analysis, and data synthesis",
            constraints=["Always cite sources", "Flag uncertainty"],
            **overrides,
        )

    @staticmethod
    def coder(**overrides) -> Persona:
        return Persona(
            role="Senior Software Engineer",
            goal="Write clean, tested, production-ready code",
            backstory="15 years of experience across multiple languages and paradigms",
            constraints=["Include error handling", "Write docstrings", "Follow existing patterns"],
            **overrides,
        )

    @staticmethod
    def reviewer(**overrides) -> Persona:
        return Persona(
            role="Code Reviewer",
            goal="Find bugs, design issues, and improvement opportunities",
            backstory="Expert at spotting edge cases and architectural problems",
            constraints=[
                "Be specific",
                "Suggest fixes, not just problems",
                "Prioritize by severity",
            ],
            **overrides,
        )

    @staticmethod
    def manager(**overrides) -> Persona:
        return Persona(
            role="Technical Lead",
            goal="Decompose complex tasks and coordinate team execution",
            backstory="20 years as architect, experienced at task decomposition and delegation",
            constraints=["Break tasks into clear subtasks", "Assign based on expertise"],
            delegation=True,
            **overrides,
        )

    @staticmethod
    def analyst(**overrides) -> Persona:
        return Persona(
            role="Data Analyst",
            goal="Extract insights from data with statistical rigor",
            backstory="Strong background in statistics, visualization, and business intelligence",
            constraints=["Quantify uncertainty", "Use appropriate visualizations"],
            **overrides,
        )
