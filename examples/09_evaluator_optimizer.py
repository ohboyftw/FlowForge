"""
Example 09 — Evaluator-Optimizer Loop (Anthropic Pattern)

A code generator writes code, a reviewer evaluates it.
The loop continues until quality is sufficient or max rounds reached.
Uses flow.loop() primitive.

Works with mock LLM by default. Set OPENAI_API_KEY or ANTHROPIC_API_KEY
for real LLM calls via litellm.
"""

from flowforge import Flow, StoreBase, Unit


# ── Typed State ──
class CodeReviewState(StoreBase):
    task: str = ""
    code: str = ""
    feedback: str = ""
    quality: float = 0.0
    iteration: int = 0


# ── Units ──
class CodeGeneratorUnit(Unit):
    """Generates or improves code based on feedback."""

    def __init__(self, llm_fn):
        self.llm_fn = llm_fn

    def prep(self, store):
        return {
            "task": store.task,
            "previous_code": store.code,
            "feedback": store.feedback,
            "iteration": store.iteration,
        }

    def exec(self, ctx):
        if ctx["iteration"] == 0:
            prompt = f"Write code for: {ctx['task']}"
        else:
            prompt = (
                f"Improve this code based on feedback:\n"
                f"Code:\n{ctx['previous_code']}\n"
                f"Feedback: {ctx['feedback']}"
            )
        return self.llm_fn(
            system="You are a senior programmer. Write clean, tested code.",
            user=prompt,
        )

    def post(self, store, result):
        store.code = result
        store.iteration += 1
        return "default"


class CodeReviewerUnit(Unit):
    """Reviews code and scores quality."""

    def __init__(self, llm_fn):
        self.llm_fn = llm_fn

    def prep(self, store):
        return {"code": store.code, "iteration": store.iteration}

    def exec(self, ctx):
        return self.llm_fn(
            system="You are a code reviewer. Score quality 0-1 and give feedback.",
            user=f"Review this code (iteration {ctx['iteration']}):\n{ctx['code']}",
        )

    def post(self, store, result):
        store.feedback = result
        # Simulate increasing quality with each iteration
        store.quality = min(1.0, 0.3 * store.iteration)
        return "default"


# ── Mock LLM ──
def _mock_llm(system: str, user: str, **kwargs) -> str:
    if "programmer" in system.lower():
        if "improve" in user.lower():
            return "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)"
        return "def fibonacci(n):\n    # TODO: implement\n    pass"
    if "reviewer" in system.lower():
        return "Feedback: Add input validation and docstring. Score: 0.6"
    return f"Response: {user[:50]}"


def main():
    llm = _mock_llm

    # Build the eval-optimizer loop
    flow = Flow()
    flow.add("generate", CodeGeneratorUnit(llm))
    flow.add("review", CodeReviewerUnit(llm))
    flow.entry("generate")

    # Loop until quality >= 0.8 or 4 rounds max
    flow.loop("generate", "review", until=lambda s: s.quality >= 0.8, max_rounds=4)

    state = CodeReviewState(task="Write a fibonacci function with memoization")
    result = flow.run(state)

    print("=== Evaluator-Optimizer Loop Example ===")
    print(f"\nIterations: {result.iteration}")
    print(f"Final quality: {result.quality}")
    print(f"\nFinal code:\n{result.code}")
    print(f"\nLast feedback: {result.feedback}")
    print(f"\n--- Mermaid ---\n{flow.to_mermaid()}")


if __name__ == "__main__":
    main()
