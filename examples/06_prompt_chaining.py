"""
Example 06 — Prompt Chaining (Anthropic Pattern)

Pipeline: Outline → Draft → Edit
Each step feeds into the next via typed store fields.

Works with mock LLM by default. Set OPENAI_API_KEY or ANTHROPIC_API_KEY
for real LLM calls via litellm.
"""


from flowforge import Flow, StoreBase, Unit


# ── Typed State ──
class ArticleState(StoreBase):
    topic: str = ""
    outline: str = ""
    draft: str = ""
    edited: str = ""


# ── Mock LLM (deterministic for testing) ──
def _mock_llm(system: str, user: str, **kwargs) -> str:
    if "outline" in system.lower():
        return "1. Introduction\n2. Key Points\n3. Conclusion"
    if "draft" in system.lower():
        return "This is a well-structured article about the given topic..."
    if "edit" in system.lower():
        return "This is a polished, well-structured article about the given topic."
    return f"Response to: {user[:50]}"


def _get_llm():
    """Use real LLM if API key available, otherwise mock."""
    import os

    if os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import litellm

            model = "gpt-4o-mini" if os.environ.get("OPENAI_API_KEY") else "claude-sonnet-4-20250514"

            def _real_llm(system, user, **kw):
                r = litellm.completion(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                return r.choices[0].message.content

            return _real_llm
        except ImportError:
            pass
    return _mock_llm


# ── Units ──
class OutlineUnit(Unit):
    def __init__(self, llm_fn):
        self.llm_fn = llm_fn

    def prep(self, store):
        return store.topic

    def exec(self, topic):
        return self.llm_fn(
            system="You are an outline specialist. Create a clear outline.",
            user=f"Create an outline for an article about: {topic}",
        )

    def post(self, store, result):
        store.outline = result
        return "default"


class DraftUnit(Unit):
    def __init__(self, llm_fn):
        self.llm_fn = llm_fn

    def prep(self, store):
        return {"topic": store.topic, "outline": store.outline}

    def exec(self, ctx):
        return self.llm_fn(
            system="You are a draft writer. Write based on the outline.",
            user=f"Topic: {ctx['topic']}\nOutline:\n{ctx['outline']}\n\nWrite the article.",
        )

    def post(self, store, result):
        store.draft = result
        return "default"


class EditUnit(Unit):
    def __init__(self, llm_fn):
        self.llm_fn = llm_fn

    def prep(self, store):
        return store.draft

    def exec(self, draft):
        return self.llm_fn(
            system="You are an editor. Polish the text for clarity and flow.",
            user=f"Edit this article:\n{draft}",
        )

    def post(self, store, result):
        store.edited = result
        return "default"


def main():
    llm = _get_llm()

    # Build the chain: outline → draft → edit
    flow = Flow()
    flow.add("outline", OutlineUnit(llm))
    flow.add("draft", DraftUnit(llm))
    flow.add("edit", EditUnit(llm))
    flow.wire("outline", "draft").wire("draft", "edit")
    flow.entry("outline")

    state = ArticleState(topic="The Future of Agent Orchestration")
    result = flow.run(state)

    print("=== Prompt Chaining Example ===")
    print(f"\nTopic: {result.topic}")
    print(f"\nOutline:\n{result.outline}")
    print(f"\nDraft:\n{result.draft[:200]}...")
    print(f"\nEdited:\n{result.edited[:200]}...")
    print(f"\n--- Mermaid ---\n{flow.to_mermaid()}")


if __name__ == "__main__":
    main()
