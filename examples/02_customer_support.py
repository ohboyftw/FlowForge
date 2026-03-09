"""
FlowForge Example 2: Customer Support Bot with Routing
═══════════════════════════════════════════════════════════

THE showcase for graph-based orchestration. Every framework demos this:
  - LangGraph: Customer support with flight booking, refunds, escalation
  - OpenAI Agents SDK: Triage → specialist handoffs
  - CrewAI: Support crew with specialist agents

Shows FlowForge's conditional routing, human-in-the-loop interrupts,
and typed state for conversation management.
"""


from flowforge import (
    Agent, Team, StoreBase, Flow, Unit, FunctionUnit,
    Persona, Task, LLMUnit, InterruptSignal,
)
from pydantic import Field
from typing import Optional
from enum import Enum


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Typed State — The conversation's memory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TicketCategory(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    REFUND = "refund"
    GENERAL = "general"
    ESCALATE = "escalate"


class SupportState(StoreBase):
    """Typed state for customer support flow."""
    # Input
    customer_message: str = ""
    customer_id: str = ""
    order_id: str = ""

    # Triage
    category: str = ""           # TicketCategory value
    sentiment: str = "neutral"   # positive / neutral / negative / angry
    urgency: str = "normal"      # low / normal / high / critical

    # Processing
    resolution: str = ""
    refund_amount: float = 0.0
    refund_approved: bool = False
    needs_human: bool = False

    # Conversation
    messages: list[str] = Field(default_factory=list)
    internal_notes: list[str] = Field(default_factory=list)

    # Metadata
    steps_taken: list[str] = Field(default_factory=list)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Units — Each handles one step in the support flow
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TriageUnit(Unit):
    """
    Classifies the customer message into a category.
    This is what LangGraph does with conditional edges —
    the return action routes to the appropriate specialist.
    """
    def prep(self, store: SupportState):
        return store.customer_message.lower()

    def exec(self, message: str) -> dict:
        # In production: LLM classification
        # Here: keyword-based for demonstration
        if any(w in message for w in ["refund", "money back", "charge"]):
            return {"category": "refund", "sentiment": "negative", "urgency": "high"}
        elif any(w in message for w in ["broken", "error", "crash", "bug"]):
            return {"category": "technical", "sentiment": "negative", "urgency": "normal"}
        elif any(w in message for w in ["bill", "invoice", "payment", "subscription"]):
            return {"category": "billing", "sentiment": "neutral", "urgency": "normal"}
        elif any(w in message for w in ["angry", "worst", "terrible", "lawsuit"]):
            return {"category": "escalate", "sentiment": "angry", "urgency": "critical"}
        else:
            return {"category": "general", "sentiment": "neutral", "urgency": "low"}

    def post(self, store: SupportState, result: dict) -> str:
        store.category = result["category"]
        store.sentiment = result["sentiment"]
        store.urgency = result["urgency"]
        store.steps_taken = store.steps_taken + [f"triage → {result['category']}"]

        # Return action = category name → routes to the right specialist
        return result["category"]


class BillingUnit(Unit):
    """Handles billing inquiries."""
    def prep(self, store: SupportState):
        return {"message": store.customer_message, "customer_id": store.customer_id}

    def exec(self, ctx: dict) -> str:
        # In production: look up billing system
        return (
            f"I've checked your account (customer {ctx['customer_id']}). "
            "Your current billing cycle ends on the 15th. "
            "You can view invoices in your account dashboard."
        )

    def post(self, store: SupportState, resolution: str) -> str:
        store.resolution = resolution
        store.messages = store.messages + [f"Agent: {resolution}"]
        store.steps_taken = store.steps_taken + ["billing_resolved"]
        return "resolved"


class TechnicalUnit(Unit):
    """Handles technical support."""
    def prep(self, store: SupportState):
        return store.customer_message

    def exec(self, message: str) -> str:
        return (
            "I understand you're experiencing a technical issue. "
            "Here are some steps to try:\n"
            "1. Clear your browser cache\n"
            "2. Try incognito/private mode\n"
            "3. Check our status page at status.example.com\n"
            "If the issue persists, I'll escalate to our engineering team."
        )

    def post(self, store: SupportState, resolution: str) -> str:
        store.resolution = resolution
        store.messages = store.messages + [f"Agent: {resolution}"]
        store.steps_taken = store.steps_taken + ["technical_resolved"]
        return "resolved"


class RefundUnit(Unit):
    """
    Handles refund requests.
    Demonstrates human-in-the-loop: refunds above a threshold
    need manager approval (via interrupt wire).
    """
    def prep(self, store: SupportState):
        return {"message": store.customer_message, "order_id": store.order_id}

    def exec(self, ctx: dict) -> dict:
        # In production: look up order, calculate refund
        # Simulate: extract amount or default
        amount = 49.99  # would come from order lookup
        needs_approval = amount > 25.00  # policy: > $25 needs human
        return {
            "amount": amount,
            "needs_approval": needs_approval,
            "reason": "Product defect reported by customer",
        }

    def post(self, store: SupportState, result: dict) -> str:
        store.refund_amount = result["amount"]
        store.internal_notes = store.internal_notes + [
            f"Refund ${result['amount']:.2f} — {result['reason']}"
        ]
        store.steps_taken = store.steps_taken + [f"refund_calculated: ${result['amount']:.2f}"]

        if result["needs_approval"]:
            store.needs_human = True
            return "needs_approval"
        else:
            store.refund_approved = True
            store.resolution = f"Refund of ${result['amount']:.2f} processed automatically."
            return "resolved"


class RefundApprovalUnit(Unit):
    """Post-approval: process the refund after human approves."""
    def prep(self, store: SupportState):
        return store.refund_amount

    def exec(self, amount: float) -> str:
        return f"Refund of ${amount:.2f} has been approved and processed."

    def post(self, store: SupportState, resolution: str) -> str:
        store.refund_approved = True
        store.resolution = resolution
        store.messages = store.messages + [f"Agent: {resolution}"]
        store.steps_taken = store.steps_taken + ["refund_approved_by_human"]
        return "resolved"


class EscalationUnit(Unit):
    """Escalates to human agent for angry/complex cases."""
    def prep(self, store: SupportState):
        return {
            "category": store.category,
            "sentiment": store.sentiment,
            "message": store.customer_message,
        }

    def exec(self, ctx: dict) -> str:
        return (
            "I understand this is frustrating, and I want to make sure "
            "you get the best possible help. I'm connecting you with a "
            "senior support specialist who can assist you directly."
        )

    def post(self, store: SupportState, resolution: str) -> str:
        store.resolution = resolution
        store.needs_human = True
        store.messages = store.messages + [f"Agent: {resolution}"]
        store.steps_taken = store.steps_taken + ["escalated_to_human"]
        return "resolved"


class WrapUpUnit(Unit):
    """Final step: summarize and close the ticket."""
    def post(self, store: SupportState, _) -> str:
        summary = (
            f"Ticket resolved via {store.category} path. "
            f"Steps: {' → '.join(store.steps_taken)}"
        )
        store.internal_notes = store.internal_notes + [summary]
        return "default"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Build the support flow graph
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_support_flow() -> Flow:
    """
    Builds the customer support graph:
    
        triage ──→ billing ────→ wrap_up
          │──→ technical ──→ wrap_up
          │──→ refund ──→ [interrupt if >$25] → approve → wrap_up
          │──→ escalation → wrap_up
          └──→ billing (default/general)
    """
    flow = Flow()

    # Add all nodes
    flow.add("triage", TriageUnit())
    flow.add("billing", BillingUnit())
    flow.add("technical", TechnicalUnit())
    flow.add("refund", RefundUnit())
    flow.add("approve_refund", RefundApprovalUnit())
    flow.add("escalation", EscalationUnit())
    flow.add("wrap_up", WrapUpUnit())

    # Triage routes to specialists based on category (action label)
    flow.wire("triage", "billing", on="billing")
    flow.wire("triage", "technical", on="technical")
    flow.wire("triage", "refund", on="refund")
    flow.wire("triage", "escalation", on="escalate")
    flow.wire("triage", "billing", on="general")  # default → billing

    # Specialists resolve → wrap_up
    flow.wire("billing", "wrap_up", on="resolved")
    flow.wire("technical", "wrap_up", on="resolved")
    flow.wire("escalation", "wrap_up", on="resolved")

    # Refund: auto-approve small amounts → wrap_up
    flow.wire("refund", "wrap_up", on="resolved")

    # Refund: large amounts need human approval (INTERRUPT!)
    flow.wire("refund", "approve_refund",
              on="needs_approval",
              interrupt=True)  # ← This pauses for human review

    flow.wire("approve_refund", "wrap_up", on="resolved")

    flow.entry("triage")
    return flow


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Demo scenarios
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def demo_billing():
    """Billing inquiry — straight-through resolution."""
    print("\n── Scenario: Billing Inquiry ──")
    flow = build_support_flow()
    state = SupportState(
        customer_message="When is my next invoice due?",
        customer_id="CUST-42",
    )
    result = flow.run(state)
    print(f"Category: {result.category}")
    print(f"Resolution: {result.resolution}")
    print(f"Steps: {' → '.join(result.steps_taken)}")


def demo_technical():
    """Technical issue — troubleshooting resolution."""
    print("\n── Scenario: Technical Issue ──")
    flow = build_support_flow()
    state = SupportState(
        customer_message="The app keeps crashing when I try to upload",
        customer_id="CUST-99",
    )
    result = flow.run(state)
    print(f"Category: {result.category}")
    print(f"Resolution: {result.resolution[:100]}...")
    print(f"Steps: {' → '.join(result.steps_taken)}")


def demo_refund_with_interrupt():
    """
    Refund request — demonstrates human-in-the-loop.
    The flow PAUSES at the approval step for human review.
    """
    print("\n── Scenario: Refund (with human approval) ──")
    flow = build_support_flow()
    state = SupportState(
        customer_message="I want a refund for my broken item",
        customer_id="CUST-77",
        order_id="ORD-1234",
    )

    try:
        result = flow.run(state)
        print("(Should not reach here — interrupt expected)")
    except InterruptSignal as sig:
        print(f"⏸  Flow paused at: {sig.from_node}")
        print(f"   Refund amount: ${sig.store.refund_amount:.2f}")
        print(f"   Needs human: {sig.store.needs_human}")
        print(f"   Steps so far: {' → '.join(sig.store.steps_taken)}")

        # Simulate human approval
        print("\n   👤 Manager reviews... APPROVED")
        sig.store.checkpoint("pre_approval")

        # Resume flow from the approval node
        flow_resumed = build_support_flow()
        # Manually run the approval + wrap_up
        RefundApprovalUnit().run(sig.store)
        WrapUpUnit().run(sig.store)

        print(f"\n   Resolution: {sig.store.resolution}")
        print(f"   Final steps: {' → '.join(sig.store.steps_taken)}")


def demo_escalation():
    """Angry customer — auto-escalates to human."""
    print("\n── Scenario: Angry Customer (escalation) ──")
    flow = build_support_flow()
    state = SupportState(
        customer_message="This is terrible service! I want to speak to a manager!",
        customer_id="CUST-13",
    )
    result = flow.run(state)
    print(f"Category: {result.category}")
    print(f"Sentiment: {result.sentiment}")
    print(f"Urgency: {result.urgency}")
    print(f"Needs human: {result.needs_human}")
    print(f"Resolution: {result.resolution[:100]}...")
    print(f"Steps: {' → '.join(result.steps_taken)}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run all demos
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║  FlowForge Example 2: Customer Support Bot       ║")
    print("║  Routing, specialists, human-in-the-loop         ║")
    print("╚══════════════════════════════════════════════════╝")

    flow = build_support_flow()
    print(f"\nGraph structure:\n{flow.describe()}")

    demo_billing()
    demo_technical()
    demo_refund_with_interrupt()
    demo_escalation()

    print("\n═══ All scenarios complete ═══")
