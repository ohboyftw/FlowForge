"""
FlowForge Example 4: Stock Analysis (Parallel Team)
═══════════════════════════════════════════════════════

THE parallel/multi-specialist example:
  - CrewAI: "Stock analysis crew" (their most popular example)
  - Agno: "web_agent + finance_agent team"
  - LangGraph: "Parallel research with fan-out/fan-in"

Shows: parallel fan-out to specialists, typed financial state,
reducers for merging parallel results, and consensus patterns.
"""


from flowforge import (
    Agent, Team, StoreBase, Flow, Unit, FunctionUnit,
    Persona, Task, LLMUnit, ReducerRegistry,
)
from pydantic import Field
from typing import Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Typed State — Financial analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AnalystReport(StoreBase):
    """Typed state for stock analysis pipeline."""
    ticker: str = ""
    company_name: str = ""

    # Parallel analysis results
    fundamental_analysis: str = ""
    technical_analysis: str = ""
    sentiment_analysis: str = ""
    risk_assessment: str = ""

    # Synthesized
    recommendation: str = ""  # BUY / HOLD / SELL
    confidence: float = 0.0
    price_target: float = 0.0
    key_risks: list[str] = Field(default_factory=list)
    summary: str = ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Specialist Units
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FundamentalUnit(Unit):
    """Analyzes financials — P/E, revenue, margins."""
    def prep(self, store: AnalystReport):
        return store.ticker

    def exec(self, ticker: str) -> str:
        return (
            f"Fundamental Analysis for {ticker}:\n"
            f"• P/E Ratio: 28.5 (industry avg: 25.1)\n"
            f"• Revenue Growth: 22% YoY\n"
            f"• Gross Margin: 74.2% (expanding)\n"
            f"• Free Cash Flow: $12.3B\n"
            f"• Debt/Equity: 0.41 (healthy)\n"
            f"Signal: BULLISH — strong fundamentals with growth"
        )

    def post(self, store: AnalystReport, analysis: str) -> str:
        store.fundamental_analysis = analysis
        return "default"


class TechnicalUnit(Unit):
    """Analyzes price patterns — moving averages, RSI, MACD."""
    def prep(self, store: AnalystReport):
        return store.ticker

    def exec(self, ticker: str) -> str:
        return (
            f"Technical Analysis for {ticker}:\n"
            f"• 50-day MA: $142.30 (price above)\n"
            f"• 200-day MA: $128.50 (price above)\n"
            f"• RSI: 62 (neutral, not overbought)\n"
            f"• MACD: Bullish crossover 3 days ago\n"
            f"• Support: $138, Resistance: $155\n"
            f"Signal: BULLISH — uptrend with momentum"
        )

    def post(self, store: AnalystReport, analysis: str) -> str:
        store.technical_analysis = analysis
        return "default"


class SentimentUnit(Unit):
    """Analyzes news sentiment and social signals."""
    def prep(self, store: AnalystReport):
        return store.ticker

    def exec(self, ticker: str) -> str:
        return (
            f"Sentiment Analysis for {ticker}:\n"
            f"• News sentiment: 0.72 (positive)\n"
            f"• Social media buzz: High (trending on fintwit)\n"
            f"• Analyst consensus: 8 Buy, 3 Hold, 1 Sell\n"
            f"• Insider activity: CEO bought 50K shares last week\n"
            f"Signal: BULLISH — strong positive sentiment"
        )

    def post(self, store: AnalystReport, analysis: str) -> str:
        store.sentiment_analysis = analysis
        return "default"


class RiskUnit(Unit):
    """Assesses risks — macro, competitive, regulatory."""
    def prep(self, store: AnalystReport):
        return store.ticker

    def exec(self, ticker: str) -> str:
        return (
            f"Risk Assessment for {ticker}:\n"
            f"• Macro risk: MEDIUM — rate uncertainty\n"
            f"• Competition: HIGH — new entrants in AI space\n"
            f"• Regulatory: LOW — favorable policy environment\n"
            f"• Concentration: MEDIUM — top 3 clients = 40% revenue\n"
            f"• Overall risk score: 6.2/10"
        )

    def post(self, store: AnalystReport, analysis: str) -> str:
        store.risk_assessment = analysis
        store.key_risks = [
            "Rate uncertainty may pressure valuations",
            "Increasing competition in AI space",
            "Client concentration risk",
        ]
        return "default"


class SynthesisUnit(Unit):
    """
    Synthesizes all parallel analyses into a final recommendation.
    This is the fan-in node that merges specialist outputs.
    """
    def prep(self, store: AnalystReport):
        return {
            "fundamental": store.fundamental_analysis,
            "technical": store.technical_analysis,
            "sentiment": store.sentiment_analysis,
            "risk": store.risk_assessment,
        }

    def exec(self, analyses: dict) -> dict:
        # Count bullish signals
        bullish_count = sum(
            1 for a in analyses.values()
            if "BULLISH" in a.upper()
        )
        total = len(analyses)

        if bullish_count >= 3:
            rec = "BUY"
            conf = 0.85
        elif bullish_count >= 2:
            rec = "HOLD"
            conf = 0.65
        else:
            rec = "SELL"
            conf = 0.70

        return {
            "recommendation": rec,
            "confidence": conf,
            "price_target": 155.0 if rec == "BUY" else 140.0,
            "summary": (
                f"Based on {bullish_count}/{total} bullish signals: "
                f"{rec} with {conf:.0%} confidence. "
                f"Strong fundamentals and technical momentum support the thesis, "
                f"tempered by competitive and macro risks."
            ),
        }

    def post(self, store: AnalystReport, result: dict) -> str:
        store.recommendation = result["recommendation"]
        store.confidence = result["confidence"]
        store.price_target = result["price_target"]
        store.summary = result["summary"]
        return "default"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Build the analysis graph
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_analysis_flow() -> Flow:
    """
    Parallel fan-out to 4 specialists, fan-in at synthesis:
    
                  ┌→ fundamental ──┐
        dispatch ─┼→ technical ────┼→ synthesize
                  ├→ sentiment ────┤
                  └→ risk ─────────┘
    """
    flow = Flow(reducers=ReducerRegistry({
        "key_risks": "extend",
    }))

    # Dispatch node (fan-out trigger)
    def dispatch(store):
        store.company_name = f"Company ({store.ticker})"
        return "default"
    flow.add("dispatch", FunctionUnit(dispatch))

    # Parallel specialists
    flow.add("fundamental", FundamentalUnit())
    flow.add("technical", TechnicalUnit())
    flow.add("sentiment", SentimentUnit())
    flow.add("risk", RiskUnit())

    # Synthesis (fan-in)
    flow.add("synthesize", SynthesisUnit())

    # Fan-out: dispatch → all 4 specialists (runs sequentially)
    # Then synthesis collects all results
    flow.wire("dispatch", "fundamental")
    flow.wire("fundamental", "technical")
    flow.wire("technical", "sentiment")
    flow.wire("sentiment", "risk")
    flow.wire("risk", "synthesize")

    flow.entry("dispatch")
    return flow


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Demo: Team approach (Agno-style)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def demo_team_approach():
    """Use Team with hierarchical strategy — manager + specialists."""
    print("\n── Team Approach (hierarchical) ──")

    def mock_llm(system, user, tools=None):
        if "Manager" in system or "Lead" in system:
            return "Subtasks: 1) Analyze fundamentals 2) Check technicals 3) Assess sentiment"
        elif "Fundamental" in system:
            return "P/E 28.5, Revenue +22% YoY, Strong buy signal"
        elif "Technical" in system:
            return "Above 200-day MA, RSI 62, MACD bullish crossover"
        elif "Sentiment" in system:
            return "News positive 0.72, Insider buying, Analyst consensus: Buy"
        return f"Analysis complete for: {user[:50]}..."

    manager = Agent("Investment Manager",
        "Synthesize specialist analyses into investment recommendation",
        backstory="20 years at Goldman Sachs, CFA charterholder",
        llm_fn=mock_llm)

    fundamental = Agent("Fundamental Analyst",
        "Analyze financial statements and valuation metrics",
        llm_fn=mock_llm)

    technical = Agent("Technical Analyst",
        "Analyze price patterns and momentum indicators",
        llm_fn=mock_llm)

    sentiment = Agent("Sentiment Analyst",
        "Analyze news, social media, and insider activity",
        llm_fn=mock_llm)

    team = Team(
        [fundamental, technical, sentiment],
        strategy="hierarchical",
        manager=manager,
    )

    store = team.run("Analyze NVDA for investment")
    print(f"Graph: {team.describe()}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Demo: Custom graph approach
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def demo_graph_approach():
    """Use custom Flow for full control."""
    print("\n── Graph Approach (fan-out/fan-in) ──")

    flow = build_analysis_flow()

    state = AnalystReport(ticker="NVDA")
    state.checkpoint("initial")

    result = flow.run(state)

    print(f"\n📊 Analysis Report: {result.ticker}")
    print(f"{'─' * 50}")
    print(f"Recommendation: {result.recommendation}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Price Target: ${result.price_target:.2f}")
    print(f"{'─' * 50}")
    print(f"Fundamental: {result.fundamental_analysis[:60]}...")
    print(f"Technical: {result.technical_analysis[:60]}...")
    print(f"Sentiment: {result.sentiment_analysis[:60]}...")
    print(f"Risk: {result.risk_assessment[:60]}...")
    print(f"{'─' * 50}")
    print(f"Key Risks:")
    for risk in result.key_risks:
        print(f"  ⚠ {risk}")
    print(f"\nSummary: {result.summary}")

    # Show we can rollback
    print(f"\n(Checkpoint available: can rollback to 'initial')")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║  FlowForge Example 4: Stock Analysis             ║")
    print("║  Parallel specialists → synthesis                 ║")
    print("╚══════════════════════════════════════════════════╝")

    demo_team_approach()
    demo_graph_approach()

    print("\n═══ All demos complete ═══")
