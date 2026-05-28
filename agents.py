import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from tools import FinanceTools

load_dotenv()


def get_model_client() -> OpenAIChatCompletionClient:
    """
    Build the OpenAIChatCompletionClient pointed at Groq's API.

    AutoGen v0.4 uses a model_client object instead of the old llm_config dict.
    OpenAIChatCompletionClient works here because Groq exposes an
    OpenAI-compatible REST API -- we just override the base_url.
    """
    # ⚠️ Updated from deprecated v0.2 llm_config dict → v0.4 OpenAIChatCompletionClient
    return OpenAIChatCompletionClient(
        model=os.getenv("MODEL_NAME", "llama-3.3-70b-versatile"),
        api_key=os.getenv("GROQ_API_KEY"),
        base_url=os.getenv("BASE_URL", "https://api.groq.com/openai/v1"),
        timeout=60,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
        },
    )


def build_agents() -> dict:
    """
    Initialise all four agents and return them in a dict.

    Fresh instances are created on every call so that no conversation
    state leaks between separate analysis runs.

    Agent order (enforced by RoundRobinGroupChat in orchestrator.py):
        1. financial_reporting_analyst  -- fundamentals
        2. technical_analyst            -- SMA / EMA / RSI
        3. strategy_agent               -- risk + trade signal → recommendation
        4. supervisor                   -- human proxy, receives final output
    """
    model_client = get_model_client()

    financial_analyst = AssistantAgent(
        name="financial_reporting_analyst",
        system_message="""
You are a Financial Reporting Analyst.
- Analyse the stock for the given ticker and write a comprehensive Markdown report.
- Include: price, market cap, P/E ratio, dividend yield, and recent price trend.
- Use ONLY the tools provided. Do NOT invent data.
- Do NOT perform technical analysis (SMA, EMA, RSI) -- that is the next agent's job.
Output: Markdown report.
""",
        model_client=model_client,
        tools=[FinanceTools.finance_data_fetch],
    )

    technical_analyst = AssistantAgent(
        name="technical_analyst",
        system_message="""
You are a Technical Analyst specialising in stock trends.
- Use ONLY the tools provided to fetch technical data.
- Analyse: SMA 20, EMA 20, RSI (14-day), Last Close.
- Identify trends, crossovers, momentum shifts, overbought/oversold conditions.
- Do NOT write a full financial report. Do NOT handle fundamental analysis.
Output: Concise Markdown section.
""",
        model_client=model_client,
        tools=[FinanceTools.technical_analysis_tool],
    )

    strategy_agent = AssistantAgent(
        name="strategy_agent",
        system_message="""
You are a Strategy Analyst responsible for recommending Buy, Hold, or Sell.
- Call BOTH tools: risk_assessment_tool AND strategy_signal_tool before responding.
- Combine risk profile (beta, volatility, risk rating) with trade signals (MACD, RSI).
- Recommendation logic:
    MACD > signal AND RSI < 70  → BUY
    MACD < signal AND RSI > 60  → SELL
    otherwise                   → HOLD
- High risk rating: add a caution note to any BUY recommendation.
- Do NOT invent data or perform raw calculations yourself.
Output: Final recommendation (BUY / HOLD / SELL) with a 2-3 sentence justification in Markdown.
End your response with the exact word: TERMINATE
""",
        model_client=model_client,
        tools=[
            FinanceTools.risk_assessment_tool,
            FinanceTools.strategy_signal_tool,
        ],
    )

    # UserProxyAgent in v0.4 is only a human input relay -- no orchestration logic.
    # The execution order is enforced by RoundRobinGroupChat in orchestrator.py.
    supervisor = UserProxyAgent(
        name="supervisor",
        description="Human user proxy -- relays the query and receives the final report.",
        input_func=lambda prompt: "",
    )

    return {
        "financial_analyst": financial_analyst,
        "technical_analyst": technical_analyst,
        "strategy_agent":    strategy_agent,
        "supervisor":        supervisor,
    }
