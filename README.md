# AutoGen Stock Analyst

A multi-agent stock analysis pipeline built with **AutoGen v0.4**. Three specialist agents run in sequence to produce a structured Markdown report: fundamentals, technical indicators, and a final Buy/Hold/Sell recommendation.

Built while learning AutoGen's v0.4 API as part of my transition from .NET/Azure development into GenAI engineering. I wanted to understand how to coordinate multiple agents with different responsibilities without them stepping on each other's work.

---

## How It Works

```
User query
    |
    v
[financial_reporting_analyst]  -- fetches price, P/E, market cap, dividend yield
    |
    v
[technical_analyst]            -- computes SMA 20, EMA 20, RSI 14
    |
    v
[strategy_agent]               -- calls risk + MACD tools, outputs BUY/HOLD/SELL
    |
    v
[supervisor]                   -- human proxy, receives consolidated report
```

- Agents run in fixed order via `RoundRobinGroupChat`
- The loop terminates when `strategy_agent` outputs `TERMINATE` or after 20 messages
- All market data is fetched live from Yahoo Finance via `yfinance`
- The LLM (Llama 3.3-70b on Groq) never invents numbers -- it only interprets what the tools return

---

## Project Structure

```
autogen-stock-analyst/
├── main.py            # entry point -- prompts for ticker, runs analysis, prints report
├── orchestrator.py    # RoundRobinGroupChat setup and termination conditions
├── agents.py          # all four agent definitions + model client config
├── tools.py           # FinanceTools: 4 static methods wrapping yfinance
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Tools

| Tool | What It Returns |
|---|---|
| `finance_data_fetch` | Price, market cap, P/E, dividend yield, last 10 closes |
| `technical_analysis_tool` | SMA 20, EMA 20, RSI 14, last close |
| `risk_assessment_tool` | Beta, volatility, 52-week change, risk rating (High/Moderate/Low) |
| `strategy_signal_tool` | MACD, MACD signal line, RSI, last close |

---

## Setup

**1. Clone and create a virtual environment**

```bash
git clone https://github.com/imbhaskarm/autogen-stock-analyst.git
cd autogen-stock-analyst
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Set up your API key**

Copy `.env.example` to `.env` and fill in your Groq key:

```env
GROQ_API_KEY=your_groq_api_key_here
MODEL_NAME=llama-3.3-70b-versatile
BASE_URL=https://api.groq.com/openai/v1
```

Get a free key at [https://console.groq.com](https://console.groq.com)

**4. Run**

```bash
python main.py
```

You will be prompted to enter a ticker:

```
Enter stock ticker (e.g. MSFT, AAPL, TSLA): MSFT
```

Or call `run_analysis()` directly from your own script:

```python
import asyncio
from orchestrator import run_analysis

report = asyncio.run(run_analysis("Analyse AAPL and give a Buy/Hold/Sell recommendation."))
print(report)
```

---

## Example Output (MSFT)

```markdown
### Microsoft Corporation (MSFT) Stock Report

**Current Price:** $424.16 | **Market Cap:** $3.15T | **P/E:** 26.56 | **Dividend Yield:** 0.86%

#### Technical Analysis
- SMA 20: $383.79 | EMA 20: $395.57 | RSI: 86.34
- RSI above 70 indicates overbought conditions.

#### Risk Assessment
- Beta: 1.11 | Annualised Volatility: 24.36% | Risk Rating: Moderate

#### Recommendation: HOLD
MACD is above the signal line (bullish momentum), but RSI at 86 signals the stock
is overbought. Waiting for a pullback before entering a new position is prudent
given the Moderate risk profile.
```

---

## Things I Learned Building This

- AutoGen v0.4 is a significant breaking change from v0.2 -- `llm_config` is gone, replaced by a proper `model_client` object. The old `register_function()` loop is also gone; tools are passed directly to `AssistantAgent(tools=[])`
- `UserProxyAgent` in v0.4 is just a human relay -- it has no orchestration role. The execution order is entirely controlled by `RoundRobinGroupChat`
- Termination needs two conditions: a `TextMentionTermination` for the happy path (agent says TERMINATE) and a `MaxMessageTermination` as a safety cap. Without the cap, a misbehaving agent can loop forever
- Keeping tools as plain static methods (not `@tool`-decorated functions) is cleaner for AutoGen -- the tool schema is inferred automatically from the docstring and type hints

---

## Tech Stack

| Tool | Purpose |
|---|---|
| AutoGen v0.4 | Multi-agent orchestration -- RoundRobinGroupChat, termination conditions |
| Groq (Llama 3.3-70b) | LLM backbone for all three analyst agents |
| yfinance | Live market data -- prices, fundamentals, historical closes |
| python-dotenv | Environment variable management |
