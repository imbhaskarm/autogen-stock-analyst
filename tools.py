import json
import yfinance as yf


class FinanceTools:
    """
    Static tool methods used by the analyst agents.

    Each method fetches live data from Yahoo Finance, computes the
    relevant metrics, and returns a JSON string the agent can parse.
    All methods follow the same return structure:
        {"name": "<tool_name>", "data": {...}}
    so agents can reliably extract results without guessing the schema.
    """

    @staticmethod
    def finance_data_fetch(ticker: str, period: str = "1mo") -> str:
        """Fetch basic stock info: name, price, market cap, P/E, dividend, recent closes."""
        try:
            stock = yf.Ticker(ticker)
            hist  = stock.history(period=period)
            info  = stock.info
            result = {
                "name": "finance_data_fetch",
                "data": {
                    "name":           info.get("longName", "N/A"),
                    "symbol":         ticker.upper(),
                    "current_price":  info.get("currentPrice", info.get("regularMarketPrice", "N/A")),
                    "currency":       info.get("currency", "USD"),
                    "summary":        info.get("longBusinessSummary", "N/A"),
                    "market_cap":     info.get("marketCap", "N/A"),
                    "pe_ratio":       info.get("trailingPE", "N/A"),
                    "price_to_book":  info.get("priceToBook", "N/A"),
                    "dividend_yield": info.get("dividendYield", "N/A"),
                    "recent_closes":  hist["Close"].round(2).tolist()[-10:],
                },
            }
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"name": "finance_data_fetch", "error": str(e)})

    @staticmethod
    def technical_analysis_tool(ticker: str) -> str:
        """Calculate 20-day SMA, 20-day EMA, 14-day RSI, and last close price."""
        try:
            stock = yf.Ticker(ticker)
            hist  = stock.history(period="3mo")
            close = hist["Close"]

            sma_20     = round(close.rolling(window=20).mean().iloc[-1], 2)
            ema_20     = round(close.ewm(span=20, adjust=False).mean().iloc[-1], 2)
            delta      = close.diff()
            gain       = delta.clip(lower=0)
            loss       = -delta.clip(upper=0)
            rs         = gain.rolling(14).mean() / loss.rolling(14).mean()
            rsi        = round((100 - (100 / (1 + rs))).iloc[-1], 2)
            last_close = round(close.iloc[-1], 2)

            result = {
                "name": "technical_analysis_tool",
                "data": {"SMA_20": sma_20, "EMA_20": ema_20, "RSI": rsi, "last_close": last_close},
            }
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"name": "technical_analysis_tool", "error": str(e)})

    @staticmethod
    def risk_assessment_tool(ticker: str) -> str:
        """
        Evaluate the risk profile of a stock.

        Returns beta, market cap, 52-week change, dividend yield, annualised
        volatility, and a calculated risk rating (High / Moderate / Low).
        """
        try:
            stock = yf.Ticker(ticker)
            info  = stock.info
            hist  = stock.history(period="1y")   # 1 year needed for annualised volatility
            close = hist["Close"]

            beta = info.get("beta", None)

            # std of daily returns x sqrt(252) converts to annualised volatility
            daily_returns         = close.pct_change().dropna()
            annualised_volatility = round(float(daily_returns.std() * (252 ** 0.5)), 4)

            week_52_change = info.get("52WeekChange", None)
            if week_52_change:
                week_52_change = round(week_52_change * 100, 2)

            if beta is not None:
                if beta > 1.5 or annualised_volatility > 0.40:
                    risk_rating = "High"
                elif beta > 1.0 or annualised_volatility > 0.25:
                    risk_rating = "Moderate"
                else:
                    risk_rating = "Low"
            else:
                risk_rating = "Unknown"

            result = {
                "name": "risk_assessment_tool",
                "data": {
                    "beta":                  beta,
                    "market_cap":            info.get("marketCap", "N/A"),
                    "week_52_change_pct":    week_52_change,
                    "dividend_yield":        info.get("dividendYield", "N/A"),
                    "annualised_volatility": annualised_volatility,
                    "risk_rating":           risk_rating,
                },
            }
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"name": "risk_assessment_tool", "error": str(e)})

    @staticmethod
    def strategy_signal_tool(ticker: str) -> str:
        """
        Generate trade signals using MACD and RSI.

        Uses 6 months of daily price data.
        Returns MACD value, MACD signal line, RSI, and last close.
        """
        try:
            stock = yf.Ticker(ticker)
            hist  = stock.history(period="6mo", interval="1d")
            close = hist["Close"]

            # MACD = EMA(12) - EMA(26); signal = 9-day EMA of MACD
            ema_12      = close.ewm(span=12, adjust=False).mean()
            ema_26      = close.ewm(span=26, adjust=False).mean()
            macd_line   = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()

            macd_value  = round(float(macd_line.iloc[-1]), 4)
            macd_signal = round(float(signal_line.iloc[-1]), 4)

            delta      = close.diff()
            gain       = delta.clip(lower=0)
            loss       = -delta.clip(upper=0)
            rs         = gain.rolling(14).mean() / loss.rolling(14).mean()
            rsi        = round(float((100 - (100 / (1 + rs))).iloc[-1]), 2)
            last_close = round(float(close.iloc[-1]), 2)

            result = {
                "name": "strategy_signal_tool",
                "data": {
                    "MACD":        macd_value,
                    "MACD_signal": macd_signal,
                    "RSI":         rsi,
                    "last_close":  last_close,
                },
            }
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"name": "strategy_signal_tool", "error": str(e)})
