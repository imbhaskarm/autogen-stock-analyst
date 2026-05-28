import asyncio
from orchestrator import run_analysis


def main():
    ticker   = input("Enter stock ticker (e.g. MSFT, AAPL, TSLA): ").strip().upper()
    task     = (
        f"Analyse {ticker} stock and provide a full report with "
        f"fundamental data, technical analysis, and a Buy/Hold/Sell recommendation."
    )
    report = asyncio.run(run_analysis(task))
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)


if __name__ == "__main__":
    main()
