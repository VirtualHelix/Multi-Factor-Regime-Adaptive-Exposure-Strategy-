import pandas as pd
import matplotlib.pyplot as plt

from src.adaptive_exposure import compute_exposure, run_backtest


# =========================================================
# Load Data
# =========================================================
def load_data(path):
    df = pd.read_csv(path)
    required_cols = [
        "QQQ_close",
        "QQQ_today_close_to_tmrw_close_return",
        "VIX",
        "LT_Score",
        "ST_Score",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df


# =========================================================
# Performance Metrics
# =========================================================
def calc_perf(equity):
    total_return = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (252 / len(equity)) - 1

    dd = (equity / equity.cummax()) - 1
    max_dd = abs(dd.min())

    calmar = cagr / max_dd if max_dd > 0 else float("nan")

    daily = equity.pct_change().fillna(0)
    sharpe = (daily.mean() / (daily.std() + 1e-9)) * 252**0.5

    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_dd": max_dd,
        "calmar": calmar,
        "sharpe": sharpe,
    }


# =========================================================
# Main Entry Point
# =========================================================
def main():
    data_path = "data/qqq_features.csv"   # <-- You can rename this file as needed

    print("Loading data...")
    df = load_data(data_path)

    print("Computing exposure...")
    df = compute_exposure(df)

    print("Running backtest...")
    df = run_backtest(df)

    # -------------------------
    # Compute Performance
    # -------------------------
    strat = calc_perf(df["cum_equity"])
    bh = calc_perf(df["bh_equity"])

    print("\n=== Adaptive Exposure Strategy ===")
    print(f"Total Return: {strat['total_return']:.2%}")
    print(f"CAGR:         {strat['cagr']:.2%}")
    print(f"Max Drawdown: {strat['max_dd']:.2%}")
    print(f"Calmar:       {strat['calmar']:.2f}")
    print(f"Sharpe:       {strat['sharpe']:.2f}")

    print("\n=== Buy & Hold (QQQ) ===")
    print(f"Total Return: {bh['total_return']:.2%}")
    print(f"CAGR:         {bh['cagr']:.2%}")
    print(f"Max Drawdown: {bh['max_dd']:.2%}")
    print(f"Calmar:       {bh['calmar']:.2f}")
    print(f"Sharpe:       {bh['sharpe']:.2f}")

    # -------------------------
    # Plot Results
    # -------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(df["cum_equity"], label="Adaptive Strategy", color="blue")
    plt.plot(df["bh_equity"], label="Buy & Hold QQQ", linestyle="--", color="orange")

    plt.title("Adaptive Financial Exposure Model vs Buy & Hold")
    plt.ylabel("Equity (Growth of $1)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
