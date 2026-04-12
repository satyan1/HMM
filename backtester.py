import pandas as pd
import numpy as np
from indicators import compute_votes

INITIAL_CAPITAL = 10_000.0
LEVERAGE = 2.5
REQUIRED_VOTES = 7
COOLDOWN_HOURS = 48


def run_backtest(df: pd.DataFrame, bull_state: int, bear_state: int) -> dict:
    """
    Simulate the regime-based strategy on a fully-labelled DataFrame.

    Required columns: Close, State, Regime + all indicator columns.
    Returns a dict with equity_curve, trades, and summary metrics.
    """
    equity = INITIAL_CAPITAL
    position = 0.0          # units held (0 = flat)
    entry_price = np.nan
    entry_time = None
    last_exit_time = None

    equity_curve = []
    trades = []

    for i, (ts, row) in enumerate(df.iterrows()):
        price = row["Close"]
        regime = row["Regime"]
        state = row["State"]

        # ── Mark-to-market ────────────────────────────────────────────────────
        if position > 0:
            unrealised_pnl = (price - entry_price) * position * LEVERAGE
            current_equity = equity + unrealised_pnl
        else:
            current_equity = equity

        equity_curve.append({"timestamp": ts, "equity": current_equity})

        # ── Exit condition ────────────────────────────────────────────────────
        if position > 0 and regime in ("Bear/Crash",):
            realised_pnl = (price - entry_price) * position * LEVERAGE
            equity += realised_pnl
            trades.append(
                {
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl": realised_pnl,
                    "exit_reason": f"Regime flip → {regime}",
                }
            )
            position = 0.0
            entry_price = np.nan
            last_exit_time = ts
            current_equity = equity
            equity_curve[-1]["equity"] = current_equity
            continue

        # ── Entry condition ───────────────────────────────────────────────────
        if position == 0:
            # Cooldown check
            if last_exit_time is not None:
                hours_since_exit = (ts - last_exit_time).total_seconds() / 3600
                if hours_since_exit < COOLDOWN_HOURS:
                    continue

            # HMM must be Bull Run
            if regime != "Bull Run":
                continue

            # Voting system
            score, _ = compute_votes(row)
            if score < REQUIRED_VOTES:
                continue

            # Enter long
            units = (equity * LEVERAGE) / price
            position = units
            entry_price = price
            entry_time = ts

    # ── Close open position at end ────────────────────────────────────────────
    if position > 0:
        last_price = df["Close"].iloc[-1]
        last_ts = df.index[-1]
        realised_pnl = (last_price - entry_price) * position * LEVERAGE
        equity += realised_pnl
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": last_ts,
                "entry_price": entry_price,
                "exit_price": last_price,
                "pnl": realised_pnl,
                "exit_reason": "End of data",
            }
        )
        equity_curve[-1]["equity"] = equity

    # ── Build DataFrames ──────────────────────────────────────────────────────
    eq_df = pd.DataFrame(equity_curve).set_index("timestamp")
    trade_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_time", "exit_time", "entry_price", "exit_price", "pnl", "exit_reason"]
    )

    # ── Metrics ──────────────────────────────────────────────────────────────
    final_equity = eq_df["equity"].iloc[-1]
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # Buy-and-hold baseline (leveraged for fair comparison)
    bnh_return = (
        (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]
    ) * LEVERAGE * 100
    alpha = total_return - bnh_return

    win_rate = 0.0
    if len(trade_df) > 0:
        win_rate = (trade_df["pnl"] > 0).mean() * 100

    # Max drawdown on equity curve
    roll_max = eq_df["equity"].cummax()
    drawdown = (eq_df["equity"] - roll_max) / roll_max * 100
    max_drawdown = drawdown.min()

    return {
        "equity_curve": eq_df,
        "trades": trade_df,
        "final_equity": final_equity,
        "total_return": total_return,
        "bnh_return": bnh_return,
        "alpha": alpha,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "n_trades": len(trade_df),
    }
