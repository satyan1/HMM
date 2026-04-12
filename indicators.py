import pandas as pd
import numpy as np


# ── RSI ──────────────────────────────────────────────────────────────────────
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── Momentum (n-bar log return) ───────────────────────────────────────────────
def momentum(close: pd.Series, period: int = 10) -> pd.Series:
    """Returns the percentage change over `period` bars."""
    return close.pct_change(period) * 100


# ── Annualised rolling volatility ─────────────────────────────────────────────
def rolling_volatility(close: pd.Series, period: int = 24) -> pd.Series:
    """Annualised volatility based on log returns, expressed as a percentage."""
    log_ret = np.log(close / close.shift(1))
    # hourly bars → ~8736 bars/year
    return log_ret.rolling(period).std() * np.sqrt(8736) * 100


# ── Volume vs SMA ────────────────────────────────────────────────────────────
def volume_above_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    return volume > volume.rolling(period).mean()


# ── ADX ──────────────────────────────────────────────────────────────────────
def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    # zero out when the other direction is larger
    plus_dm[plus_dm < minus_dm.abs()] = 0
    minus_dm[minus_dm < plus_dm.abs()] = 0

    atr = tr.ewm(com=period - 1, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(com=period - 1, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(com=period - 1, min_periods=period).mean() / atr
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(com=period - 1, min_periods=period).mean()


# ── EMA ──────────────────────────────────────────────────────────────────────
def ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


# ── MACD ─────────────────────────────────────────────────────────────────────
def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


# ── Composite: attach all indicators to DataFrame ────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["RSI"] = rsi(df["Close"])
    df["Momentum"] = momentum(df["Close"], period=10)
    df["Volatility"] = rolling_volatility(df["Close"])
    df["Vol_above_SMA"] = volume_above_sma(df["Volume"])
    df["ADX"] = adx(df["High"], df["Low"], df["Close"])
    df["EMA50"] = ema(df["Close"], 50)
    df["EMA200"] = ema(df["Close"], 200)
    df["MACD"], df["MACD_Signal"] = macd(df["Close"])
    return df


# ── Voting system: returns (score, bool_dict) ─────────────────────────────────
def compute_votes(row: pd.Series) -> tuple[int, dict]:
    """
    8 confirmation conditions.
    Returns (number_of_true_conditions, dict_of_individual_bools).
    """
    checks = {
        "RSI < 90":           row["RSI"] < 90,
        "Momentum > 1%":      row["Momentum"] > 1.0,
        "Volatility < 6%":    row["Volatility"] < 6.0,
        "Volume > SMA":       bool(row["Vol_above_SMA"]),
        "ADX > 25":           row["ADX"] > 25,
        "Price > EMA50":      row["Close"] > row["EMA50"],
        "Price > EMA200":     row["Close"] > row["EMA200"],
        "MACD > Signal":      row["MACD"] > row["MACD_Signal"],
    }
    score = sum(checks.values())
    return score, checks
