import yfinance as yf
import pandas as pd
import numpy as np


def fetch_hourly_data(ticker: str, days: int = 730) -> pd.DataFrame:
    """
    Fetch hourly OHLCV data for the given ticker.
    yfinance limits hourly data to ~730 days via chunked requests.
    """
    # yfinance max period for 1h is 730 days but must be fetched in chunks
    import datetime

    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=days)

    # Fetch in 60-day chunks (yfinance hourly limit per request)
    chunks = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + datetime.timedelta(days=59), end)
        try:
            df_chunk = yf.download(
                ticker,
                start=chunk_start.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
                interval="1h",
                progress=False,
                auto_adjust=True,
            )
            if not df_chunk.empty:
                chunks.append(df_chunk)
        except Exception:
            pass
        chunk_start = chunk_end + datetime.timedelta(hours=1)

    if not chunks:
        raise ValueError(f"No data returned for ticker: {ticker}")

    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Rename to standard names
    df.columns = [c.capitalize() for c in df.columns]
    required = ["Open", "High", "Low", "Close", "Volume"]
    df = df[required].dropna()

    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the three HMM training features:
      1. Returns      – log return of Close
      2. Range        – (High - Low) / Close
      3. Vol_vol      – rolling 24-bar std of Volume (normalised)
    """
    df = df.copy()
    df["Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Range"] = (df["High"] - df["Low"]) / df["Close"]

    vol_ma = df["Volume"].rolling(24).mean().replace(0, np.nan)
    df["Vol_vol"] = df["Volume"].rolling(24).std() / vol_ma

    df.dropna(inplace=True)
    return df
