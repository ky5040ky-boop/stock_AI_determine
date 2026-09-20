from __future__ import annotations

import numpy as np
import pandas as pd


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def sma(series: pd.Series, window: int) -> pd.Series:
    return _numeric(series).rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return _numeric(series).ewm(span=span, adjust=False, min_periods=span).mean()


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    close = _numeric(close)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # No-loss windows are conventionally RSI=100; flat windows are neutral.
    rsi = rsi.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    rsi = rsi.mask((avg_loss == 0.0) & (avg_gain == 0.0), 50.0)
    return rsi


def bollinger_bands(close: pd.Series, window: int = 20, n_std: float = 2.0):
    close = _numeric(close)
    mid = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    return mid, upper, lower


def true_range(df: pd.DataFrame) -> pd.Series:
    high = _numeric(df["High"])
    low = _numeric(df["Low"])
    close = _numeric(df["Close"])
    prev_close = close.shift(1)
    parts = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    return parts.max(axis=1)


def atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def adx_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = _numeric(df["High"])
    low = _numeric(df["Low"])

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0),
        index=df.index,
        dtype=float,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0),
        index=df.index,
        dtype=float,
    )

    atr = atr_wilder(df, period)
    plus_smoothed = plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    minus_smoothed = minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    plus_di = 100.0 * plus_smoothed / atr.replace(0.0, np.nan)
    minus_di = 100.0 * minus_smoothed / atr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance-like OHLCV frames to flat numeric columns."""
    if df is None or len(df) == 0:
        raise ValueError("Price data is empty")

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        # yfinance can return (field, ticker) columns for a single ticker.
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]

    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")

    if "Volume" not in out.columns:
        out["Volume"] = np.nan

    keep = ["Open", "High", "Low", "Close", "Volume"]
    out = out.loc[:, keep].copy()
    for col in keep:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    return out


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute causal technical features only from current/past bars.

    No centered rolling windows or future shifts are used here, so the output
    can be used safely in a walk-forward backtest.
    """
    out = normalize_ohlcv(df)
    close = out["Close"]

    out["SMA25"] = sma(close, 25)
    out["SMA75"] = sma(close, 75)
    out["EMA12"] = ema(close, 12)
    out["EMA26"] = ema(close, 26)
    out["MACD"] = out["EMA12"] - out["EMA26"]
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False, min_periods=9).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]
    out["RSI14"] = rsi_wilder(close, 14)

    bb_mid, bb_upper, bb_lower = bollinger_bands(close, 20, 2.0)
    out["BB_MID"] = bb_mid
    out["BB_UPPER"] = bb_upper
    out["BB_LOWER"] = bb_lower
    out["BB_WIDTH"] = (bb_upper - bb_lower) / bb_mid.replace(0.0, np.nan)

    out["ATR14"] = atr_wilder(out, 14)
    out["ATR_PCT"] = out["ATR14"] / close.replace(0.0, np.nan) * 100.0
    out["ADX14"] = adx_wilder(out, 14)

    out["VOL_MA20"] = out["Volume"].rolling(20, min_periods=20).mean()
    out["VOL_RATIO"] = out["Volume"] / out["VOL_MA20"].replace(0.0, np.nan)
    out["ROC20"] = close.pct_change(20)

    return out
