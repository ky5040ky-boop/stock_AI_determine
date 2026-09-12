"""テクニカル指標の計算ロジック。"""

import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window=window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = sma(close, window)
    std = close.rolling(window=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCVデータフレーム(Close列必須)に各指標列を追加して返す。"""
    out = df.copy()
    close = out["Close"]

    out["SMA25"] = sma(close, 25)
    out["SMA75"] = sma(close, 75)
    out["RSI14"] = rsi(close, 14)

    macd_line, signal_line, hist = macd(close)
    out["MACD"] = macd_line
    out["MACD_SIGNAL"] = signal_line
    out["MACD_HIST"] = hist

    upper, mid, lower = bollinger_bands(close)
    out["BB_UPPER"] = upper
    out["BB_MID"] = mid
    out["BB_LOWER"] = lower

    return out
