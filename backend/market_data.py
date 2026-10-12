"""株価データ(日足OHLCV)の取得。"""

from __future__ import annotations

import pandas as pd

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in OHLCV if c in df.columns]
    return df[cols].apply(pd.to_numeric, errors="coerce").dropna(subset=["High", "Low", "Close"])


def download_ohlcv(tickers: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    """yfinance で日足を一括取得する。取得できなかった銘柄は結果に含まれない。"""
    import yfinance as yf

    raw = yf.download(
        tickers, period=period, interval="1d", group_by="ticker",
        auto_adjust=True, progress=False, threads=True,
    )
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    for t in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if t not in raw.columns.get_level_values(0):
                continue
            df = raw[t]
        else:
            df = raw
        df = clean_ohlcv(df)
        if len(df):
            out[t] = df
    return out


def load_csv_long(path: str) -> dict[str, pd.DataFrame]:
    """縦持ちCSV(date, open, high, low, close, volume, Name/ticker 列)を銘柄ごとに分割する。"""
    d = pd.read_csv(path)
    d.columns = [c.strip().lower() for c in d.columns]
    name_col = next(c for c in ("name", "ticker", "symbol", "code") if c in d.columns)
    d["date"] = pd.to_datetime(d["date"])
    d = d.rename(columns={c.lower(): c for c in OHLCV})
    out = {}
    for t, g in d.groupby(name_col):
        df = clean_ohlcv(g.set_index("date").sort_index())
        if len(df):
            out[str(t)] = df
    return out
