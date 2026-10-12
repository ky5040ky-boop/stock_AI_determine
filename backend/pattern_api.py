"""チャートパターン検出(スクリーナー)のAPI。"""

from __future__ import annotations

import re
import threading
import time

import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query

from backend.patterns import PATTERN_NAMES, detect_patterns
from backend.universes import UNIVERSES, ticker_name

router = APIRouter(prefix="/api/patterns", tags=["patterns"])

CACHE_TTL_SEC = 30 * 60
MAX_TICKERS = 150
_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_cache_lock = threading.Lock()


def normalize_ticker(raw: str) -> str:
    """'7203' のような日本株の証券コードには '.T' を補う。"""
    t = raw.strip().upper()
    if re.fullmatch(r"\d{3}[0-9A-Z]", t):
        t += ".T"
    return t


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[cols].apply(pd.to_numeric, errors="coerce").dropna(subset=["High", "Low", "Close"])


def fetch_prices(tickers: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    """日足を一括取得する(30分キャッシュ)。取得できなかった銘柄は結果に含まれない。"""
    now = time.time()
    out: dict[str, pd.DataFrame] = {}
    with _cache_lock:
        for t in tickers:
            hit = _cache.get(f"{t}:{period}")
            if hit and now - hit[0] < CACHE_TTL_SEC:
                out[t] = hit[1]
    missing = [t for t in tickers if t not in out]
    if missing:
        try:
            raw = yf.download(
                missing, period=period, interval="1d", group_by="ticker",
                auto_adjust=True, progress=False, threads=True,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"株価データの取得に失敗しました: {exc}") from exc
        if raw is not None and not raw.empty:
            for t in missing:
                if isinstance(raw.columns, pd.MultiIndex):
                    if t not in raw.columns.get_level_values(0):
                        continue
                    df = raw[t]
                else:
                    df = raw
                df = _clean(df)
                if len(df):
                    out[t] = df
                    with _cache_lock:
                        _cache[f"{t}:{period}"] = (now, df)
    return out


@router.get("/universes")
def universes():
    return [
        {"id": k, "name": v["name"], "count": len(v["tickers"])} for k, v in UNIVERSES.items()
    ] + [{"id": "custom", "name": "カスタム(銘柄コードを入力)", "count": None}]


@router.get("/scan")
def scan(
    universe: str = Query("jp_major", description="プリセットID または custom"),
    tickers: str = Query("", description="custom時: カンマ/空白区切りの銘柄コード"),
    patterns: str = Query("", description="対象パターン(カンマ区切り)。空なら全て"),
    stages: str = Query("", description="対象ステージ(カンマ区切り)。空なら全て"),
    min_score: float = Query(0, ge=0, le=100),
    sensitivity: float = Query(1.0, ge=0.5, le=2.0, description="大きいほど小さな波も検出"),
):
    if universe == "custom":
        names = [normalize_ticker(t) for t in re.split(r"[,\s、]+", tickers) if t.strip()]
        names = list(dict.fromkeys(names))
    elif universe in UNIVERSES:
        names = list(UNIVERSES[universe]["tickers"])
    else:
        raise HTTPException(status_code=400, detail=f"不明な universe: {universe}")
    if not names:
        raise HTTPException(status_code=400, detail="銘柄コードを指定してください")
    if len(names) > MAX_TICKERS:
        raise HTTPException(status_code=400, detail=f"一度にスキャンできるのは{MAX_TICKERS}銘柄までです")

    want_patterns = {p for p in patterns.split(",") if p} or set(PATTERN_NAMES)
    want_stages = {s for s in stages.split(",") if s}

    prices = fetch_prices(names)
    results = []
    for t in names:
        df = prices.get(t)
        if df is None:
            continue
        for p in detect_patterns(df, sensitivity=sensitivity):
            if p["pattern"] not in want_patterns or p["score"] < min_score:
                continue
            if want_stages and p["stage"] not in want_stages:
                continue
            results.append({"ticker": t, "name": ticker_name(t), "last_date": str(df.index[-1].date()), **p})
    results.sort(key=lambda r: r["score"], reverse=True)

    return {
        "scanned": len(names),
        "fetched": len(prices),
        "failed": [t for t in names if t not in prices],
        "results": results,
    }


@router.get("/detail/{ticker}")
def detail(ticker: str, sensitivity: float = Query(1.0, ge=0.5, le=2.0)):
    t = normalize_ticker(ticker)
    df = fetch_prices([t]).get(t)
    if df is None:
        raise HTTPException(status_code=404, detail=f"銘柄 '{t}' のデータが見つかりませんでした")
    return {
        "ticker": t,
        "name": ticker_name(t),
        "patterns": detect_patterns(df, sensitivity=sensitivity),
        "chart": [
            {
                "date": idx.strftime("%Y-%m-%d"),
                "open": round(float(r["Open"]), 4) if "Open" in r and pd.notna(r["Open"]) else None,
                "high": round(float(r["High"]), 4),
                "low": round(float(r["Low"]), 4),
                "close": round(float(r["Close"]), 4),
            }
            for idx, r in df.iterrows()
        ],
    }
