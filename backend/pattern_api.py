"""チャートパターン検出(スクリーナー)のAPI。"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from backend.market_data import download_ohlcv
from backend.patterns import PATTERN_NAMES, STAGE_NAMES, detect_patterns
from backend.universes import UNIVERSES, ticker_name

router = APIRouter(prefix="/api/patterns", tags=["patterns"])

CACHE_TTL_SEC = 30 * 60
MAX_TICKERS = 150
_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_cache_lock = threading.Lock()

STATS_PATH = Path(__file__).resolve().parent.parent / "data" / "pattern_stats.json"
_stats_cache: dict = {}


def load_stats() -> dict | None:
    """バックテスト結果(python -m backend.pattern_backtest で作成)。ファイル更新時に読み直す。"""
    try:
        mtime = STATS_PATH.stat().st_mtime
    except OSError:
        return None
    if _stats_cache.get("mtime") != mtime:
        _stats_cache.update(mtime=mtime, data=json.loads(STATS_PATH.read_text()))
    return _stats_cache["data"]


HIST_KEYS = ("n", "completion_rate", "target_rate", "stop_rate", "win_rate", "excess_avg_20", "excess_halves")


def historical_record(stats: dict | None, p: dict) -> dict | None:
    """同じパターン・進行段階の過去の実績。"""
    if not stats:
        return None
    stage = p["stage"]
    if p["pattern"] == "symmetric_triangle" and stage == "breakout":
        stage = "breakout_up" if p["bias"] == "bullish" else "breakout_down"
    st = stats["by_pattern"].get(p["pattern"], {}).get(stage)
    return {k: st[k] for k in HIST_KEYS if k in st} if st else None


def normalize_ticker(raw: str) -> str:
    """'7203' のような日本株の証券コードには '.T' を補う。"""
    t = raw.strip().upper()
    if re.fullmatch(r"\d{3}[0-9A-Z]", t):
        t += ".T"
    return t


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
            fetched = download_ohlcv(missing, period)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"株価データの取得に失敗しました: {exc}") from exc
        with _cache_lock:
            for t, df in fetched.items():
                _cache[f"{t}:{period}"] = (now, df)
        out.update(fetched)
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
    stats = load_stats()
    results = []
    for t in names:
        df = prices.get(t)
        if df is None:
            continue
        for p in detect_patterns(df, sensitivity=sensitivity):
            if p["pattern"] not in want_patterns or (p["score"] or 0) < min_score:
                continue
            if want_stages and p["stage"] not in want_stages:
                continue
            results.append({"ticker": t, "name": ticker_name(t), "last_date": str(df.index[-1].date()),
                            **p, "history": historical_record(stats, p)})
    results.sort(key=lambda r: -1 if r["score"] is None else r["score"], reverse=True)

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


@router.get("/stats")
def stats():
    """過去データでのパターン別の実績と、確率モデルの検証結果。"""
    data = load_stats()
    if data is None:
        raise HTTPException(status_code=404, detail="検証結果がありません。python -m backend.pattern_backtest を実行してください")
    return {**data, "pattern_names": PATTERN_NAMES, "stage_names": STAGE_NAMES}
