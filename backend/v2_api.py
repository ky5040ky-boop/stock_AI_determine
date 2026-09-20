from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query

try:
    from .backtest_v2 import run_backtest
    from .indicators_v2 import compute_all, normalize_ohlcv
    from .judge_v2 import analyze_row
except ImportError:
    from backtest_v2 import run_backtest
    from indicators_v2 import compute_all, normalize_ohlcv
    from judge_v2 import analyze_row

router = APIRouter(prefix="/api/v2", tags=["strategy-v2"])

DisplayPeriod = Literal["1mo", "3mo", "6mo", "1y", "2y", "5y"]
_PERIOD_DAYS = {"1mo": 31, "3mo": 93, "6mo": 186, "1y": 366, "2y": 732, "5y": 1830}


def _download(ticker: str, period: str = "5y") -> pd.DataFrame:
    ticker = ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")
    try:
        raw = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
        return normalize_ohlcv(raw)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"price download failed: {exc}") from exc


def _json_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _date_str(idx) -> str:
    return idx.date().isoformat() if hasattr(idx, "date") else str(idx)


@router.get("/analyze/{ticker}")
def analyze_ticker(ticker: str, display_period: DisplayPeriod = "6mo"):
    # Always fetch enough warm-up history. The display window no longer changes
    # the indicator state or latest decision.
    raw = _download(ticker, period="5y")
    feat = compute_all(raw)
    latest = analyze_row(feat.iloc[-1])

    cutoff = feat.index.max() - pd.Timedelta(days=_PERIOD_DAYS[display_period])
    view = feat.loc[feat.index >= cutoff]

    chart_cols = [
        "Open", "High", "Low", "Close", "Volume", "SMA25", "SMA75",
        "RSI14", "MACD", "MACD_SIGNAL", "BB_UPPER", "BB_LOWER", "ADX14", "ATR_PCT"
    ]
    chart = []
    for idx, row in view.iterrows():
        item = {"date": _date_str(idx)}
        for col in chart_cols:
            item[col] = _json_value(row.get(col))
        chart.append(item)

    return {
        "ticker": ticker.strip().upper(),
        "display_period": display_period,
        "history_used_for_calculation": "5y",
        "latest_date": _date_str(feat.index[-1]),
        "latest_price": round(float(feat.iloc[-1]["Close"]), 4),
        "analysis": latest,
        "chart": chart,
    }


@router.get("/backtest/{ticker}")
def backtest_ticker(
    ticker: str,
    years: int = Query(5, ge=1, le=10),
    horizon: int = Query(10, ge=1, le=60),
    neutral_threshold: float = Query(0.02, ge=0.0, le=0.20),
    commission_bps: float = Query(10.0, ge=0.0, le=200.0),
    slippage_bps: float = Query(5.0, ge=0.0, le=200.0),
):
    # 10y gives warm-up headroom before the user-selected evaluation window.
    raw = _download(ticker, period="10y")
    end = raw.index.max()
    start = end - pd.Timedelta(days=int(years * 365.25))

    result = run_backtest(
        raw,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        horizon=horizon,
        neutral_threshold=neutral_threshold,
        analysis_start=start,
    )
    result.update({
        "ticker": ticker.strip().upper(),
        "evaluation_years": years,
        "evaluation_start": _date_str(start),
        "evaluation_end": _date_str(end),
    })
    return result
