"""株の売買判断ツール - FastAPIバックエンド。"""

from pathlib import Path

import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.indicators import compute_all
from backend.judge import judge

app = FastAPI(title="株の売買判断ツール")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

VALID_PERIODS = {"1mo", "3mo", "6mo", "1y", "2y", "5y"}


@app.get("/api/analyze")
def analyze(
    ticker: str = Query(..., description="銘柄コード。日本株は '7203.T' のように指定"),
    period: str = Query("6mo", description="取得期間"),
):
    ticker = ticker.strip()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker を指定してください")
    if period not in VALID_PERIODS:
        raise HTTPException(status_code=400, detail=f"period は {sorted(VALID_PERIODS)} のいずれかを指定してください")

    try:
        raw = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    except Exception as exc:  # yfinance/requests側の通信エラーなど
        raise HTTPException(status_code=502, detail=f"株価データの取得に失敗しました: {exc}") from exc

    if raw is None or raw.empty:
        raise HTTPException(status_code=404, detail=f"銘柄 '{ticker}' のデータが見つかりませんでした")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = compute_all(raw)
    verdict = judge(df)

    history = []
    tail = df.tail(120)
    for date, row in tail.iterrows():
        history.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "close": _safe_float(row["Close"]),
                "sma25": _safe_float(row.get("SMA25")),
                "sma75": _safe_float(row.get("SMA75")),
                "bbUpper": _safe_float(row.get("BB_UPPER")),
                "bbLower": _safe_float(row.get("BB_LOWER")),
                "rsi14": _safe_float(row.get("RSI14")),
                "macd": _safe_float(row.get("MACD")),
                "macdSignal": _safe_float(row.get("MACD_SIGNAL")),
            }
        )

    return {
        "ticker": ticker,
        "period": period,
        "decision": verdict.decision,
        "score": verdict.score,
        "reasons": [
            {
                "indicator": r.indicator,
                "signal": r.signal,
                "score": r.score,
                "detail": r.detail,
            }
            for r in verdict.reasons
        ],
        "latestClose": _safe_float(df.iloc[-1]["Close"]),
        "history": history,
    }


def _safe_float(value):
    if value is None or pd.isna(value):
        return None
    return round(float(value), 4)


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
