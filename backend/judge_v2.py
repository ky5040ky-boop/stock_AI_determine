from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ComponentScore:
    name: str
    score: float
    weight: float
    weighted_score: float
    reason: str


def _finite(row: pd.Series, *names: str) -> bool:
    for name in names:
        value = row.get(name, np.nan)
        if value is None or not np.isfinite(float(value)):
            return False
    return True


def detect_regime(row: pd.Series) -> dict[str, Any]:
    if not _finite(row, "ADX14", "SMA25", "SMA75", "MACD", "MACD_SIGNAL"):
        return {"type": "UNKNOWN", "direction": "NEUTRAL", "adx": None}

    adx = float(row["ADX14"])
    sma_up = row["SMA25"] > row["SMA75"]
    macd_up = row["MACD"] > row["MACD_SIGNAL"]

    if adx >= 25.0:
        regime = "TREND"
    elif adx <= 20.0:
        regime = "RANGE"
    else:
        regime = "TRANSITION"

    if sma_up and macd_up:
        direction = "UP"
    elif (not sma_up) and (not macd_up):
        direction = "DOWN"
    else:
        direction = "MIXED"

    return {"type": regime, "direction": direction, "adx": round(adx, 2)}


def _weights(regime: str) -> dict[str, float]:
    if regime == "TREND":
        return {"SMA": 1.40, "MACD": 1.30, "RSI": 0.55, "BB": 0.55, "VOLUME": 0.60}
    if regime == "RANGE":
        return {"SMA": 0.60, "MACD": 0.60, "RSI": 1.25, "BB": 1.25, "VOLUME": 0.40}
    return {"SMA": 1.00, "MACD": 1.00, "RSI": 0.85, "BB": 0.85, "VOLUME": 0.50}


def _sma_signal(row: pd.Series) -> tuple[float, str]:
    if not _finite(row, "Close", "SMA25", "SMA75"):
        return 0.0, "SMAの計算に必要な履歴が不足"
    close, s25, s75 = float(row["Close"]), float(row["SMA25"]), float(row["SMA75"])
    if close > s25 > s75:
        return 1.0, "株価>SMA25>SMA75で上昇トレンド"
    if close < s25 < s75:
        return -1.0, "株価<SMA25<SMA75で下降トレンド"
    return 0.0, "移動平均線の並びが混在"


def _macd_signal(row: pd.Series) -> tuple[float, str]:
    if not _finite(row, "MACD", "MACD_SIGNAL", "MACD_HIST"):
        return 0.0, "MACDの計算に必要な履歴が不足"
    macd = float(row["MACD"])
    sig = float(row["MACD_SIGNAL"])
    hist = float(row["MACD_HIST"])
    if macd > sig and hist > 0.0:
        return 1.0, "MACDがシグナルを上回る"
    if macd < sig and hist < 0.0:
        return -1.0, "MACDがシグナルを下回る"
    return 0.0, "MACD方向感が弱い"


def _rsi_signal(row: pd.Series) -> tuple[float, str]:
    if not _finite(row, "RSI14"):
        return 0.0, "RSIの計算に必要な履歴が不足"
    rsi = float(row["RSI14"])
    if rsi <= 30.0:
        return 1.0, f"RSI={rsi:.1f}で短期的な売られ過ぎ水準"
    if rsi >= 70.0:
        return -1.0, f"RSI={rsi:.1f}で短期的な買われ過ぎ水準"
    return 0.0, f"RSI={rsi:.1f}で中立圏"


def _bb_signal(row: pd.Series) -> tuple[float, str]:
    if not _finite(row, "Close", "BB_UPPER", "BB_LOWER"):
        return 0.0, "ボリンジャーバンドの計算に必要な履歴が不足"
    close = float(row["Close"])
    upper = float(row["BB_UPPER"])
    lower = float(row["BB_LOWER"])
    if close <= lower:
        return 1.0, "株価がボリンジャーバンド下限以下へ短期乖離"
    if close >= upper:
        return -1.0, "株価がボリンジャーバンド上限以上へ短期乖離"
    return 0.0, "株価はボリンジャーバンド内"


def _volume_signal(row: pd.Series) -> tuple[float, str]:
    if not _finite(row, "VOL_RATIO", "ROC20"):
        return 0.0, "出来高確認用の履歴が不足"
    vr = float(row["VOL_RATIO"])
    roc = float(row["ROC20"])
    if vr >= 1.2 and roc > 0.0:
        return 1.0, f"20日平均比{vr:.2f}倍の出来高を伴う上昇"
    if vr >= 1.2 and roc < 0.0:
        return -1.0, f"20日平均比{vr:.2f}倍の出来高を伴う下落"
    return 0.0, f"出来高比{vr:.2f}倍で強い確認シグナルなし"


def analyze_row(row: pd.Series, threshold: float = 0.28) -> dict[str, Any]:
    """
    Regime-aware rule engine.

    score_normalized is in roughly [-1, 1]. BUY/SELL thresholds therefore
    remain comparable even though the active weights change by regime.
    """
    regime = detect_regime(row)
    weights = _weights(regime["type"])

    raw = {
        "SMA": _sma_signal(row),
        "MACD": _macd_signal(row),
        "RSI": _rsi_signal(row),
        "BB": _bb_signal(row),
        "VOLUME": _volume_signal(row),
    }

    components: list[ComponentScore] = []
    total = 0.0
    max_abs = 0.0
    for name, (score, reason) in raw.items():
        weight = weights[name]
        weighted = score * weight
        total += weighted
        max_abs += weight
        components.append(ComponentScore(name, score, weight, weighted, reason))

    normalized = 0.0 if max_abs == 0.0 else total / max_abs
    if normalized >= threshold:
        decision = "BUY"
    elif normalized <= -threshold:
        decision = "SELL"
    else:
        decision = "HOLD"

    atr_pct = float(row["ATR_PCT"]) if _finite(row, "ATR_PCT") else None
    if atr_pct is None:
        risk = "UNKNOWN"
    elif atr_pct >= 5.0:
        risk = "HIGH"
    elif atr_pct >= 3.0:
        risk = "MEDIUM"
    else:
        risk = "NORMAL"

    # Strength, not a probability. Do not present this as prediction accuracy.
    signal_strength = min(100.0, abs(normalized) * 100.0)

    return {
        "decision": decision,
        "score": round(total, 4),
        "score_normalized": round(normalized, 4),
        "signal_strength": round(signal_strength, 1),
        "regime": regime,
        "risk": risk,
        "components": [asdict(c) for c in components],
    }
