"""テクニカル指標を組み合わせた売買判断ロジック。"""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class SignalReason:
    indicator: str
    signal: str  # "buy" | "sell" | "neutral"
    score: int
    detail: str


@dataclass
class Verdict:
    decision: str  # "BUY" | "SELL" | "HOLD"
    score: int
    reasons: list = field(default_factory=list)


BUY_THRESHOLD = 2
SELL_THRESHOLD = -2


def _sma_cross_signal(latest: pd.Series, prev: pd.Series) -> SignalReason:
    if pd.isna(latest["SMA25"]) or pd.isna(latest["SMA75"]) or pd.isna(prev["SMA25"]) or pd.isna(prev["SMA75"]):
        return SignalReason("移動平均クロス(25/75日)", "neutral", 0, "データ不足のため判定不可")

    was_below = prev["SMA25"] <= prev["SMA75"]
    is_above = latest["SMA25"] > latest["SMA75"]
    was_above = prev["SMA25"] >= prev["SMA75"]
    is_below = latest["SMA25"] < latest["SMA75"]

    if was_below and is_above:
        return SignalReason("移動平均クロス(25/75日)", "buy", 2, "ゴールデンクロスが発生")
    if was_above and is_below:
        return SignalReason("移動平均クロス(25/75日)", "sell", -2, "デッドクロスが発生")
    if latest["SMA25"] > latest["SMA75"]:
        return SignalReason("移動平均クロス(25/75日)", "buy", 1, "短期線が長期線の上(上昇トレンド継続)")
    return SignalReason("移動平均クロス(25/75日)", "sell", -1, "短期線が長期線の下(下降トレンド継続)")


def _rsi_signal(latest: pd.Series) -> SignalReason:
    rsi_value = latest["RSI14"]
    if pd.isna(rsi_value):
        return SignalReason("RSI(14)", "neutral", 0, "データ不足のため判定不可")

    if rsi_value < 30:
        return SignalReason("RSI(14)", "buy", 1, f"RSI={rsi_value:.1f}で売られすぎ水準")
    if rsi_value > 70:
        return SignalReason("RSI(14)", "sell", -1, f"RSI={rsi_value:.1f}で買われすぎ水準")
    return SignalReason("RSI(14)", "neutral", 0, f"RSI={rsi_value:.1f}で中立水準")


def _macd_signal(latest: pd.Series, prev: pd.Series) -> SignalReason:
    if pd.isna(latest["MACD"]) or pd.isna(latest["MACD_SIGNAL"]) or pd.isna(prev["MACD"]) or pd.isna(prev["MACD_SIGNAL"]):
        return SignalReason("MACD", "neutral", 0, "データ不足のため判定不可")

    was_below = prev["MACD"] <= prev["MACD_SIGNAL"]
    is_above = latest["MACD"] > latest["MACD_SIGNAL"]
    was_above = prev["MACD"] >= prev["MACD_SIGNAL"]
    is_below = latest["MACD"] < latest["MACD_SIGNAL"]

    if was_below and is_above:
        return SignalReason("MACD", "buy", 2, "MACDがシグナルを上抜け(買いシグナル)")
    if was_above and is_below:
        return SignalReason("MACD", "sell", -2, "MACDがシグナルを下抜け(売りシグナル)")
    if latest["MACD"] > latest["MACD_SIGNAL"]:
        return SignalReason("MACD", "buy", 1, "MACDがシグナルの上")
    return SignalReason("MACD", "sell", -1, "MACDがシグナルの下")


def _bollinger_signal(latest: pd.Series) -> SignalReason:
    close = latest["Close"]
    upper = latest["BB_UPPER"]
    lower = latest["BB_LOWER"]
    if pd.isna(upper) or pd.isna(lower):
        return SignalReason("ボリンジャーバンド", "neutral", 0, "データ不足のため判定不可")

    if close <= lower:
        return SignalReason("ボリンジャーバンド", "buy", 1, "株価が下限バンド付近(割安圏)")
    if close >= upper:
        return SignalReason("ボリンジャーバンド", "sell", -1, "株価が上限バンド付近(割高圏)")
    return SignalReason("ボリンジャーバンド", "neutral", 0, "株価はバンド内の中立圏")


def judge(df: pd.DataFrame) -> Verdict:
    """指標付きデータフレームの最新行を元に売買判断を行う。"""
    if len(df) < 2:
        return Verdict("HOLD", 0, [SignalReason("総合判定", "neutral", 0, "データ不足のため判定不可")])

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    reasons = [
        _sma_cross_signal(latest, prev),
        _rsi_signal(latest),
        _macd_signal(latest, prev),
        _bollinger_signal(latest),
    ]

    total_score = sum(r.score for r in reasons)

    if total_score >= BUY_THRESHOLD:
        decision = "BUY"
    elif total_score <= SELL_THRESHOLD:
        decision = "SELL"
    else:
        decision = "HOLD"

    return Verdict(decision, total_score, reasons)
