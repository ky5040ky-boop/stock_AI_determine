from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

try:
    from .indicators_v2 import compute_all
    from .judge_v2 import analyze_row
except ImportError:  # supports `uvicorn main:app` from inside backend/
    from indicators_v2 import compute_all
    from judge_v2 import analyze_row


@dataclass
class Trade:
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    return_pct: float
    holding_days: int


def _to_date(value) -> str:
    if hasattr(value, "date"):
        return value.date().isoformat()
    return str(value)


def generate_signals(feature_df: pd.DataFrame, threshold: float = 0.28) -> pd.DataFrame:
    rows = []
    for idx, row in feature_df.iterrows():
        result = analyze_row(row, threshold=threshold)
        rows.append(
            {
                "Date": idx,
                "decision": result["decision"],
                "score_normalized": result["score_normalized"],
                "regime": result["regime"]["type"],
                "risk": result["risk"],
            }
        )
    signals = pd.DataFrame(rows).set_index("Date")
    return signals


def evaluate_signal_accuracy(
    feature_df: pd.DataFrame,
    signals: pd.DataFrame,
    horizon: int = 10,
    neutral_threshold: float = 0.02,
) -> dict[str, Any]:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    close = feature_df["Close"]
    future_return = close.shift(-horizon) / close - 1.0

    eval_df = signals.copy()
    eval_df["future_return"] = future_return
    eval_df = eval_df.dropna(subset=["future_return"])

    def is_correct(row) -> bool:
        r = float(row["future_return"])
        d = row["decision"]
        if d == "BUY":
            return r > neutral_threshold
        if d == "SELL":
            return r < -neutral_threshold
        return abs(r) <= neutral_threshold

    eval_df["correct"] = eval_df.apply(is_correct, axis=1)

    by_signal: dict[str, Any] = {}
    for signal in ("BUY", "SELL", "HOLD"):
        part = eval_df[eval_df["decision"] == signal]
        by_signal[signal] = {
            "count": int(len(part)),
            "accuracy_pct": round(float(part["correct"].mean() * 100.0), 2) if len(part) else None,
            "avg_future_return_pct": round(float(part["future_return"].mean() * 100.0), 3) if len(part) else None,
        }

    non_hold = eval_df[eval_df["decision"] != "HOLD"]
    return {
        "horizon_days": int(horizon),
        "neutral_threshold_pct": round(neutral_threshold * 100.0, 3),
        "overall_accuracy_pct": round(float(eval_df["correct"].mean() * 100.0), 2) if len(eval_df) else None,
        "actionable_accuracy_pct": round(float(non_hold["correct"].mean() * 100.0), 2) if len(non_hold) else None,
        "actionable_coverage_pct": round(float(len(non_hold) / len(eval_df) * 100.0), 2) if len(eval_df) else None,
        "by_signal": by_signal,
    }


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0


def _annualized_return(start_value: float, end_value: float, n_days: int) -> float | None:
    if start_value <= 0 or end_value <= 0 or n_days < 2:
        return None
    years = n_days / 252.0
    if years <= 0:
        return None
    return (end_value / start_value) ** (1.0 / years) - 1.0


def _sharpe(equity: pd.Series) -> float | None:
    ret = equity.pct_change().dropna()
    if len(ret) < 2 or float(ret.std(ddof=1)) == 0.0:
        return None
    return float(ret.mean() / ret.std(ddof=1) * sqrt(252.0))


def run_backtest(
    raw_df: pd.DataFrame,
    *,
    initial_cash: float = 1_000_000.0,
    commission_bps: float = 10.0,
    slippage_bps: float = 5.0,
    decision_threshold: float = 0.28,
    horizon: int = 10,
    neutral_threshold: float = 0.02,
    analysis_start=None,
) -> dict[str, Any]:
    """
    Long/cash walk-forward backtest.

    Signal is computed after close on day t and can only execute at open on t+1.
    This avoids same-bar execution/look-ahead bias. BUY opens a long position;
    SELL closes it. HOLD does nothing.
    """
    if initial_cash <= 0:
        raise ValueError("initial_cash must be > 0")
    if commission_bps < 0 or slippage_bps < 0:
        raise ValueError("cost parameters must be >= 0")

    feature_df = compute_all(raw_df)
    signals = generate_signals(feature_df, threshold=decision_threshold)

    if analysis_start is not None:
        bt_df = feature_df.loc[feature_df.index >= analysis_start].copy()
        bt_signals = signals.loc[signals.index >= analysis_start].copy()
    else:
        bt_df = feature_df.copy()
        bt_signals = signals.copy()

    if len(bt_df) < 2:
        raise ValueError("Not enough data for backtest")

    commission = commission_bps / 10_000.0
    slippage = slippage_bps / 10_000.0

    cash = float(initial_cash)
    shares = 0.0
    pending_action: str | None = None
    entry_date = None
    entry_fill = None
    entry_commission = None
    entry_i = None
    trades: list[Trade] = []
    equity_points: list[tuple[Any, float]] = []

    for i, (idx, row) in enumerate(bt_df.iterrows()):
        open_price = float(row["Open"])
        close_price = float(row["Close"])

        # Execute yesterday's close signal at today's open.
        if pending_action == "BUY" and shares == 0.0:
            fill = open_price * (1.0 + slippage)
            affordable = cash / (fill * (1.0 + commission))
            if affordable > 0:
                gross = affordable * fill
                fee = gross * commission
                cash -= gross + fee
                shares = affordable
                entry_date = idx
                entry_fill = fill
                entry_commission = commission
                entry_i = i

        elif pending_action == "SELL" and shares > 0.0:
            fill = open_price * (1.0 - slippage)
            gross = shares * fill
            fee = gross * commission
            cash += gross - fee

            if entry_fill is not None:
                net_entry = entry_fill * (1.0 + float(entry_commission or 0.0))
                net_exit = fill * (1.0 - commission)
                trade_ret = net_exit / net_entry - 1.0
                trades.append(
                    Trade(
                        entry_date=_to_date(entry_date),
                        exit_date=_to_date(idx),
                        entry_price=round(float(entry_fill), 6),
                        exit_price=round(float(fill), 6),
                        return_pct=round(float(trade_ret * 100.0), 4),
                        holding_days=int(i - int(entry_i or i)),
                    )
                )
            shares = 0.0
            entry_date = entry_fill = entry_commission = entry_i = None

        equity_points.append((idx, cash + shares * close_price))

        today_signal = bt_signals.loc[idx, "decision"]
        if shares == 0.0 and today_signal == "BUY":
            pending_action = "BUY"
        elif shares > 0.0 and today_signal == "SELL":
            pending_action = "SELL"
        else:
            pending_action = None

    # Mark remaining position to the final close, then force liquidation for a
    # conservative final cash value including costs.
    last_idx = bt_df.index[-1]
    last_close = float(bt_df.iloc[-1]["Close"])
    if shares > 0.0:
        fill = last_close * (1.0 - slippage)
        gross = shares * fill
        fee = gross * commission
        cash += gross - fee
        if entry_fill is not None:
            net_entry = entry_fill * (1.0 + float(entry_commission or 0.0))
            net_exit = fill * (1.0 - commission)
            trade_ret = net_exit / net_entry - 1.0
            trades.append(
                Trade(
                    entry_date=_to_date(entry_date),
                    exit_date=_to_date(last_idx),
                    entry_price=round(float(entry_fill), 6),
                    exit_price=round(float(fill), 6),
                    return_pct=round(float(trade_ret * 100.0), 4),
                    holding_days=int((len(bt_df) - 1) - int(entry_i or (len(bt_df) - 1))),
                )
            )
        shares = 0.0
        equity_points[-1] = (last_idx, cash)

    equity = pd.Series({idx: value for idx, value in equity_points}, dtype=float).sort_index()
    start_equity = float(equity.iloc[0])
    end_equity = float(equity.iloc[-1])
    total_return = end_equity / start_equity - 1.0
    ann = _annualized_return(start_equity, end_equity, len(equity))
    max_dd = _max_drawdown(equity)
    sharpe = _sharpe(equity)

    trade_returns = np.array([t.return_pct / 100.0 for t in trades], dtype=float)
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (None if gross_profit == 0 else float("inf"))

    benchmark_start = float(bt_df.iloc[0]["Open"])
    benchmark_end = float(bt_df.iloc[-1]["Close"])
    benchmark_return = benchmark_end / benchmark_start - 1.0

    accuracy = evaluate_signal_accuracy(
        feature_df.loc[bt_df.index],
        bt_signals,
        horizon=horizon,
        neutral_threshold=neutral_threshold,
    )

    latest = analyze_row(feature_df.iloc[-1], threshold=decision_threshold)

    curve = [
        {"date": _to_date(idx), "equity": round(float(value), 2)}
        for idx, value in equity.items()
    ]

    metrics = {
        "initial_cash": round(float(initial_cash), 2),
        "final_equity": round(end_equity, 2),
        "total_return_pct": round(total_return * 100.0, 3),
        "annualized_return_pct": round(ann * 100.0, 3) if ann is not None else None,
        "buy_hold_return_pct": round(benchmark_return * 100.0, 3),
        "excess_vs_buy_hold_pct": round((total_return - benchmark_return) * 100.0, 3),
        "max_drawdown_pct": round(max_dd * 100.0, 3),
        "sharpe_ratio": round(sharpe, 3) if sharpe is not None else None,
        "trades": int(len(trades)),
        "win_rate_pct": round(float((trade_returns > 0).mean() * 100.0), 2) if len(trade_returns) else None,
        "avg_trade_return_pct": round(float(trade_returns.mean() * 100.0), 3) if len(trade_returns) else None,
        "profit_factor": round(float(profit_factor), 3) if profit_factor is not None and np.isfinite(profit_factor) else ("inf" if profit_factor == float("inf") else None),
        "commission_bps": float(commission_bps),
        "slippage_bps": float(slippage_bps),
    }

    return {
        "latest_analysis": latest,
        "metrics": metrics,
        "signal_accuracy": accuracy,
        "trades": [asdict(t) for t in trades],
        "equity_curve": curve,
    }
