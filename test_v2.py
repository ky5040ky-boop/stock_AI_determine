import numpy as np
import pandas as pd

from backend.backtest_v2 import run_backtest
from backend.indicators_v2 import compute_all
from backend.judge_v2 import analyze_row


def synthetic_ohlcv(n=320, drift=0.0012):
    idx = pd.bdate_range("2024-01-01", periods=n)
    t = np.arange(n, dtype=float)
    close = 100.0 * np.exp(drift * t + 0.015 * np.sin(t / 8.0))
    open_ = close * (1.0 + 0.001 * np.sin(t / 3.0))
    high = np.maximum(open_, close) * 1.005
    low = np.minimum(open_, close) * 0.995
    volume = 1_000_000 * (1.0 + 0.15 * np.sin(t / 5.0))
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def test_indicators_have_warmup_and_latest_values():
    df = compute_all(synthetic_ohlcv())
    assert df["SMA75"].iloc[:74].isna().all()
    assert np.isfinite(df["SMA75"].iloc[-1])
    assert np.isfinite(df["RSI14"].iloc[-1])
    assert np.isfinite(df["ADX14"].iloc[-1])


def test_decision_is_in_expected_set():
    df = compute_all(synthetic_ohlcv())
    result = analyze_row(df.iloc[-1])
    assert result["decision"] in {"BUY", "HOLD", "SELL"}
    assert -1.0 <= result["score_normalized"] <= 1.0


def test_no_lookahead_for_latest_snapshot():
    raw = synthetic_ohlcv(260)
    full = compute_all(raw)
    cut = 210
    prefix = compute_all(raw.iloc[:cut])
    a = analyze_row(full.iloc[cut - 1])
    b = analyze_row(prefix.iloc[-1])
    assert a["decision"] == b["decision"]
    assert abs(a["score_normalized"] - b["score_normalized"]) < 1e-12


def test_backtest_returns_metrics_and_uses_costs():
    raw = synthetic_ohlcv(360)
    result = run_backtest(raw, commission_bps=10, slippage_bps=5, horizon=10)
    metrics = result["metrics"]
    assert "total_return_pct" in metrics
    assert "max_drawdown_pct" in metrics
    assert "buy_hold_return_pct" in metrics
    assert result["signal_accuracy"]["horizon_days"] == 10
    assert len(result["equity_curve"]) == len(raw)
