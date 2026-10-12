import numpy as np
import pandas as pd

from backend.patterns import detect_patterns, zigzag


def path_from_waypoints(waypoints, noise=0.003, seed=0):
    """(営業日数, 価格) の折れ線からOHLCVを作る。"""
    rng = np.random.default_rng(seed)
    closes = [waypoints[0][1]]
    for (n, p) in waypoints[1:]:
        start = closes[-1]
        closes.extend(np.linspace(start, p, n + 1)[1:])
    close = np.array(closes) * (1 + rng.normal(0, noise, len(closes)))
    idx = pd.bdate_range("2025-01-01", periods=len(close))
    high = close * 1.006
    low = close * 0.994
    volume = np.full(len(close), 1_000_000.0)
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx
    )


def kinds(df):
    return {r["pattern"]: r for r in detect_patterns(df)}


def test_zigzag_alternates_and_ends_provisional():
    df = path_from_waypoints([(0, 100), (20, 120), (20, 105), (20, 125), (10, 118)])
    piv = zigzag(np.log(df["High"].values), np.log(df["Low"].values), 0.05)
    assert all(a.kind != b.kind for a, b in zip(piv, piv[1:]))
    assert piv[-1].provisional and not any(p.provisional for p in piv[:-1])


def test_double_top_approaching_neckline():
    # 上昇 → 山1(130) → 谷(115) → 山2(130) → 下落中(120)、ネックライン115はまだ割れていない
    df = path_from_waypoints([(0, 100), (40, 130), (20, 115), (20, 130), (10, 120)])
    r = kinds(df)["double_top"]
    assert r["bias"] == "bearish"
    assert r["stage"] == "approaching"
    assert abs(r["neckline"] - 115) < 3
    assert r["target"] < r["neckline"]


def test_double_bottom_forming_second_trough():
    df = path_from_waypoints([(0, 140), (40, 110), (20, 125), (20, 111)])
    r = kinds(df)["double_bottom"]
    assert r["bias"] == "bullish"
    assert r["stage"] == "second_peak"
    assert [p["label"] for p in r["points"]][1:] == ["谷1", "山(ネックライン)", "谷2"]


def test_inverse_head_and_shoulders_near_neckline():
    df = path_from_waypoints(
        [(0, 140), (30, 110), (15, 122), (15, 100), (15, 122), (15, 111), (10, 119)]
    )
    found = kinds(df)
    r = found["inverse_head_and_shoulders"]
    assert r["stage"] == "approaching"
    assert "double_bottom" not in found
    assert r["target"] > r["neckline"]


def test_head_and_shoulders_breakout():
    df = path_from_waypoints(
        [(0, 90), (30, 120), (15, 110), (15, 132), (15, 110), (15, 121), (12, 108)]
    )
    r = kinds(df)["head_and_shoulders"]
    assert r["stage"] == "breakout"


def test_steady_uptrend_has_no_patterns():
    df = path_from_waypoints([(0, 100), (200, 180)], noise=0.002)
    assert detect_patterns(df) == []


def test_stale_double_top_is_ignored():
    # ネックラインを割ってから大きく下落済み → 発生済みなので出力しない
    df = path_from_waypoints([(0, 100), (40, 130), (20, 115), (20, 130), (40, 85)])
    assert "double_top" not in kinds(df)


def test_ascending_triangle_converging():
    # 上値120で水平、下値は100→108→113と切り上がる
    df = path_from_waypoints(
        [(0, 90), (20, 120), (15, 100), (15, 120), (12, 108), (12, 120), (8, 113), (5, 117)]
    )
    r = kinds(df)["ascending_triangle"]
    assert r["bias"] == "bullish"
    assert r["stage"] == "converging"
    assert abs(r["neckline"] - 120 * 1.006) < 2
    assert r["target"] > r["neckline"] > r["close"] > r["stop"]


def test_descending_triangle_breakout():
    df = path_from_waypoints(
        [(0, 140), (20, 100), (15, 125), (15, 100), (12, 117), (12, 100), (8, 110), (6, 97)]
    )
    r = kinds(df)["descending_triangle"]
    assert r["bias"] == "bearish"
    assert r["stage"] == "breakout"


def test_symmetric_triangle_direction_pending():
    df = path_from_waypoints(
        [(0, 90), (20, 130), (15, 100), (15, 124), (12, 106), (12, 119), (8, 110), (4, 114)]
    )
    r = kinds(df)["symmetric_triangle"]
    assert r["bias"] == "neutral"
    assert r["stage"] == "converging"
    assert len(r["lines"]) == 2


def test_detection_uses_only_past_data():
    from backend.pattern_backtest import detect_at

    df = path_from_waypoints([(0, 100), (40, 130), (20, 115), (20, 130), (10, 120), (30, 140)])
    t = 90  # ダブルトップ形成中の時点
    from_prefix = detect_patterns(df.iloc[: t + 1])
    from_full = detect_at(df, t)
    assert [r["pattern"] for r in from_prefix] == [r.kind for r in from_full]
    assert [r["shape_score"] for r in from_prefix] == [r.score for r in from_full]


def test_backtest_records_completion_and_target():
    from backend.pattern_backtest import WARMUP, backtest_ticker, fit_model, summarize
    from backend.patterns import DEFAULT_PARAMS

    # 1年分の助走のあとダブルトップ → ネックライン割れ → 目標値まで下落
    df = path_from_waypoints(
        [(0, 100), (WARMUP - 40, 101), (40, 130), (20, 115), (20, 130), (20, 112), (40, 90), (60, 92)],
        noise=0.002,
    )
    res = backtest_ticker(("TEST", df, DEFAULT_PARAMS, 1.0))
    dt = [e for e in res["events"] if e["pattern"] == "double_top"]
    approaching = next(e for e in dt if e["stage"] == "approaching")
    assert approaching["completed"] == "yes"
    assert approaching["outcome"] == "target"
    assert approaching["ret_20"] > 0  # 下落示唆なので下がれば正
    assert any(e["stage"] == "breakout" for e in dt)

    s = summarize({"events": res["events"], "baseline": res["baseline"], "tickers": ["TEST"]})
    assert s["by_pattern"]["double_top"]["approaching"]["completion_rate"] == 100.0
    model = fit_model(res["events"], min_group=1)
    assert "double_top:approaching" in model["models"]["target"]["group"]
