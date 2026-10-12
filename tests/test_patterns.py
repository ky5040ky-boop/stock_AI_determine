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
