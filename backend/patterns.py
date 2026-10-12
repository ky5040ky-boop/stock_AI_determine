"""チャートパターン(ダブルトップ/ダブルボトム/三尊/逆三尊)の検出ロジック。

流れ:
1. 高値・安値を対数価格に変換し、ZigZag で山(H)と谷(L)の転換点を抽出する。
   最後の転換点は「まだ確定していない現在の値動き」を表す暫定ピボット。
2. 直近の転換点の並びが各パターンの形に当てはまるかを判定する。
3. 現在値とネックラインの位置関係から進行段階を決め、スコアを付ける。

天井型(ダブルトップ・三尊)だけを実装し、底型は価格を上下反転
(対数価格の符号を反転)して同じ判定に通すことで検出する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PATTERN_NAMES = {
    "double_top": "ダブルトップ",
    "double_bottom": "ダブルボトム",
    "head_and_shoulders": "三尊(ヘッド&ショルダー)",
    "inverse_head_and_shoulders": "逆三尊(逆ヘッド&ショルダー)",
}

STAGE_NAMES = {
    "second_peak": "2つ目の山/谷を形成中",
    "right_shoulder": "右肩を形成中",
    "approaching": "ネックラインへ接近中",
    "breakout": "ネックラインをブレイク直後",
}

# 判定パラメータ(対数価格の差 ≒ 変化率)
PEAK_TOLERANCE = 0.03       # ダブルトップの2つの山の高さの許容差(約3%)
HEAD_MIN_EXCESS = 0.015     # 三尊の頭が両肩より最低限高い幅(約1.5%)
MIN_PEAK_GAP_BARS = 8       # 2つの山の最小間隔(営業日)
MAX_PATTERN_BARS = 200      # パターン全体の最大期間(営業日)
BREAKOUT_MAX_AGE = 5        # ブレイク後、何営業日まで「直後」とみなすか


@dataclass
class Pivot:
    index: int
    value: float  # 対数価格
    kind: str     # "H" or "L"
    provisional: bool = False


def zigzag(high: np.ndarray, low: np.ndarray, threshold: float) -> list[Pivot]:
    """高値/安値(対数価格)から、threshold 以上の反転で確定する転換点列を返す。

    返り値は H と L が交互に並び、最後の要素は暫定(provisional)の転換点。
    """
    n = len(high)
    pivots: list[Pivot] = []
    if n < 2:
        return pivots

    trend = 0  # 1: 上昇中, -1: 下落中, 0: 未確定
    hi_i = lo_i = ext_i = 0
    for i in range(1, n):
        if trend == 0:
            if high[i] > high[hi_i]:
                hi_i = i
            if low[i] < low[lo_i]:
                lo_i = i
            if high[hi_i] - low[lo_i] >= threshold and hi_i != lo_i:
                if hi_i > lo_i:
                    pivots.append(Pivot(lo_i, low[lo_i], "L"))
                    trend, ext_i = 1, hi_i
                else:
                    pivots.append(Pivot(hi_i, high[hi_i], "H"))
                    trend, ext_i = -1, lo_i
        elif trend == 1:
            if high[i] >= high[ext_i]:
                ext_i = i
            elif high[ext_i] - low[i] >= threshold:
                pivots.append(Pivot(ext_i, high[ext_i], "H"))
                trend, ext_i = -1, i
        else:
            if low[i] <= low[ext_i]:
                ext_i = i
            elif high[i] - low[ext_i] >= threshold:
                pivots.append(Pivot(ext_i, low[ext_i], "L"))
                trend, ext_i = 1, i

    if trend == 1:
        pivots.append(Pivot(ext_i, high[ext_i], "H", provisional=True))
    elif trend == -1:
        pivots.append(Pivot(ext_i, low[ext_i], "L", provisional=True))
    return pivots


def adaptive_threshold(df: pd.DataFrame, sensitivity: float = 1.0) -> float:
    """銘柄のボラティリティに応じた ZigZag の反転幅(対数)を決める。

    sensitivity を大きくすると小さな波も拾う(検出数が増える)。
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    atr_pct = float((tr / close).tail(100).median())
    if not np.isfinite(atr_pct) or atr_pct <= 0:
        atr_pct = 0.015
    return float(np.clip(2.5 * atr_pct / max(sensitivity, 0.1), 0.03, 0.15))


def _volume_mean(volume: np.ndarray, center: int, half: int = 2) -> float:
    lo, hi = max(center - half, 0), min(center + half + 1, len(volume))
    seg = volume[lo:hi]
    seg = seg[np.isfinite(seg)]
    return float(seg.mean()) if len(seg) else float("nan")


def _volume_score(vol_first: float, vol_second: float) -> float:
    """2回目の山(右肩)で出来高が減っていれば高評価(買い/売りの勢いの衰え)。"""
    if not (np.isfinite(vol_first) and np.isfinite(vol_second)) or vol_first <= 0:
        return 0.5
    ratio = vol_second / vol_first
    return float(np.clip(1.5 - ratio, 0.0, 1.0))


def _stage_after_last_peak(
    hi: np.ndarray,
    close: np.ndarray,
    peak_i: int,
    peak_val: float,
    neckline_at,
    target_gap: float,
):
    """最後の山の後、現在値がネックラインに対してどこにいるかを判定する。

    neckline_at(i) はバー i におけるネックラインの値(対数)。
    返り値: (stage, progress) または None(パターンとしては期限切れ)。
    """
    last = len(close) - 1
    nl_now = neckline_at(last)
    c = close[last]
    if c > nl_now:
        height = peak_val - nl_now
        if height <= 0:
            return None
        progress = float(np.clip((peak_val - c) / height, 0.0, 1.0))
        return "approaching", progress

    # ネックライン割れ: 割れたのが直近で、まだ目標値に達していなければ「ブレイク直後」
    break_i = None
    for i in range(peak_i + 1, last + 1):
        if close[i] < neckline_at(i):
            break_i = i
            break
    if break_i is None or last - break_i > BREAKOUT_MAX_AGE:
        return None
    if c <= nl_now - target_gap:
        return None
    return "breakout", 1.0


def _detect_top_patterns(
    pivots: list[Pivot],
    hi: np.ndarray,
    lo: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    threshold: float,
) -> list[dict]:
    """天井型パターン(ダブルトップ/三尊)を対数価格上で検出する。"""
    results: list[dict] = []
    last_h = max((k for k, p in enumerate(pivots) if p.kind == "H"), default=None)
    if last_h is None or last_h < len(pivots) - 2:
        return results
    last = len(close) - 1
    rising_now = last_h == len(pivots) - 1  # 最後の山がまだ形成途中

    hs = _match_head_and_shoulders(pivots, last_h, hi, close, volume, threshold, rising_now)
    if hs:
        results.append(hs)
        return results  # 三尊が成立する場合、同じ山でのダブルトップ判定は省く

    k = last_h
    if k < 3:
        return results
    l0, h1, t, h2 = pivots[k - 3], pivots[k - 2], pivots[k - 1], pivots[k]
    peak_diff = abs(h1.value - h2.value)
    depth = min(h1.value, h2.value) - t.value
    prior_rise = h1.value - l0.value
    gap = h2.index - h1.index
    if peak_diff > PEAK_TOLERANCE or depth < threshold or prior_rise < depth * 0.8:
        return results
    if gap < MIN_PEAK_GAP_BARS or last - l0.index > MAX_PATTERN_BARS:
        return results
    if peak_diff > depth * 0.5:
        return results

    neckline = t.value
    target_gap = max(h1.value, h2.value) - neckline

    if rising_now:
        # 2つ目の山が1つ目の山と同水準まで戻ってきた段階
        if not (h1.value - PEAK_TOLERANCE <= close[last] <= h1.value + PEAK_TOLERANCE * 0.5):
            return results
        stage, progress = "second_peak", 0.0
    else:
        st = _stage_after_last_peak(hi, close, h2.index, h2.value, lambda i: neckline, target_gap)
        if st is None:
            return results
        stage, progress = st

    vol_s = _volume_score(_volume_mean(volume, h1.index), _volume_mean(volume, h2.index))
    results.append(
        _build_result(
            "double_top",
            stage,
            progress,
            similarity=1 - peak_diff / PEAK_TOLERANCE,
            size=min(depth / (2 * threshold), 1.0),
            trend=min(prior_rise / (2 * depth), 1.0),
            volume=vol_s,
            points=[(l0, "起点"), (h1, "山1"), (t, "谷(ネックライン)"), (h2, "山2")],
            neckline_line=[(t.index, neckline), (last, neckline)],
            neckline_now=neckline,
            target=neckline - target_gap,
        )
    )
    return results


def _match_head_and_shoulders(pivots, k, hi, close, volume, threshold, rising_now):
    if k < 5:
        return None
    l0, ls, t1, head, t2, rs = pivots[k - 5 : k + 1]
    last = len(close) - 1
    if last - l0.index > MAX_PATTERN_BARS:
        return None

    shoulder_top = max(ls.value, rs.value)
    if head.value - shoulder_top < HEAD_MIN_EXCESS:
        return None
    if l0.value >= t1.value:  # 直前に上昇トレンドがあること
        return None

    slope = (t2.value - t1.value) / (t2.index - t1.index)

    def neckline_at(i):
        return t2.value + slope * (i - t2.index)

    head_height = head.value - max(t1.value, t2.value)
    if head_height < threshold:
        return None
    # ネックラインの傾きが極端でないこと
    if abs(t1.value - t2.value) > head_height * 0.6:
        return None
    # 両肩の高さが近いこと
    if abs(ls.value - rs.value) > head_height * 0.6:
        return None
    # 左肩はネックラインより十分上
    if ls.value - neckline_at(ls.index) < head_height * 0.25:
        return None
    # 時間的な対称性(右肩形成中は判定を緩める)
    left_span = head.index - ls.index
    right_span = rs.index - head.index
    if right_span <= 0 or left_span <= 0:
        return None
    if not rising_now and not (1 / 3 <= left_span / right_span <= 3):
        return None

    target_gap = head.value - neckline_at(head.index)
    if rising_now:
        # 右肩: 左肩付近まで戻したが頭は超えていない
        c = close[last]
        if not (neckline_at(last) < c < head.value - HEAD_MIN_EXCESS * 0.5):
            return None
        if rs.value < ls.value - head_height * 0.5:
            return None
        stage, progress = "right_shoulder", 0.0
    else:
        if rs.value - neckline_at(rs.index) < head_height * 0.25:
            return None
        st = _stage_after_last_peak(hi, close, rs.index, rs.value, neckline_at, target_gap)
        if st is None:
            return None
        stage, progress = st

    nl_now = neckline_at(last)
    vol_s = _volume_score(_volume_mean(volume, head.index), _volume_mean(volume, rs.index))
    return _build_result(
        "head_and_shoulders",
        stage,
        progress,
        similarity=1 - abs(ls.value - rs.value) / (head_height * 0.6),
        size=min(head_height / (2 * threshold), 1.0),
        trend=min((ls.value - l0.value) / head_height, 1.0),
        volume=vol_s,
        points=[(l0, "起点"), (ls, "左肩"), (t1, "谷1"), (head, "頭"), (t2, "谷2"), (rs, "右肩")],
        neckline_line=[(t1.index, t1.value), (last, nl_now)],
        neckline_now=nl_now,
        target=nl_now - target_gap,
    )


_STAGE_WEIGHT = {"second_peak": 0.25, "right_shoulder": 0.3, "approaching": 0.45, "breakout": 1.0}


def _build_result(kind, stage, progress, similarity, size, trend, volume, points,
                  neckline_line, neckline_now, target):
    stage_s = _STAGE_WEIGHT[stage]
    if stage == "approaching":
        stage_s += (1.0 - stage_s - 0.1) * progress
    score = (
        30 * stage_s
        + 25 * float(np.clip(similarity, 0, 1))
        + 15 * float(np.clip(size, 0, 1))
        + 15 * float(np.clip(trend, 0, 1))
        + 15 * float(np.clip(volume, 0, 1))
    )
    return {
        "kind": kind,
        "stage": stage,
        "progress": progress,
        "score": round(score, 1),
        "points": [(p.index, p.value, label) for p, label in points],
        "neckline_line": neckline_line,
        "neckline_now": neckline_now,
        "target": target,
    }


_MIRROR = {"double_top": "double_bottom", "head_and_shoulders": "inverse_head_and_shoulders"}
_MIRROR_LABEL = {"山1": "谷1", "山2": "谷2", "谷(ネックライン)": "山(ネックライン)", "谷1": "山1", "谷2": "山2"}


def detect_patterns(df: pd.DataFrame, sensitivity: float = 1.0) -> list[dict]:
    """OHLCV データフレームから、形成中/直近完成のチャートパターンを検出する。

    df は日付インデックスで High/Low/Close 列を持つこと(Volume は任意)。
    """
    df = df.dropna(subset=["High", "Low", "Close"])
    if len(df) < 40:
        return []

    hi = np.log(df["High"].to_numpy(dtype=float))
    lo = np.log(df["Low"].to_numpy(dtype=float))
    close = np.log(df["Close"].to_numpy(dtype=float))
    volume = (
        df["Volume"].to_numpy(dtype=float) if "Volume" in df else np.full(len(df), np.nan)
    )
    threshold = adaptive_threshold(df, sensitivity)

    raw: list[tuple[dict, bool]] = []
    # 天井型: そのまま
    for r in _detect_top_patterns(zigzag(hi, lo, threshold), hi, lo, close, volume, threshold):
        raw.append((r, False))
    # 底型: 価格を上下反転して天井型として判定
    for r in _detect_top_patterns(zigzag(-lo, -hi, threshold), -lo, -hi, -close, volume, threshold):
        r["kind"] = _MIRROR[r["kind"]]
        raw.append((r, True))

    dates = df.index
    last_close = float(df["Close"].iloc[-1])

    def to_price(v, mirrored):
        return round(float(np.exp(-v if mirrored else v)), 4)

    out = []
    for r, mirrored in raw:
        neckline = to_price(r["neckline_now"], mirrored)
        out.append(
            {
                "pattern": r["kind"],
                "pattern_name": PATTERN_NAMES[r["kind"]],
                "bias": "bullish" if mirrored else "bearish",
                "stage": r["stage"],
                "stage_name": STAGE_NAMES[r["stage"]],
                "progress": round(r["progress"], 3),
                "score": r["score"],
                "close": round(last_close, 4),
                "neckline": neckline,
                "target": to_price(r["target"], mirrored),
                "distance_to_neckline_pct": round((neckline / last_close - 1) * 100, 2),
                "points": [
                    {
                        "date": _date(dates[i]),
                        "price": to_price(v, mirrored),
                        "label": _MIRROR_LABEL.get(label, label) if mirrored else label,
                    }
                    for i, v, label in r["points"]
                ],
                "neckline_line": [
                    {"date": _date(dates[i]), "price": to_price(v, mirrored)}
                    for i, v in r["neckline_line"]
                ],
                "zigzag_threshold_pct": round(threshold * 100, 2),
            }
        )
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def _date(idx) -> str:
    return idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
