"""チャートパターンの検出ロジック。

対応パターン:
- ダブルトップ / ダブルボトム
- 三尊(ヘッド&ショルダー) / 逆三尊
- 上昇三角形 / 下降三角形 / 対称三角形

流れ:
1. 高値・安値を対数価格に変換し、ZigZag で山(H)と谷(L)の転換点を抽出する。
   最後の転換点は「まだ確定していない現在の値動き」を表す暫定ピボット。
2. 直近の転換点の並びが各パターンの形に当てはまるかを判定する。
3. 現在値とネックライン(ブレイクライン)の位置関係から進行段階を決め、スコアを付ける。

天井型(ダブルトップ・三尊)と上昇三角形だけを実装し、底型と下降三角形は
価格を上下反転(対数価格の符号を反転)して同じ判定に通すことで検出する。
対数価格の差はおおよそ変化率(0.03 ≒ 3%)として扱える。

バックテスト(backend/pattern_backtest.py)と画面表示で同じ判定を使うため、
判定本体 detect_raw() は numpy 配列だけを受け取り、未来のデータを参照しない。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

PATTERN_NAMES = {
    "double_top": "ダブルトップ",
    "double_bottom": "ダブルボトム",
    "head_and_shoulders": "三尊(ヘッド&ショルダー)",
    "inverse_head_and_shoulders": "逆三尊(逆ヘッド&ショルダー)",
    "ascending_triangle": "上昇三角形",
    "descending_triangle": "下降三角形",
    "symmetric_triangle": "対称三角形",
}

STAGE_NAMES = {
    "second_peak": "2つ目の山/谷を形成中",
    "right_shoulder": "右肩を形成中",
    "approaching": "ネックラインへ接近中",
    "converging": "三角形が収束中(ブレイク待ち)",
    "breakout": "ブレイク直後",
}

# ブレイク前の段階(バックテストで「パターンが完成したか」を測る対象)
PRE_BREAKOUT_STAGES = {"second_peak", "right_shoulder", "approaching", "converging"}

LOOKBACK_BARS = 250  # 判定に使う直近の営業日数(約1年)

# 過去データで学習した確率モデル(python -m backend.pattern_backtest --fit-model で再作成)
MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "pattern_model.json"
MODEL_FEATURES = ("progress", "shape", "size", "trend", "volume")


@dataclass(frozen=True)
class Params:
    """判定パラメータ。値は対数価格の差(≒変化率)か営業日数。"""

    atr_mult: float = 2.5            # ZigZag の反転幅 = ATR% × atr_mult
    min_threshold: float = 0.03
    max_threshold: float = 0.15
    peak_tolerance: float = 0.03     # ダブルトップの2つの山の高さの許容差
    head_min_excess: float = 0.015   # 三尊の頭が両肩より最低限高い幅
    min_peak_gap: int = 8            # 2つの山の最小間隔
    max_pattern_bars: int = 200      # パターン全体の最大期間
    breakout_max_age: int = 5        # ブレイク後、何営業日まで「直後」とみなすか
    tri_min_bars: int = 15           # 三角形の最小期間
    tri_max_bars: int = 150          # 三角形の最大期間
    tri_flat: float = 0.15           # 水平線とみなす傾き(期間中の変化 / 初期の幅)
    tri_slope: float = 0.3           # 傾いているとみなす最小の傾き(同上)
    tri_fit: float = 0.2             # 山・谷が線から外れてよい幅(初期の幅に対する比)
    # スコアの配点(合計100)
    weights: dict = field(default_factory=lambda: {
        "stage": 30, "shape": 25, "size": 15, "trend": 15, "volume": 15,
    })


DEFAULT_PARAMS = Params()


@dataclass
class Pivot:
    index: int
    value: float  # 対数価格
    kind: str     # "H" or "L"
    provisional: bool = False


@dataclass
class Line:
    """index i における値 = value + slope * (i - anchor)。"""

    anchor: int
    value: float
    slope: float = 0.0

    def at(self, i):
        return self.value + self.slope * (i - self.anchor)

    def mirrored(self) -> "Line":
        return Line(self.anchor, -self.value, -self.slope)


@dataclass
class RawPattern:
    """対数価格・配列インデックスで表した検出結果(内部用)。"""

    kind: str
    direction: int          # +1: 上昇示唆, -1: 下落示唆, 0: 方向未定
    stage: str
    progress: float
    score: float
    components: dict
    points: list            # [(index, value, label)]
    lines: list             # 描画用の線 [Line]
    breakout_line: Line     # ブレイク判定に使う線(ネックライン)
    target: float           # 目標値(対数)
    stop: float             # パターン否定の目安(対数)

    def mirrored(self) -> "RawPattern":
        return replace(
            self,
            kind=_MIRROR_KIND.get(self.kind, self.kind),
            direction=-self.direction,
            points=[(i, -v, _MIRROR_LABEL.get(lb, lb)) for i, v, lb in self.points],
            lines=[ln.mirrored() for ln in self.lines],
            breakout_line=self.breakout_line.mirrored(),
            target=-self.target,
            stop=-self.stop,
        )


_MIRROR_KIND = {
    "double_top": "double_bottom",
    "head_and_shoulders": "inverse_head_and_shoulders",
    "ascending_triangle": "descending_triangle",
}
_MIRROR_LABEL = {
    "山1": "谷1", "山2": "谷2", "谷(ネックライン)": "山(ネックライン)",
    "谷1": "山1", "谷2": "山2", "山": "谷", "谷": "山",
}


# ---------------------------------------------------------------------------
# ZigZag
# ---------------------------------------------------------------------------

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


def true_range_pct(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """(実数価格の)真の値幅 / 終値。"""
    prev = np.concatenate([[np.nan], close[:-1]])
    tr = np.fmax(high - low, np.fmax(np.abs(high - prev), np.abs(low - prev)))
    return tr / close


def threshold_from_atr(atr_pct: float, params: Params = DEFAULT_PARAMS,
                       sensitivity: float = 1.0) -> float:
    """銘柄のボラティリティ(直近100日の真の値幅%の中央値)から ZigZag の反転幅を決める。

    sensitivity を大きくすると小さな波も拾う(検出数が増える)。
    """
    if not np.isfinite(atr_pct) or atr_pct <= 0:
        atr_pct = 0.015
    t = params.atr_mult * atr_pct / max(sensitivity, 0.1)
    return float(np.clip(t, params.min_threshold, params.max_threshold))


# ---------------------------------------------------------------------------
# 共通ヘルパ
# ---------------------------------------------------------------------------

def _volume_mean(volume: np.ndarray, lo: int, hi: int) -> float:
    seg = volume[max(lo, 0):min(hi, len(volume))]
    seg = seg[np.isfinite(seg)]
    return float(seg.mean()) if len(seg) else float("nan")


def _volume_decline_score(vol_first: float, vol_second: float) -> float:
    """後半で出来高が減っていれば高評価(それまでの勢いの衰え)。"""
    if not (np.isfinite(vol_first) and np.isfinite(vol_second)) or vol_first <= 0:
        return 0.5
    return float(np.clip(1.5 - vol_second / vol_first, 0.0, 1.0))


def _first_cross_below(close: np.ndarray, line: Line, start: int):
    for i in range(start, len(close)):
        if close[i] < line.at(i):
            return i
    return None


def _stage_after_last_peak(close, peak_i, peak_val, neckline: Line, target_gap, params):
    """最後の山の後、現在値がネックラインに対してどこにいるかを判定する。

    返り値: (stage, progress) または None(パターンとしては期限切れ)。
    """
    last = len(close) - 1
    nl_now = neckline.at(last)
    c = close[last]
    if c > nl_now:
        height = peak_val - nl_now
        if height <= 0:
            return None
        return "approaching", float(np.clip((peak_val - c) / height, 0.0, 1.0))

    # ネックライン割れ: 割れたのが直近で、まだ目標値に達していなければ「ブレイク直後」
    break_i = _first_cross_below(close, neckline, peak_i + 1)
    if break_i is None or last - break_i > params.breakout_max_age:
        return None
    if c <= nl_now - target_gap:
        return None
    return "breakout", 1.0


_STAGE_BASE = {
    "second_peak": 0.25, "right_shoulder": 0.3, "approaching": 0.45,
    "converging": 0.35, "breakout": 1.0,
}


def _score(stage, progress, components, params: Params) -> float:
    stage_s = _STAGE_BASE[stage]
    if stage in ("approaching", "converging"):
        stage_s += (0.9 - stage_s) * progress
    comps = {"stage": stage_s, **components}
    total = sum(params.weights[k] * float(np.clip(comps.get(k, 0.5), 0, 1)) for k in params.weights)
    return round(total * 100 / sum(params.weights.values()), 1)


# ---------------------------------------------------------------------------
# ダブルトップ / 三尊(天井型)
# ---------------------------------------------------------------------------

def _detect_tops(pivots, close, volume, threshold, params) -> list[RawPattern]:
    last_h = max((k for k, p in enumerate(pivots) if p.kind == "H"), default=None)
    if last_h is None or last_h < len(pivots) - 2:
        return []
    rising_now = last_h == len(pivots) - 1  # 最後の山がまだ形成途中

    hs = _match_head_and_shoulders(pivots, last_h, close, volume, threshold, rising_now, params)
    if hs:
        return [hs]  # 三尊が成立する場合、同じ山でのダブルトップ判定は省く
    dt = _match_double_top(pivots, last_h, close, volume, threshold, rising_now, params)
    return [dt] if dt else []


def _match_double_top(pivots, k, close, volume, threshold, rising_now, params):
    if k < 3:
        return None
    last = len(close) - 1
    l0, h1, t, h2 = pivots[k - 3:k + 1]
    peak_diff = abs(h1.value - h2.value)
    depth = min(h1.value, h2.value) - t.value
    prior_rise = h1.value - l0.value
    if peak_diff > params.peak_tolerance or peak_diff > depth * 0.5:
        return None
    if depth < threshold or prior_rise < depth * 0.8:
        return None
    if h2.index - h1.index < params.min_peak_gap or last - h1.index > params.max_pattern_bars:
        return None

    neckline = Line(t.index, t.value)
    target_gap = max(h1.value, h2.value) - t.value
    if rising_now:
        # 2つ目の山が1つ目の山と同水準まで戻ってきた段階
        if not (h1.value - params.peak_tolerance <= close[last] <= h1.value + params.peak_tolerance * 0.5):
            return None
        stage, progress = "second_peak", 0.0
    else:
        st = _stage_after_last_peak(close, h2.index, h2.value, neckline, target_gap, params)
        if st is None:
            return None
        stage, progress = st

    comps = {
        "shape": 1 - peak_diff / params.peak_tolerance,
        "size": depth / (2 * threshold),
        "trend": prior_rise / (2 * depth),
        "volume": _volume_decline_score(
            _volume_mean(volume, h1.index - 2, h1.index + 3),
            _volume_mean(volume, h2.index - 2, h2.index + 3),
        ),
    }
    return RawPattern(
        kind="double_top", direction=-1, stage=stage, progress=progress,
        score=_score(stage, progress, comps, params), components=comps,
        points=[(l0.index, l0.value, "起点"), (h1.index, h1.value, "山1"),
                (t.index, t.value, "谷(ネックライン)"), (h2.index, h2.value, "山2")],
        lines=[neckline], breakout_line=neckline,
        target=t.value - target_gap,
        stop=max(h1.value, h2.value) + (params.peak_tolerance if rising_now else 0.0),
    )


def _match_head_and_shoulders(pivots, k, close, volume, threshold, rising_now, params):
    if k < 5:
        return None
    l0, ls, t1, head, t2, rs = pivots[k - 5:k + 1]
    last = len(close) - 1
    if last - ls.index > params.max_pattern_bars:
        return None
    if head.value - max(ls.value, rs.value) < params.head_min_excess:
        return None
    if l0.value >= t1.value:  # 直前に上昇トレンドがあること
        return None

    neckline = Line(t2.index, t2.value, (t2.value - t1.value) / (t2.index - t1.index))
    head_height = head.value - max(t1.value, t2.value)
    if head_height < threshold:
        return None
    if abs(t1.value - t2.value) > head_height * 0.6:   # ネックラインの傾きが極端でない
        return None
    if abs(ls.value - rs.value) > head_height * 0.6:   # 両肩の高さが近い
        return None
    if ls.value - neckline.at(ls.index) < head_height * 0.25:
        return None
    left_span, right_span = head.index - ls.index, rs.index - head.index
    if right_span <= 0 or left_span <= 0:
        return None
    if not rising_now and not (1 / 3 <= left_span / right_span <= 3):
        return None

    target_gap = head.value - neckline.at(head.index)
    if rising_now:
        # 右肩: 左肩付近まで戻したが頭は超えていない
        c = close[last]
        if not (neckline.at(last) < c < head.value - params.head_min_excess * 0.5):
            return None
        if rs.value < ls.value - head_height * 0.5:
            return None
        stage, progress = "right_shoulder", 0.0
    else:
        if rs.value - neckline.at(rs.index) < head_height * 0.25:
            return None
        st = _stage_after_last_peak(close, rs.index, rs.value, neckline, target_gap, params)
        if st is None:
            return None
        stage, progress = st

    comps = {
        "shape": 1 - abs(ls.value - rs.value) / (head_height * 0.6),
        "size": head_height / (2 * threshold),
        "trend": (ls.value - l0.value) / head_height,
        "volume": _volume_decline_score(
            _volume_mean(volume, head.index - 2, head.index + 3),
            _volume_mean(volume, rs.index - 2, rs.index + 3),
        ),
    }
    return RawPattern(
        kind="head_and_shoulders", direction=-1, stage=stage, progress=progress,
        score=_score(stage, progress, comps, params), components=comps,
        points=[(l0.index, l0.value, "起点"), (ls.index, ls.value, "左肩"),
                (t1.index, t1.value, "谷1"), (head.index, head.value, "頭"),
                (t2.index, t2.value, "谷2"), (rs.index, rs.value, "右肩")],
        lines=[Line(t1.index, t1.value, neckline.slope)], breakout_line=neckline,
        target=neckline.at(last) - target_gap,
        stop=head.value,
    )


# ---------------------------------------------------------------------------
# 三角保ち合い
# ---------------------------------------------------------------------------

def _fit_line(pts: list[Pivot]) -> Line:
    x = np.array([p.index for p in pts], dtype=float)
    y = np.array([p.value for p in pts])
    slope, intercept = np.polyfit(x, y, 1)
    return Line(int(x[0]), float(intercept + slope * x[0]), float(slope))


def _detect_triangles(pivots, close, volume, threshold, params, allow_symmetric=True):
    """上昇三角形(上値水平・下値切り上げ)と対称三角形を検出する。

    下降三角形は価格を上下反転した上昇三角形として検出する。
    """
    confirmed = [p for p in pivots if not p.provisional]
    last = len(close) - 1
    for m in (6, 5, 4):  # 接点の多い三角形を優先
        if len(confirmed) < m:
            continue
        seg = confirmed[-m:]
        highs = [p for p in seg if p.kind == "H"]
        lows = [p for p in seg if p.kind == "L"]
        start = seg[0].index
        span = last - start
        if not (params.tri_min_bars <= span <= params.tri_max_bars):
            continue
        upper, lower = _fit_line(highs), _fit_line(lows)
        w0 = upper.at(start) - lower.at(start)
        w_now = upper.at(last) - lower.at(last)
        if w0 < threshold or w_now <= 0.15 * w0 or w_now > 0.8 * w0:
            continue  # 十分な値幅がない・収束していない・頂点を過ぎている
        resid = max(
            max(abs(p.value - upper.at(p.index)) for p in highs),
            max(abs(p.value - lower.at(p.index)) for p in lows),
        )
        if resid > params.tri_fit * w0:
            continue
        ru = (upper.at(last) - upper.at(start)) / w0
        rl = (lower.at(last) - lower.at(start)) / w0
        if abs(ru) <= params.tri_flat and rl >= params.tri_slope:
            kind = "ascending_triangle"
        elif allow_symmetric and ru <= -params.tri_slope and rl >= params.tri_slope:
            kind = "symmetric_triangle"
        else:
            continue
        result = _triangle_stage(kind, seg, upper, lower, w0, w_now, close, volume,
                                 threshold, resid, params)
        if result:
            return [result]
        return []  # 形は三角形だが期限切れ/否定 → 小さい m で再判定しない
    return []


def _triangle_stage(kind, seg, upper, lower, w0, w_now, close, volume, threshold, resid, params):
    last = len(close) - 1
    c = close[last]
    after = seg[-1].index + 1

    def first_break(direction):
        for i in range(after, last + 1):
            if (direction > 0 and close[i] > upper.at(i)) or (direction < 0 and close[i] < lower.at(i)):
                return i
        return None

    if lower.at(last) <= c <= upper.at(last):
        stage, progress = "converging", float(np.clip((1 - w_now / w0) / 0.75, 0, 1))
        if kind == "ascending_triangle":
            direction = 1
        else:  # 対称三角形はブレイクするまで方向未定。近い方の線を表示上のネックラインにする
            direction = 0
    else:
        direction = 1 if c > upper.at(last) else -1
        if kind == "ascending_triangle" and direction < 0:
            return None  # 下値支持線割れ = パターン否定
        bi = first_break(direction)
        if bi is None or last - bi > params.breakout_max_age:
            return None
        stage, progress = "breakout", 1.0

    if direction > 0:
        bl, stop_line = upper, lower
    elif direction < 0:
        bl, stop_line = lower, upper
    else:
        near_upper = upper.at(last) - c <= c - lower.at(last)
        bl, stop_line = (upper, lower) if near_upper else (lower, upper)
    sign = 1 if bl is upper else -1
    if stage == "breakout" and sign * (c - (bl.at(last) + sign * w0)) >= 0:
        return None  # 目標値に到達済み

    start = seg[0].index
    third = max((last - start) // 3, 1)
    comps = {
        "shape": 1 - resid / (params.tri_fit * w0),
        "size": w0 / (2 * threshold),
        "trend": (len(seg) - 4) / 2,  # 接点の数(4〜6)
        "volume": _volume_decline_score(
            _volume_mean(volume, start, start + third),
            _volume_mean(volume, last - third + 1, last + 1),
        ),
    }
    labels = {"H": "山", "L": "谷"}
    return RawPattern(
        kind=kind, direction=direction, stage=stage, progress=progress,
        score=_score(stage, progress, comps, params), components=comps,
        points=[(p.index, p.value, labels[p.kind]) for p in seg],
        lines=[upper, lower], breakout_line=bl,
        target=bl.at(last) + sign * w0,
        stop=stop_line.at(last),
    )


# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------

def detect_raw(hi, lo, close, volume, threshold, params: Params = DEFAULT_PARAMS) -> list[RawPattern]:
    """対数価格の配列から全パターンを検出する(未来のデータは参照しない)。"""
    out: list[RawPattern] = []
    piv = zigzag(hi, lo, threshold)
    out += _detect_tops(piv, close, volume, threshold, params)
    out += _detect_triangles(piv, close, volume, threshold, params, allow_symmetric=True)
    # 底型: 価格を上下反転して天井型として判定
    mpiv = zigzag(-lo, -hi, threshold)
    for r in _detect_tops(mpiv, -close, volume, threshold, params):
        out.append(r.mirrored())
    for r in _detect_triangles(mpiv, -close, volume, threshold, params, allow_symmetric=False):
        out.append(r.mirrored())
    return out


@lru_cache(maxsize=1)
def load_model(path: str = str(MODEL_PATH)) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def model_features(r: RawPattern) -> dict:
    return {"progress": r.progress, **{k: float(np.clip(v, 0, 1)) for k, v in r.components.items()}}


def predict_probabilities(r: RawPattern, model: dict | None = None) -> dict:
    """学習済みモデルで「完成確率」「目標到達確率」を推定する(モデルがなければ None)。

    - complete: ブレイク前の段階から、否定される前にネックラインを抜ける確率(40営業日以内)
    - target:   否定ラインより先に目標値へ到達する確率(60営業日以内)
    """
    model = model if model is not None else load_model()
    out = {"complete": None, "target": None}
    if not model:
        return out
    feats = model_features(r)
    group = f"{r.kind}:{r.stage}"
    for task, m in model["models"].items():
        if group not in m["group"]:
            continue
        z = m["bias"] + m["group"][group] + sum(m["coef"][f] * feats.get(f, 0.5) for f in m["coef"])
        out[task] = float(1 / (1 + np.exp(-z)))
    return out


def detect_patterns(df: pd.DataFrame, sensitivity: float = 1.0,
                    params: Params = DEFAULT_PARAMS) -> list[dict]:
    """OHLCV データフレームから、形成中/直近完成のチャートパターンを検出する。

    df は日付インデックスで High/Low/Close 列を持つこと(Volume は任意)。
    判定には直近 LOOKBACK_BARS 本を使う。
    """
    df = df.dropna(subset=["High", "Low", "Close"]).tail(LOOKBACK_BARS)
    if len(df) < 40:
        return []

    h = df["High"].to_numpy(dtype=float)
    l_ = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    volume = df["Volume"].to_numpy(dtype=float) if "Volume" in df else np.full(len(df), np.nan)
    atr_pct = float(np.nanmedian(true_range_pct(h, l_, c)[-100:]))
    threshold = threshold_from_atr(atr_pct, params, sensitivity)

    raws = detect_raw(np.log(h), np.log(l_), np.log(c), volume, threshold, params)
    dates = df.index
    last = len(df) - 1
    last_close = float(c[-1])

    def price(v):
        return round(float(np.exp(v)), 4)

    model = load_model()
    out = []
    for r in raws:
        neckline = price(r.breakout_line.at(last))
        probs = predict_probabilities(r, model)
        pct = {k: (round(v * 100, 1) if v is not None else None) for k, v in probs.items()}
        if model:
            score = pct["target"]  # 方向未定(対称三角形の収束中)は None
        else:
            score = r.score        # モデルがない場合は形の整い具合による簡易スコア
        out.append({
            "pattern": r.kind,
            "pattern_name": PATTERN_NAMES[r.kind],
            "bias": {1: "bullish", -1: "bearish", 0: "neutral"}[r.direction],
            "stage": r.stage,
            "stage_name": STAGE_NAMES[r.stage],
            "progress": round(r.progress, 3),
            "score": score,
            "p_complete": pct["complete"],
            "p_target": pct["target"],
            "shape_score": r.score,
            "close": round(last_close, 4),
            "neckline": neckline,
            "target": price(r.target),
            "stop": price(r.stop),
            "distance_to_neckline_pct": round((neckline / last_close - 1) * 100, 2),
            "points": [
                {"date": _date(dates[i]), "price": price(v), "label": label}
                for i, v, label in r.points
            ],
            "lines": [
                [{"date": _date(dates[ln.anchor]), "price": price(ln.value)},
                 {"date": _date(dates[last]), "price": price(ln.at(last))}]
                for ln in r.lines
            ],
            "zigzag_threshold_pct": round(threshold * 100, 2),
        })
    out.sort(key=lambda x: -1 if x["score"] is None else x["score"], reverse=True)
    return out


def _date(idx) -> str:
    return idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
