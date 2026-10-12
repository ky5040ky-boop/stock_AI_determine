"""チャートパターンの的中率検証(ウォークフォワード・バックテスト)。

各営業日 t について「t 日の終値までのデータだけ」でパターンを判定し、
あるパターン・進行段階が新たに検出された日をイベントとして記録する。
その後の値動きから以下を測る(エントリーは翌営業日の始値)。

- 方向別リターン: 示唆の方向(上昇なら+、下落なら符号反転)で見た 5/10/20/40 営業日後の騰落率
- 勝率: 20営業日後に示唆の方向へ動いていた割合
- 目標値到達率 / 否定率: 60営業日以内に目標値と否定ライン(ダブルトップなら山の高値)の
  どちらに先に到達したか(同じ日に両方なら否定とみなす)
- 完成率: ブレイク前の段階で検出されたものが、40営業日以内(否定される前)にネックラインを抜けた割合

比較対象(ベースライン)として、同じ銘柄・期間の全営業日で同じ測り方をした結果も出す。
「勝率 − ベースライン勝率」が、パターンを使うことで上乗せされた分の目安になる。

使い方:
    # yfinance でプリセット銘柄を取得して検証し、画面用の統計を保存
    python -m backend.pattern_backtest --universe jp_major --period 10y

    # 縦持ちCSV(date, open, high, low, close, volume, Name)で検証
    python -m backend.pattern_backtest --csv all_stocks_5yr.csv --split-date 2016-01-01
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backend.patterns import (
    DEFAULT_PARAMS,
    LOOKBACK_BARS,
    MODEL_FEATURES,
    MODEL_PATH,
    PATTERN_NAMES,
    PRE_BREAKOUT_STAGES,
    STAGE_NAMES,
    Params,
    RawPattern,
    detect_raw,
    model_features,
    threshold_from_atr,
    true_range_pct,
)

HORIZONS = (5, 10, 20, 40)
WIN_HORIZON = 20
OUTCOME_WINDOW = 60
COMPLETION_WINDOW = 40
COOLDOWN = 10          # 同じ銘柄・パターン・段階のイベントを再カウントしない日数
WARMUP = LOOKBACK_BARS  # 最初の1年は判定に必要な過去データとして使う

DEFAULT_STATS_PATH = Path(__file__).resolve().parent.parent / "data" / "pattern_stats.json"


class Prepared:
    """1銘柄分の配列を前計算したもの。"""

    def __init__(self, df: pd.DataFrame):
        df = df.dropna(subset=["High", "Low", "Close"])
        self.dates = df.index
        self.open = df["Open"].to_numpy(dtype=float) if "Open" in df else df["Close"].to_numpy(dtype=float)
        self.high = df["High"].to_numpy(dtype=float)
        self.low = df["Low"].to_numpy(dtype=float)
        self.close = df["Close"].to_numpy(dtype=float)
        self.volume = df["Volume"].to_numpy(dtype=float) if "Volume" in df else np.full(len(df), np.nan)
        self.lh, self.ll, self.lc = np.log(self.high), np.log(self.low), np.log(self.close)
        self.trp = true_range_pct(self.high, self.low, self.close)
        self.n = len(df)

    def detect(self, t: int, params: Params = DEFAULT_PARAMS, sensitivity: float = 1.0):
        """t 日目の終値時点で detect_patterns() と同じ判定をする。インデックスは window 先頭基準。"""
        s = max(0, t - LOOKBACK_BARS + 1)
        if t - s + 1 < 40:
            return s, []
        # window が LOOKBACK_BARS 本あれば直近100本に window 先頭は含まれず、
        # s == 0 なら系列先頭も前日終値なしで同じ計算になるため、全体の配列から切り出してよい
        atr = float(np.nanmedian(self.trp[max(s, t - 99):t + 1]))
        thr = threshold_from_atr(atr, params, sensitivity)
        sl = slice(s, t + 1)
        return s, detect_raw(self.lh[sl], self.ll[sl], self.lc[sl], self.volume[sl], thr, params)


def detect_at(df: pd.DataFrame, t: int, params: Params = DEFAULT_PARAMS) -> list[RawPattern]:
    """テスト用: df の t 行目時点の検出結果(スコア順)。"""
    _, raws = Prepared(df).detect(t, params)
    return sorted(raws, key=lambda r: r.score, reverse=True)


def _evaluate(p: Prepared, t: int, s: int, r: RawPattern) -> dict:
    """t 日に検出されたパターン r のその後を測る。"""
    entry_i = t + 1
    entry = p.open[entry_i] if np.isfinite(p.open[entry_i]) else p.close[t]
    d = r.direction
    ev = {
        "pattern": r.kind,
        "stage": r.stage,
        "direction": d,
        "score": r.score,
        "features": {k: round(v, 3) for k, v in model_features(r).items()},
        "date": str(p.dates[t].date()),
    }
    for h in HORIZONS:
        j = t + h
        ev[f"ret_{h}"] = (d if d else 1) * (p.close[j] / entry - 1) if j < p.n else None

    target, stop = float(np.exp(r.target)), float(np.exp(r.stop))
    end = min(p.n, entry_i + OUTCOME_WINDOW)
    outcome = None
    if d != 0 and end - entry_i == OUTCOME_WINDOW:
        outcome = "none"
        for i in range(entry_i, end):
            hit_stop = p.high[i] >= stop if d < 0 else p.low[i] <= stop
            hit_target = p.low[i] <= target if d < 0 else p.high[i] >= target
            if hit_stop:
                outcome = "stop"
                break
            if hit_target:
                outcome = "target"
                break
    ev["outcome"] = outcome

    # ブレイク前の段階: ネックライン(ブレイクライン)を抜けたか
    if r.stage in PRE_BREAKOUT_STAGES:
        bl = r.breakout_line
        end = min(p.n, entry_i + COMPLETION_WINDOW)
        completed = None
        if end - entry_i == COMPLETION_WINDOW:
            completed = "none"
            if d == 0:  # 対称三角形: どちらに抜けたか
                up, lo = r.lines[0], r.lines[1]
                for i in range(entry_i, end):
                    if p.lc[i] > up.at(i - s):
                        completed = "up"
                        break
                    if p.lc[i] < lo.at(i - s):
                        completed = "down"
                        break
            else:
                for i in range(entry_i, end):
                    if (p.high[i] >= stop) if d < 0 else (p.low[i] <= stop):
                        break
                    if d * (p.lc[i] - bl.at(i - s)) > 0:
                        completed = "yes"
                        break
        ev["completed"] = completed
    return ev


def backtest_ticker(args) -> dict:
    """1銘柄のイベント一覧とベースラインを返す(ProcessPoolExecutor 用に引数はタプル)。"""
    ticker, df, params, sensitivity = args
    p = Prepared(df)
    events = []
    prev_keys: set = set()
    last_event: dict = {}
    for t in range(WARMUP, p.n - 1):
        s, raws = p.detect(t, params, sensitivity)
        keys = {}
        for r in raws:
            keys[(r.kind, r.stage)] = r
        for key, r in keys.items():
            if key in prev_keys or t - last_event.get(key, -10**9) <= COOLDOWN:
                continue
            last_event[key] = t
            ev = _evaluate(p, t, s, r)
            ev["ticker"] = ticker
            events.append(ev)
        prev_keys = set(keys)

    # ベースライン: 同じ期間の全営業日で翌日始値から20営業日後の騰落率
    base = []
    for t in range(WARMUP, p.n - 1 - WIN_HORIZON):
        e = p.open[t + 1] if np.isfinite(p.open[t + 1]) else p.close[t]
        base.append((str(p.dates[t].date()), p.close[t + WIN_HORIZON] / e - 1))
    return {"ticker": ticker, "events": events, "baseline": base}


def run_backtest(data: dict[str, pd.DataFrame], params: Params = DEFAULT_PARAMS,
                 sensitivity: float = 1.0, workers: int | None = None) -> dict:
    jobs = [(t, df, params, sensitivity) for t, df in data.items() if len(df) > WARMUP + 60]
    workers = workers or os.cpu_count() or 1
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(backtest_ticker, jobs, chunksize=4))
    else:
        results = [backtest_ticker(j) for j in jobs]
    events = [e for r in results for e in r["events"]]
    baseline = [b for r in results for b in r["baseline"]]
    return {"events": events, "baseline": baseline, "tickers": [r["ticker"] for r in results]}


# ---------------------------------------------------------------------------
# 集計
# ---------------------------------------------------------------------------

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)) * 100, 2) if xs else None


def _rate(xs, value):
    xs = [x for x in xs if x is not None]
    return round(sum(x == value for x in xs) / len(xs) * 100, 1) if xs else None


def _group_stats(evs: list[dict], mkt: dict, mid_date: str | None) -> dict:
    """mkt: 日付 → その日に全銘柄へ投資した場合の平均20日騰落率(市場平均)。"""
    d = evs[0]["direction"] if evs else 0
    rets = [e[f"ret_{WIN_HORIZON}"] for e in evs if e[f"ret_{WIN_HORIZON}"] is not None]
    st = {"n": len(evs)}
    for h in HORIZONS:
        st[f"avg_ret_{h}"] = _mean([e[f"ret_{h}"] for e in evs])
    if d != 0 and rets:
        st["win_rate"] = round(sum(r > 0 for r in rets) / len(rets) * 100, 1)
        st["median_ret_20"] = round(float(np.median(rets)) * 100, 2)
        # 市場平均との差(示唆の方向で見た超過リターン)
        ex = [(e["date"], e[f"ret_{WIN_HORIZON}"] - d * mkt[e["date"]]) for e in evs
              if e[f"ret_{WIN_HORIZON}"] is not None and e["date"] in mkt]
        if ex:
            st["excess_avg_20"] = _mean([x for _, x in ex])
            st["excess_win_rate"] = round(sum(x > 0 for _, x in ex) / len(ex) * 100, 1)
            if mid_date:
                st["excess_halves"] = [_mean([x for dt, x in ex if dt < mid_date]),
                                       _mean([x for dt, x in ex if dt >= mid_date])]
        st["target_rate"] = _rate([e["outcome"] for e in evs], "target")
        st["stop_rate"] = _rate([e["outcome"] for e in evs], "stop")
    comp = [e.get("completed") for e in evs if "completed" in e]
    if comp:
        if d == 0:
            st["breakout_up_rate"] = _rate(comp, "up")
            st["breakout_down_rate"] = _rate(comp, "down")
        else:
            st["completion_rate"] = _rate(comp, "yes")
    return st


def summarize(result: dict, start: str | None = None, end: str | None = None) -> dict:
    """イベントを パターン×段階 ごとに集計する。start/end で期間を絞れる。"""
    def in_range(date):
        return (start is None or date >= start) and (end is None or date < end)

    evs = [e for e in result["events"] if in_range(e["date"])]
    base = [(dt, r) for dt, r in result["baseline"] if in_range(dt)]
    by_date = defaultdict(list)
    for dt, r in base:
        by_date[dt].append(r)
    mkt = {dt: float(np.mean(rs)) for dt, rs in by_date.items()}
    dates = sorted(e["date"] for e in evs)
    mid_date = dates[len(dates) // 2] if dates else None

    groups = defaultdict(list)
    for e in evs:
        groups[(e["pattern"], e["stage"], e["direction"])].append(e)
    by_pattern: dict = defaultdict(dict)
    for (pat, stage, d), g in sorted(groups.items()):
        key = stage if pat != "symmetric_triangle" or stage != "breakout" else f"breakout_{'up' if d > 0 else 'down'}"
        by_pattern[pat][key] = _group_stats(g, mkt, mid_date)

    rs = [r for _, r in base]
    return {
        "period": {"start": dates[0] if dates else None, "end": dates[-1] if dates else None,
                   "mid": mid_date},
        "events": len(evs),
        "baseline": {
            "days": len(rs),
            "up_rate_20": round(sum(r > 0 for r in rs) / len(rs) * 100, 1) if rs else None,
            "avg_ret_20": _mean(rs),
        },
        "by_pattern": dict(by_pattern),
    }


# ---------------------------------------------------------------------------
# 確率モデル(ロジスティック回帰)
# ---------------------------------------------------------------------------

def _task_rows(events: list[dict], task: str):
    """task="complete": ブレイク前に検出 → ネックラインを抜けたか。
    task="target": 方向のあるパターン → 否定ラインより先に目標値へ到達したか。"""
    rows = []
    for e in events:
        if task == "complete":
            if e.get("completed") is None:
                continue
            y = e["completed"] in ("yes", "up", "down")
        else:
            if e["direction"] == 0 or e["outcome"] is None:
                continue
            y = e["outcome"] == "target"
        rows.append((f"{e['pattern']}:{e['stage']}", e["features"], float(y)))
    return rows


def _design(rows, groups):
    gi = {g: k for k, g in enumerate(groups)}
    X = np.zeros((len(rows), 1 + len(groups) + len(MODEL_FEATURES)))
    X[:, 0] = 1.0
    for n, (g, f, _) in enumerate(rows):
        X[n, 1 + gi[g]] = 1.0
        X[n, 1 + len(groups):] = [f.get(k, 0.5) for k in MODEL_FEATURES]
    return X, np.array([y for _, _, y in rows])


def _logistic(X, y, l2=1.0, iters=30):
    """L2正則化つきロジスティック回帰(ニュートン法)。切片には正則化をかけない。"""
    w = np.zeros(X.shape[1])
    reg = np.full(X.shape[1], l2)
    reg[0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        grad = X.T @ (p - y) + reg * w
        hess = (X.T * (p * (1 - p))) @ X + np.diag(reg)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def _auc(score, y):
    order = pd.Series(score).rank().to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    return round(float((order[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), 3)


def fit_model(events: list[dict], min_group: int = 30) -> dict:
    models, quality = {}, {}
    for task in ("complete", "target"):
        rows = _task_rows(events, task)
        counts = defaultdict(int)
        for g, _, _ in rows:
            counts[g] += 1
        groups = sorted(g for g, c in counts.items() if c >= min_group)
        rows = [r for r in rows if r[0] in groups]
        if not rows:
            continue
        X, y = _design(rows, groups)
        w = _logistic(X, y)
        models[task] = {
            "bias": round(float(w[0]), 4),
            "group": {g: round(float(w[1 + k]), 4) for k, g in enumerate(groups)},
            "coef": {f: round(float(w[1 + len(groups) + k]), 4) for k, f in enumerate(MODEL_FEATURES)},
        }
        quality[task] = {"n": len(rows), "base_rate": round(float(y.mean()), 3),
                         "auc_in_sample": _auc(X @ w, y)}
    return {"features": list(MODEL_FEATURES), "models": models, "quality": quality}


def validate_model(events: list[dict], split_date: str) -> dict:
    """split_date より前で学習し、以降のデータで予測精度(AUC)と校正を確かめる。"""
    train = [e for e in events if e["date"] < split_date]
    test = [e for e in events if e["date"] >= split_date]
    model = fit_model(train)
    out = {}
    for task, m in model["models"].items():
        rows = [r for r in _task_rows(test, task) if r[0] in m["group"]]
        if not rows:
            continue
        groups = list(m["group"])
        X, y = _design(rows, groups)
        w = np.array([m["bias"], *m["group"].values(), *m["coef"].values()])
        p = 1 / (1 + np.exp(-X @ w))
        bins = []
        for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
            mask = (p >= lo) & (p < hi)
            if mask.sum() >= 30:
                bins.append({"predicted": round(float(p[mask].mean()) * 100, 1),
                             "actual": round(float(y[mask].mean()) * 100, 1), "n": int(mask.sum())})
        out[task] = {"n": len(rows), "auc": _auc(p, y), "calibration": bins}
    return out


def print_summary(s: dict, title: str = "") -> None:
    print(f"\n=== {title} イベント {s['events']}件 / 期間 {s['period']['start']}〜{s['period']['end']} ===")
    b = s["baseline"]
    print(f"ベースライン: 20日後に上昇していた割合 {b['up_rate_20']}% / 平均 {b['avg_ret_20']}%")
    print(f"{'パターン':<18}{'段階':<16}{'件数':>6}{'勝率':>7}{'対市場':>7}{'平均20d':>9}"
          f"{'目標到達':>9}{'否定':>7}{'完成':>7}")
    for pat, stages in s["by_pattern"].items():
        for stage, st in stages.items():
            comp = st.get("completion_rate")
            if comp is None and "breakout_up_rate" in st:
                comp = f"↑{st['breakout_up_rate']}/↓{st['breakout_down_rate']}"
            print(f"{PATTERN_NAMES[pat][:9]:<18}{STAGE_NAMES.get(stage, stage)[:8]:<16}{st['n']:>6}"
                  f"{_f(st.get('win_rate')):>7}{_f(st.get('excess_avg_20')):>7}{_f(st.get('avg_ret_20')):>9}"
                  f"{_f(st.get('target_rate')):>9}{_f(st.get('stop_rate')):>7}{_f(comp):>7}")


def _f(v):
    return "-" if v is None else str(v)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="チャートパターンの的中率を過去データで検証する")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--universe", default="jp_major", help="プリセット銘柄リストID (yfinance で取得)")
    src.add_argument("--tickers", help="カンマ区切りの銘柄コード (yfinance で取得)")
    src.add_argument("--csv", help="縦持ちCSV (date, open, high, low, close, volume, Name)")
    ap.add_argument("--period", default="10y", help="yfinance の取得期間 (例: 5y, 10y, max)")
    ap.add_argument("--sensitivity", type=float, default=1.0)
    ap.add_argument("--split-date", help="学習/検証を分ける日付 (YYYY-MM-DD)。省略時はイベントの中央の日付")
    ap.add_argument("--out", default=str(DEFAULT_STATS_PATH), help="統計JSONの保存先 ('' で保存しない)")
    ap.add_argument("--model-out", default=str(MODEL_PATH), help="確率モデルの保存先 ('' で保存しない)")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)

    if a.csv:
        from backend.market_data import load_csv_long
        data = load_csv_long(a.csv)
        source = f"CSV: {os.path.basename(a.csv)}"
    else:
        from backend.market_data import download_ohlcv
        from backend.universes import UNIVERSES
        if a.tickers:
            from backend.pattern_api import normalize_ticker
            tickers = [normalize_ticker(t) for t in a.tickers.split(",") if t.strip()]
            source = "yfinance: " + ",".join(tickers)
        else:
            tickers = list(UNIVERSES[a.universe]["tickers"])
            source = f"yfinance: {UNIVERSES[a.universe]['name']}"
        data = download_ohlcv(tickers, a.period)
    data = drop_suspicious(data)
    print(f"{len(data)}銘柄を検証します…")

    result = run_backtest(data, sensitivity=a.sensitivity, workers=a.workers)
    summary = summarize(result)
    print_summary(summary, "全期間")
    split = a.split_date or summary["period"]["mid"]
    print_summary(summarize(result, end=split), f"{split} より前")
    print_summary(summarize(result, start=split), f"{split} 以降")

    # 確率モデル: 前半で学習 → 後半で検証してから、全期間で学習し直して保存する
    validation = validate_model(result["events"], split)
    print(f"\n確率モデルの検証({split} より前で学習 → 以降で予測):")
    for task, v in validation.items():
        label = {"complete": "完成確率", "target": "目標到達確率"}[task]
        cal = ", ".join(f"予測{b['predicted']}%→実績{b['actual']}%" for b in v["calibration"])
        print(f"  {label}: AUC {v['auc']} (n={v['n']}) / {cal}")
    model = fit_model(result["events"])
    model.update({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "validation": {"split_date": split, **validation},
    })
    if a.model_out:
        Path(a.model_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.model_out).write_text(json.dumps(model, ensure_ascii=False, indent=1))
        print(f"確率モデルを保存しました: {a.model_out}")

    if a.out:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": source,
            "tickers": len(result["tickers"]),
            "sensitivity": a.sensitivity,
            "params": {k: v for k, v in asdict(DEFAULT_PARAMS).items()},
            "method": {
                "entry": "検出翌営業日の始値",
                "win_horizon_days": WIN_HORIZON,
                "outcome_window_days": OUTCOME_WINDOW,
                "completion_window_days": COMPLETION_WINDOW,
            },
            **summary,
            "model_validation": {"split_date": split, **validation},
        }
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1))
        print(f"\n統計を保存しました: {a.out}")


def drop_suspicious(data: dict[str, pd.DataFrame], max_gap: float = 0.4) -> dict[str, pd.DataFrame]:
    """未調整の株式分割などで1日に40%超の窓が空いている銘柄は除外する。"""
    out = {}
    for t, df in data.items():
        if "Open" in df:
            gap = np.log(df["Open"] / df["Close"].shift()).abs()
            if (gap > np.log(1 + max_gap)).any():
                continue
        out[t] = df
    return out


if __name__ == "__main__":
    main()
