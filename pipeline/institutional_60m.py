from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from .fetch_sidecar_data import TushareClient, _ts_code_from_symbol
    from .schemas import Candidate, CandidateRun
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fetch_sidecar_data import TushareClient, _ts_code_from_symbol
    from schemas import Candidate, CandidateRun


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "rules_preselect.yaml"
DEFAULT_CANDIDATES = PROJECT_ROOT / "data" / "candidates" / "candidates_latest.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "candidates"


def _resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_candidate_run(path: Path) -> CandidateRun:
    data = json.loads(path.read_text(encoding="utf-8"))
    return CandidateRun.from_dict(data)


def _strategy_matches(candidate: Candidate, source_strategy: str) -> bool:
    if not source_strategy:
        return True
    if source_strategy in str(candidate.strategy).split("+"):
        return True
    strategies = candidate.extra.get("strategies") if candidate.extra else None
    return isinstance(strategies, list) and source_strategy in strategies


def _datetime_bounds(pick_date: str, lookback_days: int) -> tuple[str, str]:
    end_day = datetime.strptime(pick_date, "%Y-%m-%d")
    start_day = end_day - timedelta(days=int(lookback_days))
    return (
        start_day.strftime("%Y-%m-%d 09:00:00"),
        end_day.strftime("%Y-%m-%d 15:30:00"),
    )


def _normalize_60m_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    dt_col = next(
        (c for c in ["trade_time", "trade_date", "datetime", "date", "time"] if c in out.columns),
        None,
    )
    if dt_col is None:
        raise ValueError("60m data has no datetime column")
    out["datetime"] = pd.to_datetime(out[dt_col], errors="coerce")
    out = out.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates("datetime", keep="last")
    for column in ["open", "high", "low", "close", "vol", "volume", "amount"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    if "high" not in out.columns:
        out["high"] = out["close"]
    if "low" not in out.columns:
        out["low"] = out["close"]
    return out.reset_index(drop=True)


def _read_60m_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return _normalize_60m_frame(pd.read_csv(path, dtype={"ts_code": str}))


def _cache_is_current(df: pd.DataFrame, pick_date: str) -> bool:
    if df.empty or "datetime" not in df.columns:
        return False
    latest_day = pd.to_datetime(df["datetime"].max()).strftime("%Y-%m-%d")
    return latest_day >= pick_date


def _write_60m_cache(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _fetch_one_60m(
    client: TushareClient,
    code: str,
    path: Path,
    *,
    pick_date: str,
    lookback_days: int,
    incremental: bool,
) -> tuple[pd.DataFrame, str | None]:
    cached = _read_60m_cache(path)
    if incremental and _cache_is_current(cached, pick_date):
        return cached, None

    start_date, end_date = _datetime_bounds(pick_date, lookback_days)
    try:
        fresh = client.call(
            "stk_mins",
            ts_code=_ts_code_from_symbol(code),
            freq="60min",
            start_date=start_date,
            end_date=end_date,
        )
        fresh = _normalize_60m_frame(fresh)
        merged = pd.concat([cached, fresh], ignore_index=True) if not cached.empty else fresh
        merged = _normalize_60m_frame(merged)
        _write_60m_cache(path, merged)
        return merged, None
    except Exception as exc:
        if not cached.empty:
            return cached, str(exc)
        return pd.DataFrame(), str(exc)


def ensure_60m_data(
    candidates: list[Candidate],
    *,
    pick_date: str,
    cfg_60m: dict[str, Any],
    fetch_missing: bool,
    fetch_sleep: float,
    limit_codes: int,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, str]]]:
    kline_dir = _resolve_project_path(cfg_60m.get("kline_dir", "./data/kline_60m"))
    lookback_days = int(cfg_60m.get("lookback_days", 120))
    codes = list(dict.fromkeys(c.code for c in candidates))
    if limit_codes > 0:
        codes = codes[:limit_codes]
    if not codes:
        return {}, []

    client = TushareClient() if fetch_missing else None
    frames: dict[str, pd.DataFrame] = {}
    failures: list[dict[str, str]] = []
    rate_limited = False

    for idx, code in enumerate(codes, 1):
        path = kline_dir / f"{code}.csv"
        if fetch_missing and not rate_limited and client is not None:
            df, error = _fetch_one_60m(
                client,
                code,
                path,
                pick_date=pick_date,
                lookback_days=lookback_days,
                incremental=True,
            )
            if error:
                failures.append({"code": code, "error": error})
                if "频率" in error or "frequency" in error.lower() or "freq" in error.lower():
                    rate_limited = True
            if fetch_sleep > 0 and idx < len(codes):
                time.sleep(fetch_sleep)
        else:
            df = _read_60m_cache(path)
            if df.empty:
                failures.append({"code": code, "error": "no cached 60m data"})
        frames[code] = df
        print(f"[60m {idx}/{len(codes)}] {code} bars={len(df)}")

    return frames, failures


def _round_float(value: object, ndigits: int = 4) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(val):
        return None
    return round(val, ndigits)


def evaluate_strategy2(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    df = _normalize_60m_frame(df)
    min_bars = int(cfg.get("min_bars", 80))
    if len(df) < min_bars:
        return False, {"reason": "not_enough_60m_bars", "bars": int(len(df))}

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    ma_window = int(cfg.get("ma_window", 60))
    ma = close.rolling(ma_window, min_periods=ma_window).mean()
    slope_window = int(cfg.get("ma_slope_window", 5))
    ma_slope = ma / ma.shift(slope_window) - 1.0
    distance = close / ma - 1.0

    fast = int(cfg.get("macd_fast", 12))
    slow = int(cfg.get("macd_slow", 26))
    signal = int(cfg.get("macd_signal", 9))
    dif = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = dif - dea

    boll_window = int(cfg.get("boll_window", 20))
    boll_mid = close.rolling(boll_window, min_periods=boll_window).mean()
    boll_std = close.rolling(boll_window, min_periods=boll_window).std(ddof=0)
    boll_width = (2.0 * float(cfg.get("boll_std", 2.0)) * boll_std) / boll_mid.replace(0, np.nan)
    boll_threshold = float(cfg.get("max_boll_width", 0.12))
    if bool(cfg.get("use_dynamic_boll_width", True)):
        lookback = int(cfg.get("boll_width_quantile_lookback", 120))
        quantile = float(cfg.get("boll_width_quantile", 0.80))
        dynamic = boll_width.dropna().tail(lookback).quantile(quantile)
        if np.isfinite(dynamic):
            boll_threshold = min(boll_threshold, float(dynamic))

    beauty_lookback = int(cfg.get("beauty_lookback", 5))
    recent_low_distance = (low.tail(beauty_lookback).to_numpy() / ma.tail(beauty_lookback).to_numpy() - 1.0)
    recent_low_distance = float(np.nanmin(recent_low_distance))
    recent_high = float(high.tail(beauty_lookback).max())
    prev_high = float(high.iloc[-beauty_lookback * 2:-beauty_lookback].max())
    high_lift = recent_high / prev_high - 1.0 if prev_high > 0 else np.nan

    latest = {
        "bars": int(len(df)),
        "last_datetime": str(df["datetime"].iloc[-1]),
        "close": _round_float(close.iloc[-1], 4),
        "ma60": _round_float(ma.iloc[-1], 4),
        "ma60_slope": _round_float(ma_slope.iloc[-1], 6),
        "distance_to_ma60": _round_float(distance.iloc[-1], 6),
        "dif": _round_float(dif.iloc[-1], 6),
        "dea": _round_float(dea.iloc[-1], 6),
        "macd_hist": _round_float(hist.iloc[-1], 6),
        "boll_width": _round_float(boll_width.iloc[-1], 6),
        "boll_width_threshold": _round_float(boll_threshold, 6),
        "recent_low_distance_to_ma60": _round_float(recent_low_distance, 6),
        "recent_high_lift": _round_float(high_lift, 6),
    }

    checks = {
        "close_above_ma60": (not bool(cfg.get("require_close_above_ma60", True)))
        or close.iloc[-1] > ma.iloc[-1],
        "ma60_slope": ma_slope.iloc[-1] >= float(cfg.get("min_ma60_slope", 0.0)),
        "distance_not_too_far": distance.iloc[-1] <= float(cfg.get("max_distance_above_ma60", 0.12)),
        "macd_positive": dif.iloc[-1] > dea.iloc[-1],
        "boll_width_ok": boll_width.iloc[-1] <= boll_threshold,
        "recent_low_holds_ma60": recent_low_distance >= float(cfg.get("recent_low_ma60_tolerance", -0.02)),
        "recent_high_lift": high_lift >= float(cfg.get("recent_high_lift_min", 0.0)),
    }
    if bool(cfg.get("require_macd_hist_positive", True)):
        checks["macd_hist_positive"] = hist.iloc[-1] > 0

    checks = {key: bool(value) for key, value in checks.items()}
    passed = all(bool(v) for v in checks.values())
    latest["checks"] = checks
    if not passed:
        latest["failed_checks"] = [k for k, v in checks.items() if not bool(v)]
    return passed, latest


def filter_candidates_60m(
    candidates: list[Candidate],
    frames: dict[str, pd.DataFrame],
    cfg_60m: dict[str, Any],
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    passed: list[Candidate] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        df = frames.get(candidate.code, pd.DataFrame())
        ok, metrics = evaluate_strategy2(df, cfg_60m)
        if ok:
            extra = dict(candidate.extra or {})
            extra["institutional_60m"] = metrics
            strategy_parts = str(candidate.strategy).split("+")
            if "institutional_2" not in strategy_parts:
                strategy_parts.append("institutional_2")
            passed.append(replace(candidate, strategy="+".join(strategy_parts), extra=extra))
        else:
            rejected.append({"code": candidate.code, **metrics})
    return passed, rejected


def save_outputs(
    run: CandidateRun,
    *,
    output_dir: Path,
    replace_latest: bool,
    summary: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dated = output_dir / f"institutional_60m_{run.pick_date}.json"
    latest = output_dir / "institutional_60m_latest.json"
    _atomic_write_json(dated, run.to_dict())
    _atomic_write_json(latest, run.to_dict())
    paths = {"dated": str(dated), "latest": str(latest)}
    if replace_latest:
        canonical = output_dir / "candidates_latest.json"
        canonical_dated = output_dir / f"candidates_{run.pick_date}.json"
        _atomic_write_json(canonical, run.to_dict())
        _atomic_write_json(canonical_dated, run.to_dict())
        paths["canonical_latest"] = str(canonical)
        paths["canonical_dated"] = str(canonical_dated)

    summary_path = output_dir / f"institutional_60m_summary_{run.pick_date}.json"
    _atomic_write_json(summary_path, summary)
    paths["summary"] = str(summary_path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch cached 60-minute bars and apply institutional strategy 2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--source-strategy", default="institutional_1")
    parser.add_argument("--no-fetch", action="store_true", help="use cached 60m files only")
    parser.add_argument("--fetch-sleep", type=float, default=0.0)
    parser.add_argument("--limit-codes", type=int, default=0, help="debug helper")
    parser.add_argument("--replace-latest", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(_resolve_project_path(args.config))
    cfg_60m = cfg.get("institutional_60m", {}) or {}
    if not bool(cfg_60m.get("enabled", True)):
        print("[INFO] institutional_60m disabled")
        return

    source_run = _load_candidate_run(_resolve_project_path(args.candidates))
    source_candidates = [
        c for c in source_run.candidates if _strategy_matches(c, args.source_strategy)
    ]
    if args.limit_codes > 0:
        source_candidates = source_candidates[:args.limit_codes]

    frames, fetch_failures = ensure_60m_data(
        source_candidates,
        pick_date=source_run.pick_date,
        cfg_60m=cfg_60m,
        fetch_missing=not args.no_fetch,
        fetch_sleep=args.fetch_sleep,
        limit_codes=args.limit_codes,
    )
    passed, rejected = filter_candidates_60m(source_candidates, frames, cfg_60m)
    out_run = CandidateRun(
        run_date=datetime.now().date().isoformat(),
        pick_date=source_run.pick_date,
        candidates=passed,
        meta={
            **(source_run.meta or {}),
            "stage": "institutional_60m",
            "source_candidates": str(_resolve_project_path(args.candidates)),
            "source_strategy": args.source_strategy,
            "source_count": len(source_candidates),
            "passed_count": len(passed),
        },
    )
    summary = {
        "pick_date": source_run.pick_date,
        "source_count": len(source_candidates),
        "passed_count": len(passed),
        "rejected_count": len(rejected),
        "fetch_failures": fetch_failures,
        "rejected": rejected,
    }
    paths = save_outputs(
        out_run,
        output_dir=_resolve_project_path(args.output),
        replace_latest=bool(args.replace_latest),
        summary=summary,
    )
    print(f"[INFO] institutional 60m passed: {len(passed)}/{len(source_candidates)}")
    for key, path in paths.items():
        print(f"[INFO] {key}: {path}")


if __name__ == "__main__":
    main()
