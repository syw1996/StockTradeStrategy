"""
pipeline/select_stock.py
量化初选核心逻辑。

职责：
  - 读取 rules_preselect.yaml 参数
  - 加载 data/raw/*.csv 日线数据
  - 运行 B1 策略（KDJ + 知行均线）和砖型图策略
  - 返回 List[Candidate]（纯 Python 对象，不写文件）
  - 写文件由 cli.py 调用 io.py 完成
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml 

from schemas import Candidate
from Selector import B1Selector, BrickChartSelector, GoldenNeedleSelector, KGMomentumSelector
from pipeline_core import MarketDataPreparer, TopTurnoverPoolBuilder

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "rules_preselect.yaml"
_STOCK_CSV_RE = re.compile(r"^\d{6}\.csv$", re.IGNORECASE)


def _resolve_cfg_path(path_like: str | Path, base_dir: Path = _PROJECT_ROOT) -> Path:
    """将配置中的相对路径解析为项目根目录下的绝对路径。"""
    p = Path(path_like)
    return p if p.is_absolute() else (base_dir / p)


# =============================================================================
# 配置 & 数据加载
# =============================================================================

def load_config(config_path: Optional[str] = None) -> dict:
    """加载 rules_preselect.yaml，返回原始 dict."""
    path = _resolve_cfg_path(config_path) if config_path else _DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def resolve_preselect_output_dir(
    *,
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Path:
    """返回候选输出目录，优先级：CLI参数 > 配置文件 global.output_dir > 默认值。"""
    if output_dir:
        return _resolve_cfg_path(output_dir)
    cfg = load_config(config_path)
    g = cfg.get("global", {})
    return _resolve_cfg_path(g.get("output_dir", "./data/candidates"))


def load_raw_data(
    data_dir: str,
    end_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """读取 data_dir 下六位股票代码 CSV，统一处理列名/日期/排序."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data_dir 不存在: {data_dir}")

    end_ts = pd.to_datetime(end_date) if end_date else None
    data: Dict[str, pd.DataFrame] = {}
    skipped_non_stock = 0

    for fname in sorted(os.listdir(data_dir)):
        if not _STOCK_CSV_RE.fullmatch(fname):
            if fname.lower().endswith(".csv"):
                skipped_non_stock += 1
            continue
        code = fname.rsplit(".", 1)[0]
        fpath = os.path.join(data_dir, fname)

        df = pd.read_csv(fpath)
        df.columns = [c.lower() for c in df.columns]
        if "date" not in df.columns:
            logger.warning("跳过 %s：没有 date 列", fname)
            continue

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        if end_ts is not None:
            df = df[df["date"] <= end_ts].reset_index(drop=True)

        if not df.empty:
            data[code] = df

    if not data:
        raise ValueError(f"未找到任何 CSV 数据: {data_dir}")

    logger.info("读取股票数量: %d", len(data))
    if skipped_non_stock:
        logger.info("跳过非个股 CSV: %d 个", skipped_non_stock)
    return data


# =============================================================================
# 工具函数
# =============================================================================

def _sorted_zx(m1: int, m2: int, m3: int, m4: int) -> Tuple[int, int, int, int]:
    """保证均线参数从小到大排列."""
    a = sorted([int(m1), int(m2), int(m3), int(m4)])
    return a[0], a[1], a[2], a[3]


def _resolve_pick_date(
    prepared: Dict[str, pd.DataFrame],
    pick_date: Optional[str] = None,
) -> pd.Timestamp:
    """确定选股基准日期：None → 最晚可用交易日，否则向前搜索最近日期."""
    all_dates = sorted(
        {d for df in prepared.values() if isinstance(df.index, pd.DatetimeIndex) for d in df.index}
    )
    if not all_dates:
        raise ValueError("prepared 数据中没有可用日期。")
    if pick_date is None:
        return all_dates[-1]

    target = pd.to_datetime(pick_date)
    arr = np.array(all_dates, dtype="datetime64[ns]")
    idx = int(np.searchsorted(arr, target.to_datetime64(), side="right")) - 1
    if idx < 0:
        raise ValueError(f"pick_date={pick_date} 早于最早可用日期={all_dates[0].date()}")
    return all_dates[idx]


def _calc_warmup(cfg: dict, buffer: int) -> int:
    """根据启用策略的参数计算最长所需 warmup bars."""
    warmup = 120

    cfg_b1 = cfg.get("b1", {})
    if cfg_b1.get("enabled", True):
        warmup = max(warmup, int(cfg_b1.get("zx_m4", 371)) + buffer)

    cfg_brick = cfg.get("brick", {})
    if cfg_brick.get("enabled", True):
        warmup = max(
            warmup,
            int(cfg_brick.get("wma_long", 120)) * 5 + buffer,
            int(cfg_brick.get("zxdkx_m4", 114)) + buffer,
        )

    cfg_golden_needle = cfg.get("golden_needle", {})
    if cfg_golden_needle.get("enabled", False):
        warmup = max(
            warmup,
            int(cfg_golden_needle.get("n2", 21))
            + int(cfg_golden_needle.get("cond6_long_window", 9))
            - 1
            + buffer,
        )

    cfg_kg_momentum = cfg.get("kg_momentum", {})
    if cfg_kg_momentum.get("enabled", False):
        warmup = max(warmup, 65 + buffer)

    return warmup


def _optional_float(cfg: dict, key: str, default: Optional[float]) -> Optional[float]:
    value = cfg.get(key, default)
    if value is None:
        return None
    return float(value)


def _format_filter_source(source: str, pick_date: pd.Timestamp) -> str:
    """渲染过滤数据源路径模板."""
    return source.format(
        date=pick_date.strftime("%Y-%m-%d"),
        yyyymmdd=pick_date.strftime("%Y%m%d"),
    )


def _resolve_sidecar_path(path_like: str | Path, *, data_dir: Path) -> Path:
    """过滤 sidecar 文件相对 data_dir 解析，绝对路径保持不变."""
    p = Path(path_like)
    return p if p.is_absolute() else (data_dir / p)


def _is_st_name(name: object) -> bool:
    """判断股票名称是否为 ST/*ST/SST/S*ST 等风险警示前缀."""
    text = "" if pd.isna(name) else str(name).strip().upper()
    return (
        text.startswith("*ST")
        or text.startswith("ST")
        or text.startswith("S*ST")
        or text.startswith("SST")
    )


def _load_st_codes(validation_path: Path) -> set[str]:
    """从 daily_update_validation 文件加载 ST 股票代码集合."""
    if not validation_path.exists():
        raise FileNotFoundError(f"ST 过滤文件不存在: {validation_path}")
    df = pd.read_csv(validation_path, dtype={"symbol": str, "name": str})
    if "symbol" not in df.columns or "name" not in df.columns:
        raise ValueError(f"ST 过滤文件缺少 symbol/name 列: {validation_path}")
    symbols = df.loc[df["name"].map(_is_st_name), "symbol"].dropna().astype(str)
    return {s.zfill(6) for s in symbols}


def _load_stocklist_st_codes(stocklist_path: Path) -> set[str]:
    """Load ST codes from the project's stocklist as a fallback risk source."""
    if not stocklist_path.exists():
        logger.warning("Stocklist ST filter skipped; file not found: %s", stocklist_path)
        return set()

    df = pd.read_csv(stocklist_path, dtype={"symbol": str, "name": str})
    if "symbol" not in df.columns or "name" not in df.columns:
        logger.warning("Stocklist ST filter skipped; missing symbol/name: %s", stocklist_path)
        return set()

    symbols = df.loc[df["name"].map(_is_st_name), "symbol"].dropna().astype(str)
    return {s.zfill(6) for s in symbols}


def _load_min_mv_allowed_codes(market_cap_path: Path, min_total_mv: float) -> set[str]:
    """从市值文件加载总市值达到阈值的股票代码集合."""
    if not market_cap_path.exists():
        raise FileNotFoundError(f"市值过滤文件不存在: {market_cap_path}")
    df = pd.read_csv(market_cap_path, dtype={"symbol": str})
    if "symbol" not in df.columns:
        if "ts_code" not in df.columns:
            raise ValueError(f"市值过滤文件缺少 symbol 或 ts_code 列: {market_cap_path}")
        df["symbol"] = df["ts_code"].astype(str).str.split(".", n=1).str[0]
    if "total_mv" not in df.columns:
        raise ValueError(f"市值过滤文件缺少 total_mv 列: {market_cap_path}")
    total_mv = pd.to_numeric(df["total_mv"], errors="coerce")
    symbols = df.loc[total_mv >= min_total_mv, "symbol"].dropna().astype(str)
    return {s.zfill(6) for s in symbols}


def _apply_universe_filters(
    pool_codes: List[str],
    *,
    pick_date: pd.Timestamp,
    data_dir: Path,
    cfg: dict,
) -> List[str]:
    """按 rules_preselect.yaml 中的 universe_filters 过滤股票池."""
    filter_cfg = cfg.get("universe_filters", {}) or {}
    if not filter_cfg:
        return pool_codes

    excluded: Dict[str, set[str]] = {}

    if bool(filter_cfg.get("exclude_st", False)):
        template = str(filter_cfg.get("validation_file_template", "daily_update_validation_{yyyymmdd}.csv"))
        validation_path = _resolve_sidecar_path(
            _format_filter_source(template, pick_date),
            data_dir=data_dir,
        )
        stocklist_path = Path(__file__).resolve().with_name("stocklist.csv")
        excluded["st"] = _load_st_codes(validation_path) | _load_stocklist_st_codes(stocklist_path)

    min_total_mv = filter_cfg.get("min_total_mv")
    if min_total_mv is not None:
        market_cap_file = str(filter_cfg.get("market_cap_file", "largecap.csv"))
        market_cap_path = _resolve_sidecar_path(
            _format_filter_source(market_cap_file, pick_date),
            data_dir=data_dir,
        )
        min_mv_allowed = _load_min_mv_allowed_codes(market_cap_path, float(min_total_mv))
        excluded["min_total_mv"] = set(pool_codes) - min_mv_allowed

    if not excluded:
        return pool_codes

    excluded_all: set[str] = set().union(*excluded.values())
    filtered = [code for code in pool_codes if code not in excluded_all]
    logger.info(
        "股票池过滤: 原始=%d, 过滤后=%d, ST=%d, 市值不足=%d",
        len(pool_codes),
        len(filtered),
        len(excluded.get("st", set()) & set(pool_codes)),
        len(excluded.get("min_total_mv", set()) & set(pool_codes)),
    )
    return filtered


def _merge_strategy_candidates(candidates: List[Candidate]) -> List[Candidate]:
    """同一股票多策略命中时合并为一条候选，并记录完整策略列表."""
    merged: Dict[str, Candidate] = {}
    strategies_by_code: Dict[str, List[str]] = {}

    for candidate in candidates:
        existing = merged.get(candidate.code)
        if existing is None:
            merged[candidate.code] = candidate
            strategies_by_code[candidate.code] = [candidate.strategy]
            continue

        strategies = strategies_by_code[candidate.code]
        if candidate.strategy not in strategies:
            strategies.append(candidate.strategy)

        if existing.brick_growth is None and candidate.brick_growth is not None:
            existing.brick_growth = candidate.brick_growth

        if candidate.extra:
            existing.extra[f"{candidate.strategy}_extra"] = candidate.extra

    deduped: List[Candidate] = []
    for code, candidate in merged.items():
        strategies = strategies_by_code[code]
        if len(strategies) > 1:
            candidate.strategy = "+".join(strategies)
            candidate.extra = {**candidate.extra, "strategies": strategies}
        deduped.append(candidate)
    return deduped


# =============================================================================
# B1 策略
# =============================================================================

def run_b1(
    prepared: Dict[str, pd.DataFrame],
    pick_date: pd.Timestamp,
    pool_codes: List[str],
    cfg_b1: dict,
) -> List[Candidate]:
    """在流动性池内运行 B1 策略，返回 Candidate 列表.

    优化：对每只股票先调用 prepare_df() 预计算所有指标列，
    再用 vec_picks_from_prepared() 直接查表，避免重复计算。
    """
    zx_m1, zx_m2, zx_m3, zx_m4 = _sorted_zx(
        cfg_b1["zx_m1"], cfg_b1["zx_m2"], cfg_b1["zx_m3"], cfg_b1["zx_m4"]
    )
    selector = B1Selector(
        j_threshold=float(cfg_b1["j_threshold"]),
        j_q_threshold=float(cfg_b1["j_q_threshold"]),
        zx_m1=zx_m1, zx_m2=zx_m2, zx_m3=zx_m3, zx_m4=zx_m4,
    )

    date_str = pick_date.strftime("%Y-%m-%d")
    candidates: List[Candidate] = []

    for code in pool_codes:
        df = prepared.get(code)
        if df is None or pick_date not in df.index:
            continue
        try:
            pf = selector.prepare_df(df)
            if selector.vec_picks_from_prepared(pf, start=pick_date, end=pick_date):
                row = pf.loc[pick_date]
                candidates.append(Candidate(
                    code=code,
                    date=date_str,
                    strategy="b1",
                    close=float(row["close"]),
                    turnover_n=float(row["turnover_n"]),
                ))
        except Exception as exc:
            logger.debug("B1 skip %s: %s", code, exc)

    logger.info("B1 选出: %d 只", len(candidates))
    return candidates


# =============================================================================
# 砖型图策略
# =============================================================================

def run_brick(
    prepared: Dict[str, pd.DataFrame],
    pick_date: pd.Timestamp,
    pool_codes: List[str],
    cfg_brick: dict,
) -> List[Candidate]:
    """在流动性池内运行砖型图策略，返回按 brick_growth 降序的 Candidate 列表.

    优化：对每只股票先调用 prepare_df() 预计算 brick/zxdq/wma_bull 等列，
    再用 vec_picks_from_prepared() 直接查表，brick_growth 也直接读预计算列，
    避免重复计算。
    """
    selector = BrickChartSelector(
        daily_return_threshold=float(cfg_brick.get("daily_return_threshold", 0.05)),
        brick_growth_ratio=float(cfg_brick.get("brick_growth_ratio", 1.0)),
        min_prior_green_bars=int(cfg_brick.get("min_prior_green_bars", 2)),
        zxdq_ratio=cfg_brick.get("zxdq_ratio"),
        zxdq_span=int(cfg_brick.get("zxdq_span", 10)),
        require_zxdq_gt_zxdkx=bool(cfg_brick.get("require_zxdq_gt_zxdkx", True)),
        zxdkx_m1=int(cfg_brick.get("zxdkx_m1", 14)),
        zxdkx_m2=int(cfg_brick.get("zxdkx_m2", 28)),
        zxdkx_m3=int(cfg_brick.get("zxdkx_m3", 57)),
        zxdkx_m4=int(cfg_brick.get("zxdkx_m4", 114)),
        require_weekly_ma_bull=bool(cfg_brick.get("require_weekly_ma_bull", True)),
        wma_short=int(cfg_brick.get("wma_short", 20)),
        wma_mid=int(cfg_brick.get("wma_mid", 60)),
        wma_long=int(cfg_brick.get("wma_long", 120)),
        n=int(cfg_brick.get("n", 4)),
        m1=int(cfg_brick.get("m1", 4)),
        m2=int(cfg_brick.get("m2", 6)),
        m3=int(cfg_brick.get("m3", 6)),
        t=float(cfg_brick.get("t", 4.0)),
        shift1=float(cfg_brick.get("shift1", 90.0)),
        shift2=float(cfg_brick.get("shift2", 100.0)),
        sma_w1=int(cfg_brick.get("sma_w1", 1)),
        sma_w2=int(cfg_brick.get("sma_w2", 1)),
        sma_w3=int(cfg_brick.get("sma_w3", 1)),
    )

    date_str = pick_date.strftime("%Y-%m-%d")
    candidates: List[Candidate] = []

    for code in pool_codes:        
        df = prepared.get(code)        
        if df is None or pick_date not in df.index:
            continue
        try:
            pf = selector.prepare_df(df)
            if selector.vec_picks_from_prepared(pf, start=pick_date, end=pick_date):
                row = pf.loc[pick_date]
                bg = float(row["brick_growth"]) if "brick_growth" in pf.columns else selector.brick_growth_on_date(pf, pick_date)
                candidates.append(Candidate(
                    code=code,
                    date=date_str,
                    strategy="brick",
                    close=float(row["close"]),
                    turnover_n=float(row["turnover_n"]),
                    brick_growth=bg if np.isfinite(bg) else None,
                ))
        except Exception as exc:
            logger.debug("Brick skip %s: %s", code, exc)

    candidates.sort(key=lambda c: c.brick_growth or -999, reverse=True)
    logger.info("Brick 选出: %d 只", len(candidates))
    return candidates


# =============================================================================
# 黄金针策略
# =============================================================================

def _needle_signal_type(row: pd.Series) -> str:
    """返回当日黄金针细分类，便于候选结果解释。"""
    is_golden = bool(row.get("golden_needle", False))
    is_platinum = bool(row.get("platinum_needle", False))
    if is_golden and is_platinum:
        return "golden+platinum"
    if is_golden:
        return "golden"
    if is_platinum:
        return "platinum"
    return ""


def _needle_triggered_conditions(row: pd.Series) -> List[str]:
    return [f"COND{i}" for i in range(1, 7) if bool(row.get(f"needle_cond{i}", False))]


def run_golden_needle(
    prepared: Dict[str, pd.DataFrame],
    pick_date: pd.Timestamp,
    pool_codes: List[str],
    cfg_golden_needle: dict,
) -> List[Candidate]:
    """运行通达信黄金针/白金针选股公式，返回 Candidate 列表。"""
    selector = GoldenNeedleSelector(
        n1=int(cfg_golden_needle.get("n1", 3)),
        n2=int(cfg_golden_needle.get("n2", 21)),
        include_platinum=bool(cfg_golden_needle.get("include_platinum", True)),
        cond1_long_min=float(cfg_golden_needle.get("cond1_long_min", 79.0)),
        cond1_short_max=float(cfg_golden_needle.get("cond1_short_max", 30.0)),
        cond2_gap_min=float(cfg_golden_needle.get("cond2_gap_min", 58.0)),
        cond2_long_min=float(cfg_golden_needle.get("cond2_long_min", 70.0)),
        cond6_count_window=int(cfg_golden_needle.get("cond6_count_window", 8)),
        cond6_count_min=int(cfg_golden_needle.get("cond6_count_min", 4)),
        cond6_count_short_max=float(cfg_golden_needle.get("cond6_count_short_max", 75.0)),
        cond6_long_window=int(cfg_golden_needle.get("cond6_long_window", 9)),
        cond6_long_min=float(cfg_golden_needle.get("cond6_long_min", 85.0)),
        cond6_short_max=float(cfg_golden_needle.get("cond6_short_max", 60.0)),
    )

    date_str = pick_date.strftime("%Y-%m-%d")
    candidates: List[Candidate] = []

    for code in pool_codes:
        df = prepared.get(code)
        if df is None or pick_date not in df.index:
            continue
        try:
            pf = selector.prepare_df(df)
            if selector.vec_picks_from_prepared(pf, start=pick_date, end=pick_date):
                row = pf.loc[pick_date]
                candidates.append(Candidate(
                    code=code,
                    date=date_str,
                    strategy="golden_needle",
                    close=float(row["close"]),
                    turnover_n=float(row["turnover_n"]),
                    extra={
                        "needle_short": round(float(row["needle_short"]), 2),
                        "needle_long": round(float(row["needle_long"]), 2),
                        "needle_signal": _needle_signal_type(row),
                        "needle_conditions": _needle_triggered_conditions(row),
                    },
                ))
        except Exception as exc:
            logger.debug("Golden needle skip %s: %s", code, exc)

    logger.info("GoldenNeedle 选出: %d 只", len(candidates))
    return candidates


# =============================================================================
# KG 动能策略
# =============================================================================

def _round_float(value: object, ndigits: int = 2) -> Optional[float]:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(val):
        return None
    return round(val, ndigits)


def _kg_signal_type(row: pd.Series, cfg_kg: dict) -> str:
    momentum = float(row.get("kg_momentum", np.nan))
    x_momentum = float(row.get("kg_x_momentum", np.nan))
    score = float(row.get("kg_score", np.nan))
    if not (np.isfinite(momentum) and np.isfinite(x_momentum)):
        return ""

    reversal = (
        momentum < float(cfg_kg.get("reversal_momentum_max", 20.0))
        and x_momentum >= float(cfg_kg.get("reversal_x_min", 50.0))
    )
    if bool(cfg_kg.get("require_score_for_reversal", False)):
        score_min = _optional_float(cfg_kg, "score_min", 10.0)
        score_max = _optional_float(cfg_kg, "score_max", 50.0)
        if score_min is not None:
            reversal = reversal and np.isfinite(score) and score >= score_min
        if score_max is not None:
            reversal = reversal and np.isfinite(score) and score <= score_max

    momentum_rebound = (
        momentum >= float(cfg_kg.get("trend_momentum_min", 30.0))
        and momentum <= float(cfg_kg.get("trend_momentum_max", 50.0))
        and x_momentum >= float(cfg_kg.get("trend_x_min", 30.0))
        and x_momentum <= float(cfg_kg.get("trend_x_max", 50.0))
    )
    score_min = _optional_float(cfg_kg, "score_min", 10.0)
    score_max = _optional_float(cfg_kg, "score_max", 50.0)
    if score_min is not None:
        momentum_rebound = momentum_rebound and np.isfinite(score) and score >= score_min
    if score_max is not None:
        momentum_rebound = momentum_rebound and np.isfinite(score) and score <= score_max

    trend = momentum_rebound
    if bool(cfg_kg.get("trend_require_close_above_ma20", True)):
        trend = trend and float(row.get("close", np.nan)) > float(row.get("kg_ma20", np.nan))
    trend_ma20_slope_min = _optional_float(cfg_kg, "trend_ma20_slope_min", 0.0)
    if trend_ma20_slope_min is not None:
        trend = trend and float(row.get("kg_ma20_slope5", np.nan)) >= trend_ma20_slope_min
    if bool(cfg_kg.get("trend_require_close_above_ma60", False)):
        trend = trend and float(row.get("close", np.nan)) > float(row.get("kg_ma60", np.nan))
    trend_ma60_slope_min = _optional_float(cfg_kg, "trend_ma60_slope_min", None)
    if trend_ma60_slope_min is not None:
        trend = trend and float(row.get("kg_ma60_slope5", np.nan)) >= trend_ma60_slope_min
    if bool(cfg_kg.get("trend_require_ma20_above_ma60", False)):
        trend = trend and float(row.get("kg_ma20", np.nan)) > float(row.get("kg_ma60", np.nan))
    trend_low20_distance_min = _optional_float(cfg_kg, "trend_low20_distance_min", None)
    if trend_low20_distance_min is not None:
        trend = trend and float(row.get("kg_low20_distance", np.nan)) >= trend_low20_distance_min

    if reversal and trend:
        return "reversal+trend_continue"
    if reversal:
        return "reversal"
    if trend:
        return "trend_continue"
    if momentum_rebound:
        return "momentum_rebound"
    return str(cfg_kg.get("mode", "kg_momentum"))


def run_kg_momentum(
    prepared: Dict[str, pd.DataFrame],
    pick_date: pd.Timestamp,
    pool_codes: List[str],
    cfg_kg: dict,
) -> List[Candidate]:
    """运行 KG/JRX 动能选股公式，返回 Candidate 列表。"""
    selector = KGMomentumSelector(
        mode=str(cfg_kg.get("mode", "both")),
        reversal_momentum_max=float(cfg_kg.get("reversal_momentum_max", 20.0)),
        reversal_x_min=float(cfg_kg.get("reversal_x_min", 50.0)),
        trend_momentum_min=float(cfg_kg.get("trend_momentum_min", 30.0)),
        trend_momentum_max=float(cfg_kg.get("trend_momentum_max", 50.0)),
        trend_x_min=float(cfg_kg.get("trend_x_min", 30.0)),
        trend_x_max=float(cfg_kg.get("trend_x_max", 50.0)),
        score_min=_optional_float(cfg_kg, "score_min", 10.0),
        score_max=_optional_float(cfg_kg, "score_max", 50.0),
        overheat_momentum_max=_optional_float(cfg_kg, "overheat_momentum_max", 60.0),
        require_volume_confirm=bool(cfg_kg.get("require_volume_confirm", True)),
        volume_ratio_min=float(cfg_kg.get("volume_ratio_min", 0.9)),
        require_score_for_reversal=bool(cfg_kg.get("require_score_for_reversal", False)),
        trend_require_close_above_ma20=bool(cfg_kg.get("trend_require_close_above_ma20", True)),
        trend_ma20_slope_min=_optional_float(cfg_kg, "trend_ma20_slope_min", 0.0),
        trend_require_close_above_ma60=bool(cfg_kg.get("trend_require_close_above_ma60", False)),
        trend_ma60_slope_min=_optional_float(cfg_kg, "trend_ma60_slope_min", None),
        trend_require_ma20_above_ma60=bool(cfg_kg.get("trend_require_ma20_above_ma60", False)),
        trend_low20_distance_min=_optional_float(cfg_kg, "trend_low20_distance_min", None),
    )

    date_str = pick_date.strftime("%Y-%m-%d")
    candidates: List[Candidate] = []

    for code in pool_codes:
        df = prepared.get(code)
        if df is None or pick_date not in df.index:
            continue
        try:
            pf = selector.prepare_df(df)
            if selector.vec_picks_from_prepared(pf, start=pick_date, end=pick_date):
                row = pf.loc[pick_date]
                candidates.append(Candidate(
                    code=code,
                    date=date_str,
                    strategy="kg_momentum",
                    close=float(row["close"]),
                    turnover_n=float(row["turnover_n"]),
                    extra={
                        "kg_signal": _kg_signal_type(row, cfg_kg),
                        "kg_momentum": _round_float(row.get("kg_momentum")),
                        "kg_x_momentum": _round_float(row.get("kg_x_momentum")),
                        "kg_score": _round_float(row.get("kg_score")),
                        "kg_vol_ratio": _round_float(row.get("kg_vol_ratio")),
                        "kg_bonus": _round_float(row.get("kg_bonus_ratio"), 3),
                        "kg_total_penalty": _round_float(row.get("kg_total_penalty")),
                        "kg_overhead_v20": _round_float(row.get("kg_overhead_v20")),
                        "kg_ret_z": _round_float(row.get("kg_ret_z")),
                        "kg_j_delta": _round_float(row.get("kg_j_delta")),
                        "kg_rsi_delta": _round_float(row.get("kg_rsi_delta")),
                        "kg_ret20": _round_float(row.get("kg_ret20"), 4),
                        "kg_ret60": _round_float(row.get("kg_ret60"), 4),
                        "kg_ma20_slope5": _round_float(row.get("kg_ma20_slope5"), 4),
                        "kg_ma60_slope5": _round_float(row.get("kg_ma60_slope5"), 4),
                        "kg_low20_distance": _round_float(row.get("kg_low20_distance"), 4),
                    },
                ))
        except Exception as exc:
            logger.debug("KGMomentum skip %s: %s", code, exc)

    candidates.sort(
        key=lambda c: (
            c.extra.get("kg_score") if c.extra.get("kg_score") is not None else -999,
            c.extra.get("kg_x_momentum") if c.extra.get("kg_x_momentum") is not None else -999,
        ),
        reverse=True,
    )
    logger.info("KGMomentum 选出: %d 只", len(candidates))
    return candidates


# =============================================================================
# 主入口
# =============================================================================

def run_preselect(
    *,
    config_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    end_date: Optional[str] = None,
    pick_date: Optional[str] = None,
) -> Tuple[pd.Timestamp, List[Candidate]]:
    """
    量化初选主函数，返回 (pick_date_ts, List[Candidate])。
    不写任何文件，由 cli.py 负责落盘。

    参数
    ----
    config_path : rules_preselect.yaml 路径（None = 默认）
    data_dir    : CSV 目录（None = 读配置）
    end_date    : 数据截断日期（回测用）
    pick_date   : 选股基准日期（None = 自动最新）
    """
    cfg = load_config(config_path)
    g = cfg.get("global", {})

    _data_dir_path = _resolve_cfg_path(data_dir or g.get("data_dir", "./data/raw"))
    _data_dir = str(_data_dir_path)
    top_m = int(g.get("top_m", 20))
    n_turnover_days = int(g.get("n_turnover_days", 43))
    min_bars_buffer = int(g.get("min_bars_buffer", 10))

    # 1) 加载原始数据
    raw_data = load_raw_data(_data_dir, end_date=end_date)

    # 2) 计算 warmup_bars
    warmup = _calc_warmup(cfg, min_bars_buffer)

    # 3) 通用数据预处理
    preparer = MarketDataPreparer(
        end_date=pd.to_datetime(end_date) if end_date else None,
        warmup_bars=warmup,
        n_turnover_days=n_turnover_days,
        selector=None,
    )
    prepared = preparer.prepare(raw_data)

    # 4) 确定选股日期
    pick_ts = _resolve_pick_date(prepared, pick_date)
    logger.info("选股日期: %s", pick_ts.date())

    # 5) 构建流动性池
    pool_codes = TopTurnoverPoolBuilder(top_m=top_m).build(prepared).get(pick_ts, [])
    if not pool_codes:
        logger.warning("流动性池为空，pick_date=%s", pick_ts.date())
        return pick_ts, []

    logger.info("流动性池: %d 只", len(pool_codes))
    pool_codes = _apply_universe_filters(
        pool_codes,
        pick_date=pick_ts,
        data_dir=_data_dir_path,
        cfg=cfg,
    )
    if not pool_codes:
        logger.warning("过滤后股票池为空，pick_date=%s", pick_ts.date())
        return pick_ts, []

    # 6) 运行各策略
    all_candidates: List[Candidate] = []

    if cfg.get("b1", {}).get("enabled", True):
        all_candidates.extend(run_b1(prepared, pick_ts, pool_codes, cfg["b1"]))

    if cfg.get("brick", {}).get("enabled", True):
        all_candidates.extend(run_brick(prepared, pick_ts, pool_codes, cfg["brick"]))

    if cfg.get("golden_needle", {}).get("enabled", False):
        all_candidates.extend(run_golden_needle(
            prepared,
            pick_ts,
            pool_codes,
            cfg.get("golden_needle", {}),
        ))

    if cfg.get("kg_momentum", {}).get("enabled", False):
        all_candidates.extend(run_kg_momentum(
            prepared,
            pick_ts,
            pool_codes,
            cfg.get("kg_momentum", {}),
        ))

    # 7) 去重/合并（同一只股票可记录多策略命中）
    deduped = _merge_strategy_candidates(all_candidates)

    logger.info("初选完成，候选股票: %d 只", len(deduped))
    return pick_ts, deduped
