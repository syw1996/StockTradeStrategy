from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

PRICE_COLUMNS = ("open", "high", "low", "close")


def _resolve_project_path(path_like: str | Path, project_root: Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else project_root / path


def _normalise_factor_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    date_col = "trade_date" if "trade_date" in out.columns else "date"
    if date_col not in out.columns or "adj_factor" not in out.columns:
        return pd.DataFrame(columns=["date", "adj_factor"])

    date_raw = out[date_col].astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
    out["date"] = pd.to_datetime(date_raw, format="%Y%m%d", errors="coerce")
    out["adj_factor"] = pd.to_numeric(out["adj_factor"], errors="coerce")
    out = out[["date", "adj_factor"]].dropna().drop_duplicates("date", keep="last")
    return out.sort_values("date")


def _load_factor_frame(code: str, df: pd.DataFrame, factor_dir: Optional[Path]) -> pd.DataFrame:
    if "adj_factor" in df.columns:
        factor = _normalise_factor_frame(df[["date", "adj_factor"]].copy())
        if not factor.empty:
            return factor

    if factor_dir is None:
        return pd.DataFrame(columns=["date", "adj_factor"])

    factor_path = factor_dir / f"{str(code).zfill(6)}.csv"
    if not factor_path.exists():
        return pd.DataFrame(columns=["date", "adj_factor"])

    return _normalise_factor_frame(pd.read_csv(factor_path))


def _apply_qfq_to_one(
    code: str,
    df: pd.DataFrame,
    *,
    factor_dir: Optional[Path],
) -> tuple[pd.DataFrame, bool, str]:
    if df.empty or not all(col in df.columns for col in PRICE_COLUMNS):
        return df, False, "missing_ohlc"

    factor = _load_factor_frame(code, df, factor_dir)
    if factor.empty:
        return df, False, "missing_factor"

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    data_dates = out["date"].dropna()
    if data_dates.empty:
        return df, False, "missing_date"

    factor_start = factor["date"].min()
    factor_end = factor["date"].max()
    data_start = data_dates.min()
    data_end = data_dates.max()
    if factor_start > data_start:
        return (
            df,
            False,
            f"factor_start={factor_start.date()} after data_start={data_start.date()}",
        )
    if factor_end < data_end:
        return (
            df,
            False,
            f"factor_end={factor_end.date()} before data_end={data_end.date()}",
        )

    factor_series = (
        factor.set_index("date")["adj_factor"]
        .sort_index()
        .reindex(out["date"])
        .ffill()
    )
    if factor_series.isna().any():
        return df, False, "factor_gap"

    anchor = factor_series.dropna().iloc[-1]
    if not np.isfinite(anchor) or float(anchor) == 0.0:
        return df, False, "invalid_anchor_factor"

    ratio = factor_series.astype(float) / float(anchor)
    for col in PRICE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce") * ratio.to_numpy()

    out["adj_factor"] = factor_series.to_numpy()
    out["adj_ratio"] = ratio.to_numpy()
    return out, True, ""


def apply_price_adjustment(
    data: Dict[str, pd.DataFrame],
    adjustment_cfg: Optional[dict],
    *,
    project_root: Path,
) -> Dict[str, pd.DataFrame]:
    """Apply an in-memory qfq view using cached Tushare adj_factor files.

    Raw CSV files are never modified. The adjustment is anchored to each
    stock's latest available row in the loaded dataframe, so the latest close
    remains equal to the raw market close while historical OHLC is adjusted.
    """
    cfg = adjustment_cfg or {}
    if not bool(cfg.get("enabled", False)):
        return data

    method = str(cfg.get("method", "qfq")).strip().lower()
    if method in {"none", "raw", "off"}:
        return data
    if method != "qfq":
        raise ValueError(f"unsupported price adjustment method: {method}")

    factor_dir_value = cfg.get("adj_factor_dir", "./data/pit_metadata/adj_factor_by_code")
    factor_dir = _resolve_project_path(factor_dir_value, project_root) if factor_dir_value else None
    on_missing = str(cfg.get("on_missing", "warn")).strip().lower()

    adjusted: Dict[str, pd.DataFrame] = {}
    adjusted_count = 0
    missing_count = 0
    missing_examples: list[str] = []
    for code, df in data.items():
        out, ok, reason = _apply_qfq_to_one(code, df, factor_dir=factor_dir)
        adjusted[code] = out
        if ok:
            adjusted_count += 1
        else:
            missing_count += 1
            if len(missing_examples) < 10:
                missing_examples.append(f"{code}:{reason}")

    if missing_count:
        msg = (
            f"qfq adjustment missing/invalid for {missing_count} stocks "
            f"(factor_dir={factor_dir})"
        )
        if missing_examples:
            msg += f"; examples={', '.join(missing_examples)}"
        if on_missing == "error":
            raise FileNotFoundError(msg)
        if on_missing != "ignore":
            logger.warning(msg)
    logger.info("qfq price adjustment applied: adjusted=%d, missing=%d", adjusted_count, missing_count)
    return adjusted
