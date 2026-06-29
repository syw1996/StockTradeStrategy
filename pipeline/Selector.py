"""
Selector.py — 模块化、向量化、Numba 加速选股框架
=================================================

设计原则
--------
1. **每个指标 / 条件独立成 Filter dataclass**
   - ``__call__(hist) -> bool``         逐行点查（回调 / 调试用）
   - ``vec_mask(df)  -> ndarray[bool]`` 全量向量化（批量回测用）

2. **PipelineSelector 基类**
   - ``prepare_df(df)``                  预计算所有中间列，返回含 ``_vec_pick`` 的 df
   - ``passes_df_on_date(df, date)``     单日判断
   - ``vec_picks_from_prepared(df)``     从预计算列批量获取通过日期

3. **Numba 加速**
   - KDJ 递推、砖型图核心循环、连续绿柱计数均使用 ``@njit`` 加速

Selector 一览
-------------
- ``B1Selector``          KDJ 分位 + 知行线 + 周线多头排列
- ``BrickChartSelector``  砖型图形态 + 知行线 + 周线多头排列
- ``GoldenNeedleSelector`` 黄金针/白金针 RSL 形态
- ``KGMomentumSelector``  KG/JRX 动能反转与趋势延续形态
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np
import pandas as pd
from numba import njit as _njit

# =============================================================================
# Numba 加速核心函数
# =============================================================================

# ── KDJ 核心递推 ──────────────────────────────────────────────────────────
@_njit(cache=True)
def _kdj_core(rsv: np.ndarray) -> tuple:          # noqa: UP006
    n = len(rsv)
    K = np.empty(n, dtype=np.float64)
    D = np.empty(n, dtype=np.float64)
    K[0] = D[0] = 50.0
    for i in range(1, n):
        K[i] = 2.0 / 3.0 * K[i - 1] + 1.0 / 3.0 * rsv[i]
        D[i] = 2.0 / 3.0 * D[i - 1] + 1.0 / 3.0 * K[i]
    J = 3.0 * K - 2.0 * D
    return K, D, J

# ── 连续绿柱计数 ──────────────────────────────────────────────────────────
@_njit(cache=True)
def _green_run(brick_vals: np.ndarray) -> np.ndarray:
    """green_run[i] = 截至 i-1 连续绿柱根数（brick < 0）。"""
    n = len(brick_vals)
    out = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        if brick_vals[i - 1] < 0.0:
            out[i] = out[i - 1] + 1
        else:
            out[i] = 0
    return out

# ── 成交量最大日非阴线核心 ───────────────────────────────────────────────
@_njit(cache=True)
def _max_vol_not_bearish(
    vol: np.ndarray, open_: np.ndarray, close: np.ndarray, n: int,
) -> np.ndarray:
    """滚动 n 日窗口内，成交量最大那天不为阴线（close >= open）。"""
    length = len(vol)
    mask = np.zeros(length, dtype=np.bool_)
    for i in range(length):
        start = max(0, i - n + 1)
        max_v   = vol[start]
        max_idx = start
        for j in range(start + 1, i + 1):
            if vol[j] > max_v:
                max_v   = vol[j]
                max_idx = j
        mask[i] = close[max_idx] >= open_[max_idx]
    return mask

# ── 砖型图核心 ────────────────────────────────────────────────────────────
@_njit(cache=True)
def _compute_brick_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    n: int, m1: int, m2: int, m3: int,
    t: float, shift1: float, shift2: float,
    sma_w1: int, sma_w2: int, sma_w3: int,
) -> np.ndarray:
        length = len(close)
        hhv = np.empty(length, dtype=np.float64)
        llv = np.empty(length, dtype=np.float64)
        for i in range(length):
            start = max(0, i - n + 1)
            h_max = high[start]; l_min = low[start]
            for j in range(start + 1, i + 1):
                if high[j] > h_max: h_max = high[j]
                if low[j]  < l_min: l_min = low[j]
            hhv[i] = h_max; llv[i] = l_min

        a1 = sma_w1 / m1; b1 = 1.0 - a1
        var2a = np.empty(length, dtype=np.float64)
        for i in range(length):
            rng = hhv[i] - llv[i]
            if rng == 0.0: rng = 0.01
            v1 = (hhv[i] - close[i]) / rng * 100.0 - shift1
            var2a[i] = (v1 + shift2) if i == 0 else (a1 * v1 + b1 * (var2a[i - 1] - shift2) + shift2)

        a2 = sma_w2 / m2; b2 = 1.0 - a2
        a3 = sma_w3 / m3; b3 = 1.0 - a3
        var4a = np.empty(length, dtype=np.float64)
        var5a = np.empty(length, dtype=np.float64)
        for i in range(length):
            rng = hhv[i] - llv[i]
            if rng == 0.0: rng = 0.01
            v3 = (close[i] - llv[i]) / rng * 100.0
            if i == 0:
                var4a[i] = v3; var5a[i] = v3 + shift2
            else:
                var4a[i] = a2 * v3 + b2 * var4a[i - 1]
                var5a[i] = a3 * var4a[i] + b3 * (var5a[i - 1] - shift2) + shift2

        raw = np.empty(length, dtype=np.float64)
        for i in range(length):
            diff = var5a[i] - var2a[i]
            raw[i] = diff - t if diff > t else 0.0

        brick = np.empty(length, dtype=np.float64)
        brick[0] = 0.0
        for i in range(1, length):
            brick[i] = raw[i] - raw[i - 1]
        return brick


# =============================================================================
# 指标计算辅助函数
# =============================================================================

def compute_kdj(df: pd.DataFrame, n: int = 9) -> pd.DataFrame:
    """返回带 K/D/J 列的 DataFrame（Numba 加速 KDJ 递推）。"""
    if df.empty:
        return df.assign(K=np.nan, D=np.nan, J=np.nan)
    low_n  = df["low"].rolling(window=n, min_periods=1).min()
    high_n = df["high"].rolling(window=n, min_periods=1).max()
    rsv    = ((df["close"] - low_n) / (high_n - low_n + 1e-9) * 100).to_numpy(dtype=np.float64)

    K, D, J = _kdj_core(rsv)
    return df.assign(K=K, D=D, J=J)


def _tdx_sma(series: pd.Series, period: int, weight: int = 1) -> pd.Series:
    """通达信 SMA(X,N,M)，alpha = weight/period。"""
    return series.ewm(alpha=weight / period, adjust=False).mean()


def compute_zx_lines(
    df: pd.DataFrame,
    m1: int = 14, m2: int = 28, m3: int = 57, m4: int = 114,
    zxdq_span: int = 10,
) -> tuple[pd.Series, pd.Series]:
    """返回 (zxdq, zxdkx)。zxdq=double-EWM；zxdkx=四均线平均。"""
    close = df["close"].astype(float)
    zxdq  = close.ewm(span=zxdq_span, adjust=False).mean().ewm(span=zxdq_span, adjust=False).mean()
    zxdkx = (
        close.rolling(m1, min_periods=m1).mean()
        + close.rolling(m2, min_periods=m2).mean()
        + close.rolling(m3, min_periods=m3).mean()
        + close.rolling(m4, min_periods=m4).mean()
    ) / 4.0
    return zxdq, zxdkx


def compute_weekly_close(df: pd.DataFrame) -> pd.Series:
    """日线 → 周线收盘价（每周最后一个实际交易日）。

    不依赖固定 resample 锚点（周日/周五），而是直接按
    ISO 周编号分组取最后一行，index 保持为真实交易日日期。
    """
    close = (
        df["close"].astype(float)
        if isinstance(df.index, pd.DatetimeIndex)
        else df.set_index("date")["close"].astype(float)
    )
    # 按 ISO 年+周分组，取每组最后一个交易日的收盘价
    # isocalendar().week 返回 1-53，加上年份避免跨年混淆
    idx = close.index
    year_week = idx.isocalendar().year.astype(str) + "-" + idx.isocalendar().week.astype(str).str.zfill(2)
    weekly = close.groupby(year_week).last()
    # 将 index 换回真实日期（每周最后交易日）
    last_date_per_week = close.groupby(year_week).apply(lambda s: s.index[-1])
    weekly.index = pd.DatetimeIndex(last_date_per_week.values)
    return weekly.dropna()


def compute_weekly_ma_bull(
    df: pd.DataFrame,
    ma_periods: tuple[int, int, int] = (20, 60, 120),
) -> pd.Series:
    """
    周线均线多头排列标志（MA_short > MA_mid > MA_long），
    forward-fill 到日线 index，返回 bool Series。

    周线收盘价 index 为真实交易日，reindex 后 ffill 可正确对齐。
    """
    weekly_close = compute_weekly_close(df)
    s, m, l = ma_periods
    ma_s = weekly_close.rolling(s, min_periods=s).mean()
    ma_m = weekly_close.rolling(m, min_periods=m).mean()
    ma_l = weekly_close.rolling(l, min_periods=l).mean()
    bull = (ma_s > ma_m) & (ma_m > ma_l)

    daily_index = (
        df.index if isinstance(df.index, pd.DatetimeIndex)
        else pd.DatetimeIndex(df["date"])
    )
    # 转 float（1.0/0.0/NaN）→ reindex → ffill → 填 0 → bool
    # 避免 bool reindex 后升级为 object dtype 触发 FutureWarning
    bull_daily = (
        bull.astype(float)
        .reindex(daily_index)
        .ffill()
        .fillna(0.0)
        .astype(bool)
    )
    return bull_daily


def compute_brick_chart(
    df: pd.DataFrame,
    *,
    n: int = 4, m1: int = 4, m2: int = 6, m3: int = 6,
    t: float = 4.0, shift1: float = 90.0, shift2: float = 100.0,
    sma_w1: int = 1, sma_w2: int = 1, sma_w3: int = 1,
) -> pd.Series:
    """通达信砖型图公式 → 砖高 Series（red>0，green<0）。"""
    arr = _compute_brick_numba(
        df["high"].to_numpy(dtype=np.float64),
        df["low"].to_numpy(dtype=np.float64),
        df["close"].to_numpy(dtype=np.float64),
        n, m1, m2, m3, float(t), float(shift1), float(shift2),
        sma_w1, sma_w2, sma_w3,
    )
    return pd.Series(arr, index=df.index, name="brick")


def _rolling_count(cond: pd.Series, window: int) -> pd.Series:
    """通达信 COUNT(cond, window) 等价实现。"""
    return cond.astype(int).rolling(window, min_periods=window).sum()


def _rolling_every(cond: pd.Series, window: int) -> pd.Series:
    """通达信 EVERY(cond, window) 等价实现。"""
    return _rolling_count(cond, window) == window


def compute_golden_needle(
    df: pd.DataFrame,
    *,
    n1: int = 3,
    n2: int = 21,
    cond1_long_min: float = 79.0,
    cond1_short_max: float = 30.0,
    cond2_gap_min: float = 58.0,
    cond2_long_min: float = 70.0,
    cond3_count_window: int = 5,
    cond3_count_min: int = 3,
    cond3_long_window: int = 6,
    cond3_long_min: float = 80.0,
    cond3_short_max: float = 70.0,
    cond4_count_window: int = 3,
    cond4_count_min: int = 2,
    cond4_long_window: int = 4,
    cond4_long_min: float = 80.0,
    cond4_short_max: float = 60.0,
    cond5_long_min: float = 75.0,
    cond5_short_max: float = 50.0,
    cond6_count_window: int = 8,
    cond6_count_min: int = 4,
    cond6_count_short_max: float = 75.0,
    cond6_long_window: int = 9,
    cond6_long_min: float = 85.0,
    cond6_short_max: float = 60.0,
) -> pd.DataFrame:
    """
    通达信黄金针公式的向量化 Python 版本。

    短期 = 100 * (C - LLV(L, N1)) / (HHV(C, N1) - LLV(L, N1))
    长期 = 100 * (C - LLV(L, N2)) / (HHV(C, N2) - LLV(L, N2))

    返回列：
      - needle_short / needle_long
      - needle_cond1 ... needle_cond6
      - golden_needle: COND1 OR COND2
      - platinum_needle: COND3 OR COND4 OR COND5 OR COND6
      - golden_needle_pick: 六个条件任一触发
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    close = out["close"].astype(float)
    low = out["low"].astype(float)

    llv1 = low.rolling(n1, min_periods=n1).min()
    hhv1 = close.rolling(n1, min_periods=n1).max()
    llv2 = low.rolling(n2, min_periods=n2).min()
    hhv2 = close.rolling(n2, min_periods=n2).max()

    short_den = (hhv1 - llv1).replace(0, np.nan)
    long_den = (hhv2 - llv2).replace(0, np.nan)
    short = 100.0 * (close - llv1) / short_den
    long = 100.0 * (close - llv2) / long_den

    cond1 = (long >= cond1_long_min) & (short <= cond1_short_max)
    cond2 = ((long - short) > cond2_gap_min) & (long > cond2_long_min)
    cond3 = (
        (_rolling_count(short < cond3_short_max, cond3_count_window) >= cond3_count_min)
        & _rolling_every(long > cond3_long_min, cond3_long_window)
        & (short < cond3_short_max)
    )
    cond4 = (
        (_rolling_count(short < cond4_short_max, cond4_count_window) >= cond4_count_min)
        & _rolling_every(long > cond4_long_min, cond4_long_window)
        & (short < cond4_short_max)
    )
    cond5 = (cond1 | cond2).shift(1, fill_value=False) & (long > cond5_long_min) & (short < cond5_short_max)
    cond6 = (
        (_rolling_count(short < cond6_count_short_max, cond6_count_window) >= cond6_count_min)
        & _rolling_every(long > cond6_long_min, cond6_long_window)
        & (short < cond6_short_max)
    )

    out["needle_short"] = short
    out["needle_long"] = long
    out["needle_cond1"] = cond1.fillna(False)
    out["needle_cond2"] = cond2.fillna(False)
    out["needle_cond3"] = cond3.fillna(False)
    out["needle_cond4"] = cond4.fillna(False)
    out["needle_cond5"] = cond5.fillna(False)
    out["needle_cond6"] = cond6.fillna(False)
    out["golden_needle"] = out["needle_cond1"] | out["needle_cond2"]
    out["platinum_needle"] = (
        out["needle_cond3"]
        | out["needle_cond4"]
        | out["needle_cond5"]
        | out["needle_cond6"]
    )
    out["golden_needle_pick"] = out["golden_needle"] | out["platinum_needle"]
    return out


def compute_kg_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    KG/JRX 动能 V3 BETA 的向量化实现。

    核心输出：
      - kg_momentum: 基础动能（黄柱）
      - kg_x_momentum: X 动能（红柱）
      - kg_score: 真实得分（蓝色虚线原值，未加视觉偏移 +10）
      - kg_pick_reversal / kg_pick_momentum_rebound / kg_pick_trend_continue
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    close = out["close"].astype(float)
    open_ = out["open"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    volume = out["volume"].astype(float)

    lc = close.shift(1)
    diff = close - lc

    sma1 = _tdx_sma(diff.clip(lower=0), 3, 1)
    sma2 = _tdx_sma(diff.abs(), 3, 1)
    rsi3 = (sma1 / sma2.replace(0, np.nan) * 100.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    llv9 = low.rolling(9, min_periods=1).min()
    hhv9 = high.rolling(9, min_periods=1).max()
    rsv_denom = hhv9 - llv9
    rsv = ((close - llv9) / rsv_denom.replace(0, np.nan) * 100.0).fillna(50.0)
    k_val = _tdx_sma(rsv, 3, 1)
    d_val = _tdx_sma(k_val, 3, 1)
    j_val = 3.0 * k_val - 2.0 * d_val

    n1 = j_val - j_val.shift(1)
    n2 = rsi3 - rsi3.shift(1)

    pctchg = ((close - lc) / lc.replace(0, np.nan) * 100.0).replace([np.inf, -np.inf], np.nan)
    vol_ratio = (volume / volume.shift(1).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    true_bull = (close > open_) & (close > lc)

    mino_refc = pd.concat([open_, lc], axis=1).min(axis=1)
    sdenom = high - mino_refc
    upper_shadow_ratio = ((high - close) / sdenom.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    candle_min = pd.concat([open_, close], axis=1).min(axis=1)
    day_range = high - low
    lower_shadow_ratio = ((candle_min - low) / day_range.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    vol_slope = (1.2 - 1.0) / (6.0 - 2.5)
    vol_boost = pd.Series(1.0, index=out.index)
    vol_boost = vol_boost.where(~(true_bull & (vol_ratio >= 2.5)), 1.0 + vol_slope * (vol_ratio - 2.5))
    vol_boost = vol_boost.where(~(true_bull & (vol_ratio >= 6.0)), 1.2)

    shadow_factor = pd.Series(1.0, index=out.index)
    shadow_factor = shadow_factor.where(~true_bull, (0.70 - upper_shadow_ratio) * 1.3)

    base_momentum = (n1 + n2) / 2.0 * shadow_factor * vol_boost
    x_diff = (n1 + n2) - (n1.shift(1) + n2.shift(1))
    x_momentum = pd.Series(
        np.where(true_bull & (x_diff > 0), x_diff / 2.0 * shadow_factor * vol_boost, 0.0),
        index=out.index,
    )

    ret_mean = pctchg.rolling(45, min_periods=45).mean()
    ret_std = pctchg.rolling(45, min_periods=45).std()
    ret_z = ((pctchg - ret_mean) / ret_std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    vol_mean45 = volume.rolling(45, min_periods=45).mean()
    vol_ratio_ma = (volume / vol_mean45.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    llv20 = low.rolling(20, min_periods=20).min()
    hhv20 = high.rolling(20, min_periods=20).max()
    high_pos = ((high - llv20) / (hhv20 - llv20).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    true_bear = (open_ > close) & (close < lc)
    fake_bear_bull = (open_ > close) & (close >= lc)
    real_overhead_flow = pd.Series(
        np.where(true_bear, high_pos * vol_ratio_ma * np.maximum(0.0, -ret_z), 0.0),
        index=out.index,
    )
    fake_bear_drop = ((open_ - close) / lc.replace(0, np.nan) * 100.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    fake_overhead_flow = pd.Series(
        np.where(fake_bear_bull, high_pos * vol_ratio_ma * fake_bear_drop * 0.1, 0.0),
        index=out.index,
    )
    overhead_v20 = (real_overhead_flow + fake_overhead_flow).rolling(20, min_periods=20).sum().shift(1)

    bonus_ratio = ((vol_ratio - 5.0) / 5.0).clip(lower=0.0, upper=1.0)
    j_norm = ((j_val.shift(1) - 19.0) / 11.0).clip(lower=0.0, upper=1.0)
    rsi_norm = ((rsi3.shift(1) - 29.0) / 11.0).clip(lower=0.0, upper=1.0)
    z_norm = ((ret_z - 2.5) / 0.7).clip(lower=0.0, upper=1.0)
    v20_norm = ((overhead_v20 - 2.0) / 8.0).clip(lower=0.0, upper=1.0)

    total_penalty = 20.0 * j_norm + 30.0 * rsi_norm + 10.0 * z_norm + 35.0 * v20_norm
    score = base_momentum + bonus_ratio * 15.0 - total_penalty
    ma20 = close.rolling(20, min_periods=20).mean()
    ma60 = close.rolling(60, min_periods=60).mean()
    ma20_slope5 = ma20 / ma20.shift(5) - 1.0
    ma60_slope5 = ma60 / ma60.shift(5) - 1.0
    ret20 = close / close.shift(20) - 1.0
    ret60 = close / close.shift(60) - 1.0
    low20 = low.rolling(20, min_periods=20).min()
    low20_distance = close / low20.replace(0, np.nan) - 1.0

    out["kg_rsi3"] = rsi3
    out["kg_j"] = j_val
    out["kg_j_delta"] = n1
    out["kg_rsi_delta"] = n2
    out["kg_vol_ratio"] = vol_ratio
    out["kg_true_bull"] = true_bull.fillna(False)
    out["kg_upper_shadow_ratio"] = upper_shadow_ratio
    out["kg_lower_shadow_ratio"] = lower_shadow_ratio
    out["kg_shadow_factor"] = shadow_factor
    out["kg_vol_boost"] = vol_boost
    out["kg_momentum"] = base_momentum
    out["kg_x_momentum"] = x_momentum
    out["kg_ret_z"] = ret_z
    out["kg_overhead_v20"] = overhead_v20
    out["kg_bonus_ratio"] = bonus_ratio
    out["kg_j_norm"] = j_norm
    out["kg_rsi_norm"] = rsi_norm
    out["kg_z_norm"] = z_norm
    out["kg_v20_norm"] = v20_norm
    out["kg_total_penalty"] = total_penalty
    out["kg_score"] = score
    out["kg_ma20"] = ma20
    out["kg_ma60"] = ma60
    out["kg_ma20_slope5"] = ma20_slope5
    out["kg_ma60_slope5"] = ma60_slope5
    out["kg_ret20"] = ret20
    out["kg_ret60"] = ret60
    out["kg_low20_distance"] = low20_distance
    out["kg_pick_reversal"] = (base_momentum < 20.0) & (x_momentum >= 50.0)
    out["kg_pick_momentum_rebound"] = (
        base_momentum.between(30.0, 50.0)
        & x_momentum.between(30.0, 50.0)
        & score.between(10.0, 50.0)
    )
    out["kg_pick_trend_continue"] = (
        out["kg_pick_momentum_rebound"]
        & (close > ma20)
        & (ma20_slope5 >= 0.0)
    )
    out["kg_pick"] = out["kg_pick_reversal"] | out["kg_pick_trend_continue"]
    return out


# =============================================================================
# Protocol / 基类
# =============================================================================

class StockFilter(Protocol):
    """单股票过滤器：给定截至 date 的历史 DataFrame，返回是否通过。"""
    def __call__(self, hist: pd.DataFrame) -> bool: ...


class PipelineSelector:
    """
    通用 Selector 基类。

    子类通过 ``prepare_df()`` 预计算中间列（含 ``_vec_pick``），
    之后调用 ``vec_picks_from_prepared()`` 批量获取通过日期（回测提速 10-50×）。
    """

    def __init__(
        self,
        filters: Sequence[StockFilter],
        *,
        date_col: str = "date",
        min_bars: int = 1,
        extra_bars_buffer: int = 0,
    ) -> None:
        self.filters           = list(filters)
        self.date_col          = date_col
        self.min_bars          = int(min_bars)
        self.extra_bars_buffer = int(extra_bars_buffer)

    # ── 内部工具 ─────────────────────────────────────────────────────────────

    def _get_hist(self, df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
        if self.date_col in df.columns:
            return df[df[self.date_col] <= date]
        if isinstance(df.index, pd.DatetimeIndex):
            return df.loc[:date]
        raise KeyError(
            f"DataFrame must have '{self.date_col}' column or a DatetimeIndex."
        )

    def _passes(self, hist: pd.DataFrame) -> bool:
        for f in self.filters:
            if not f(hist):
                return False
        return True

    # ── 公开 API ─────────────────────────────────────────────────────────────

    def get_hist(self, df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
        return self._get_hist(df, date)

    def passes_hist(self, hist: pd.DataFrame) -> bool:
        if hist is None or hist.empty:
            return False
        if len(hist) < self.min_bars + self.extra_bars_buffer:
            return False
        return self._passes(hist)

    def passes_df_on_date(self, df: pd.DataFrame, date: pd.Timestamp) -> bool:
        return self.passes_hist(self._get_hist(df, date))

    def select(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> List[str]:
        return [
            code for code, df in data.items()
            if self.passes_df_on_date(df, date)
        ]

    # ── 向量化批量接口（子类实现 prepare_df） ────────────────────────────────

    def prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """子类重写：预计算所有中间列及 ``_vec_pick``。"""
        return df

    def vec_picks_from_prepared(
        self,
        df: pd.DataFrame,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> List[pd.Timestamp]:
        """从已 prepare_df 的 df 快速获取通过日期列表（比逐日调用快 10-50×）。"""
        if "_vec_pick" not in df.columns:
            return []
        mask = df["_vec_pick"].astype(bool)
        if start is not None:
            mask = mask & (df.index >= start)
        if end is not None:
            mask = mask & (df.index <= end)
        return list(df.index[mask])


# =============================================================================
# ── 独立 Filter 模块 ──────────────────────────────────────────────────────────
#
# 每个 Filter 提供两套接口：
#   __call__(hist: pd.DataFrame) -> bool       点查（含 fallback 计算，调试用）
#   vec_mask(df: pd.DataFrame)  -> np.ndarray  全量向量化（prepare_df 内调用）
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 1. KDJ 分位过滤
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class KDJQuantileFilter:
    """
    J 值过滤：今日 J < j_threshold  OR  今日 J ≤ 历史累积 j_q_threshold 分位。

    vec_mask 使用 expanding().quantile() 保持无未来信息（真正历史分位）。
    """
    j_threshold:   float = -5.0
    j_q_threshold: float = 0.10
    kdj_n:         int   = 9

    def _j_series(self, hist: pd.DataFrame) -> pd.Series:
        if "J" in hist.columns:
            return hist["J"].astype(float)
        return compute_kdj(hist, n=self.kdj_n)["J"].astype(float)

    def __call__(self, hist: pd.DataFrame) -> bool:
        j = self._j_series(hist).dropna()
        if j.empty:
            return False
        j_today = float(j.iloc[-1])
        j_q     = float(j.quantile(self.j_q_threshold))
        return (j_today < self.j_threshold) or (j_today <= j_q)

    def vec_mask(self, df: pd.DataFrame) -> np.ndarray:
        """
        向量化：expanding 历史分位（无未来泄漏）。
        """
        J = self._j_series(df)
        j_vals  = J.to_numpy(dtype=float)
        j_q_exp = J.expanding(min_periods=1).quantile(self.j_q_threshold).to_numpy(dtype=float)
        return (j_vals < self.j_threshold) | (j_vals <= j_q_exp)


# ─────────────────────────────────────────────────────────────────────────────
# 2. 知行线条件过滤
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ZXConditionFilter:
    """
    知行线过滤：
      - close > zxdkx（长期均线）      [require_close_gt_long]
      - zxdq  > zxdkx（快线在均线上）  [require_short_gt_long]

    优先读 df 中预计算列 'zxdq' / 'zxdkx'。
    """
    zx_m1:  int = 10
    zx_m2:  int = 50
    zx_m3:  int = 200
    zx_m4:  int = 300
    zxdq_span: int = 10
    require_close_gt_long: bool = True
    require_short_gt_long: bool = True

    def _zx_vals(self, hist: pd.DataFrame) -> tuple[float, float, float]:
        """返回 (zxdq, zxdkx, close) 最新值。"""
        c = float(hist["close"].iloc[-1])
        if "zxdq" in hist.columns and "zxdkx" in hist.columns:
            s = float(hist["zxdq"].iloc[-1])
            lv = hist["zxdkx"].iloc[-1]
            l  = float(lv) if pd.notna(lv) else float("nan")
        else:
            zxdq, zxdkx = compute_zx_lines(
                hist, self.zx_m1, self.zx_m2, self.zx_m3, self.zx_m4,
                zxdq_span=self.zxdq_span,
            )
            s = float(zxdq.iloc[-1])
            l = float(zxdkx.iloc[-1]) if pd.notna(zxdkx.iloc[-1]) else float("nan")
        return s, l, c

    def __call__(self, hist: pd.DataFrame) -> bool:
        if hist.empty:
            return False
        s, l, c = self._zx_vals(hist)
        if not (np.isfinite(s) and np.isfinite(l)):
            return False
        if self.require_close_gt_long and not (c > l):
            return False
        if self.require_short_gt_long and not (s > l):
            return False
        return True

    def vec_mask(self, df: pd.DataFrame) -> np.ndarray:
        if "zxdq" in df.columns and "zxdkx" in df.columns:
            zxdq_v  = df["zxdq"].to_numpy(dtype=float)
            zxdkx_v = df["zxdkx"].to_numpy(dtype=float)
        else:
            zs, zk  = compute_zx_lines(
                df, self.zx_m1, self.zx_m2, self.zx_m3, self.zx_m4,
                zxdq_span=self.zxdq_span,
            )
            zxdq_v  = zs.to_numpy(dtype=float)
            zxdkx_v = zk.to_numpy(dtype=float)
        close_v = df["close"].to_numpy(dtype=float)
        mask    = np.isfinite(zxdq_v) & np.isfinite(zxdkx_v)
        if self.require_close_gt_long:
            mask &= close_v > zxdkx_v
        if self.require_short_gt_long:
            mask &= zxdq_v > zxdkx_v
        return mask


# ─────────────────────────────────────────────────────────────────────────────
# 3. 周线均线多头排列过滤
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WeeklyMABullFilter:
    """
    周线均线多头排列：MA_short > MA_mid > MA_long（默认 20/60/120 周）。
    优先读预计算列 'wma_bull'。
    """
    wma_short: int = 20
    wma_mid:   int = 60
    wma_long:  int = 120

    def __call__(self, hist: pd.DataFrame) -> bool:
        if "wma_bull" in hist.columns:
            return bool(hist["wma_bull"].iloc[-1])
        wc = compute_weekly_close(hist)
        if len(wc) < self.wma_long:
            return False
        ma_s = wc.rolling(self.wma_short, min_periods=self.wma_short).mean()
        ma_m = wc.rolling(self.wma_mid,   min_periods=self.wma_mid).mean()
        ma_l = wc.rolling(self.wma_long,  min_periods=self.wma_long).mean()
        sv, mv, lv = float(ma_s.iloc[-1]), float(ma_m.iloc[-1]), float(ma_l.iloc[-1])
        return bool(np.isfinite(sv) and np.isfinite(mv) and np.isfinite(lv) and sv > mv > lv)

    def vec_mask(self, df: pd.DataFrame) -> np.ndarray:
        if "wma_bull" in df.columns:
            return df["wma_bull"].to_numpy(dtype=bool)
        return compute_weekly_ma_bull(
            df, ma_periods=(self.wma_short, self.wma_mid, self.wma_long)
        ).to_numpy(dtype=bool)


# ─────────────────────────────────────────────────────────────────────────────
# 4. 最大成交量非阴线过滤
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MaxVolNotBearishFilter:
    """
    成交量最大日非阴线过滤：
    过去 n 个交易日（含当日）内，成交量最大的那一天不能是阴线
    （阴线定义：close < open）。

    ``vec_mask`` 使用 Numba 加速滚动窗口，复杂度 O(N×n)。
    """
    n: int = 20

    def __call__(self, hist: pd.DataFrame) -> bool:
        window = hist.tail(self.n)
        if window.empty or "volume" not in window.columns:
            return False
        idx_max_vol = window["volume"].idxmax()
        row = window.loc[idx_max_vol]
        return float(row["close"]) >= float(row["open"])

    def vec_mask(self, df: pd.DataFrame) -> np.ndarray:
        return _max_vol_not_bearish(
            df["volume"].to_numpy(dtype=np.float64),
            df["open"].to_numpy(dtype=np.float64),
            df["close"].to_numpy(dtype=np.float64),
            self.n,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. 砖型图相关 Filter（拆分为独立模块）
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BrickComputeParams:
    """
    砖型图计算参数容器，提供 compute / compute_arr 两个接口。
    作为其他 Filter 的共享配置嵌入使用。
    """
    n:      int   = 4
    m1:     int   = 4
    m2:     int   = 6
    m3:     int   = 6
    t:      float = 4.0
    shift1: float = 90.0
    shift2: float = 100.0
    sma_w1: int   = 1
    sma_w2: int   = 1
    sma_w3: int   = 1

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """返回砖高 Series（index 同 df）。"""
        return compute_brick_chart(
            df, n=self.n, m1=self.m1, m2=self.m2, m3=self.m3,
            t=self.t, shift1=self.shift1, shift2=self.shift2,
            sma_w1=self.sma_w1, sma_w2=self.sma_w2, sma_w3=self.sma_w3,
        )

    def compute_arr(self, df: pd.DataFrame) -> np.ndarray:
        """返回砖高 ndarray（直接调用 Numba JIT，避免 Series 包装开销）。"""
        return _compute_brick_numba(
            df["high"].to_numpy(dtype=np.float64),
            df["low"].to_numpy(dtype=np.float64),
            df["close"].to_numpy(dtype=np.float64),
            self.n, self.m1, self.m2, self.m3,
            float(self.t), float(self.shift1), float(self.shift2),
            self.sma_w1, self.sma_w2, self.sma_w3,
        )


@dataclass(frozen=True)
class BrickPatternFilter:
    """
    砖型图形态过滤（条件 1-5）：
      1. 今日涨幅 < daily_return_threshold
      2. 今日红柱（brick > 0）
      3. 昨日绿柱（brick[-1] < 0）
      4. 今日红柱高度 >= brick_growth_ratio × 昨日绿柱绝对高度
      5. 红柱前连续绿柱数 >= min_prior_green_bars

    优先读 df 中预计算列 'brick'。
    """
    daily_return_threshold: float = 0.05
    brick_growth_ratio:     float = 1.0
    min_prior_green_bars:   int   = 1
    brick_params: BrickComputeParams = field(default_factory=BrickComputeParams)

    def _brick_arr(self, hist: pd.DataFrame) -> np.ndarray:
        if "brick" in hist.columns:
            return hist["brick"].to_numpy(dtype=float)
        return self.brick_params.compute_arr(hist)

    def __call__(self, hist: pd.DataFrame) -> bool:
        min_len = max(3, 1 + self.min_prior_green_bars + 1)
        if len(hist) < min_len:
            return False
        close = hist["close"].to_numpy(dtype=float)
        c0, c1 = close[-1], close[-2]
        if c1 <= 0 or (c0 / c1 - 1.0) >= self.daily_return_threshold:
            return False
        vals = self._brick_arr(hist)
        b0, b1 = vals[-1], vals[-2]
        if not (b0 > 0 and b1 < 0):
            return False
        if b0 < self.brick_growth_ratio * abs(b1):
            return False
        # 连续绿柱
        green_count = 1
        i = len(vals) - 3
        while green_count < self.min_prior_green_bars and i > 0:
            if vals[i] < 0:
                green_count += 1
                i -= 1
            else:
                break
        return green_count >= self.min_prior_green_bars

    def vec_mask(self, df: pd.DataFrame) -> np.ndarray:
        """向量化：O(N)，使用预计算 'brick' 列（优先）或实时计算。"""
        bv  = self._brick_arr(df)
        cv  = df["close"].to_numpy(dtype=float)

        # 安全 shift（不用 np.roll，避免边界回绕）
        bp  = np.empty_like(bv);  bp[0]  = np.nan; bp[1:]  = bv[:-1]
        cp  = np.empty_like(cv);  cp[0]  = np.nan; cp[1:]  = cv[:-1]
        abp = np.abs(bp)

        cond_ret    = (cv / cp - 1.0) < self.daily_return_threshold
        cond_red    = bv > 0
        cond_green  = bp < 0
        cond_growth = bv >= self.brick_growth_ratio * abp

        if self.min_prior_green_bars <= 1:
            cond_gc = cond_green
        else:
            gr      = _green_run(bv)
            cond_gc = cond_green & (gr >= self.min_prior_green_bars)

        return cond_ret & cond_red & cond_gc & cond_growth

    def brick_growth_arr(self, df: pd.DataFrame) -> np.ndarray:
        """每日砖型图增长倍数数组（用于 top-k 排序）。"""
        bv  = self._brick_arr(df)
        bp  = np.empty_like(bv); bp[0] = np.nan; bp[1:] = bv[:-1]
        abp = np.abs(bp)
        safe = np.where(abp > 0, abp, 1.0)   # 分母置 1 避免除零警告
        return np.where(abp > 0, bv / safe, bv)


@dataclass(frozen=True)
class ZXDQRatioFilter:
    """
    砖型图选股条件 6：close < zxdq × zxdq_ratio。
    优先读预计算列 'zxdq'。
    """
    zxdq_ratio: float = 1.0
    zxdq_span:  int   = 10
    zxdkx_m1: int = 14; zxdkx_m2: int = 28; zxdkx_m3: int = 57; zxdkx_m4: int = 114

    def _zxdq_arr(self, df: pd.DataFrame) -> np.ndarray:
        if "zxdq" in df.columns:
            return df["zxdq"].to_numpy(dtype=float)
        zs, _ = compute_zx_lines(
            df, self.zxdkx_m1, self.zxdkx_m2, self.zxdkx_m3, self.zxdkx_m4,
            zxdq_span=self.zxdq_span,
        )
        return zs.to_numpy(dtype=float)

    def __call__(self, hist: pd.DataFrame) -> bool:
        zxdq_arr = self._zxdq_arr(hist)
        zv = float(zxdq_arr[-1])
        if not np.isfinite(zv) or zv <= 0:
            return False
        return float(hist["close"].iloc[-1]) < zv * self.zxdq_ratio

    def vec_mask(self, df: pd.DataFrame) -> np.ndarray:
        zxdq_v  = self._zxdq_arr(df)
        close_v = df["close"].to_numpy(dtype=float)
        return (
            np.isfinite(zxdq_v)
            & (zxdq_v > 0)
            & (close_v < zxdq_v * self.zxdq_ratio)
        )


@dataclass(frozen=True)
class GoldenNeedlePatternFilter:
    """黄金针/白金针信号过滤器。"""
    include_platinum: bool = True
    n1: int = 3
    n2: int = 21
    cond1_long_min: float = 79.0
    cond1_short_max: float = 30.0
    cond2_gap_min: float = 58.0
    cond2_long_min: float = 70.0
    cond6_count_window: int = 8
    cond6_count_min: int = 4
    cond6_count_short_max: float = 75.0
    cond6_long_window: int = 9
    cond6_long_min: float = 85.0
    cond6_short_max: float = 60.0

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        return compute_golden_needle(
            df,
            n1=self.n1,
            n2=self.n2,
            cond1_long_min=self.cond1_long_min,
            cond1_short_max=self.cond1_short_max,
            cond2_gap_min=self.cond2_gap_min,
            cond2_long_min=self.cond2_long_min,
            cond6_count_window=self.cond6_count_window,
            cond6_count_min=self.cond6_count_min,
            cond6_count_short_max=self.cond6_count_short_max,
            cond6_long_window=self.cond6_long_window,
            cond6_long_min=self.cond6_long_min,
            cond6_short_max=self.cond6_short_max,
        )

    def __call__(self, hist: pd.DataFrame) -> bool:
        if hist.empty:
            return False
        if "golden_needle_pick" not in hist.columns:
            hist = self._compute(hist)
        latest = hist.iloc[-1]
        if self.include_platinum:
            return bool(latest.get("golden_needle_pick", False))
        return bool(latest.get("golden_needle", False))

    def vec_mask(self, df: pd.DataFrame) -> np.ndarray:
        col = "golden_needle_pick" if self.include_platinum else "golden_needle"
        if col not in df.columns:
            df = self._compute(df)
        return df[col].to_numpy(dtype=bool)


@dataclass(frozen=True)
class KGMomentumPatternFilter:
    """KG/JRX 动能信号过滤器。"""
    mode: str = "both"
    reversal_momentum_max: float = 20.0
    reversal_x_min: float = 50.0
    trend_momentum_min: float = 30.0
    trend_momentum_max: float = 50.0
    trend_x_min: float = 30.0
    trend_x_max: float = 50.0
    score_min: Optional[float] = 10.0
    score_max: Optional[float] = 50.0
    overheat_momentum_max: Optional[float] = 60.0
    require_volume_confirm: bool = True
    volume_ratio_min: float = 0.9
    require_score_for_reversal: bool = False
    trend_require_close_above_ma20: bool = True
    trend_ma20_slope_min: Optional[float] = 0.0
    trend_require_close_above_ma60: bool = False
    trend_ma60_slope_min: Optional[float] = None
    trend_require_ma20_above_ma60: bool = False
    trend_low20_distance_min: Optional[float] = None

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        return compute_kg_momentum(df)

    def _trend_context_mask(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        mask = pd.Series(True, index=df.index)
        if self.trend_require_close_above_ma20:
            mask &= close > df["kg_ma20"].astype(float)
        if self.trend_require_close_above_ma60:
            mask &= close > df["kg_ma60"].astype(float)
        if self.trend_require_ma20_above_ma60:
            mask &= df["kg_ma20"].astype(float) > df["kg_ma60"].astype(float)
        if self.trend_ma20_slope_min is not None:
            mask &= df["kg_ma20_slope5"].astype(float) >= self.trend_ma20_slope_min
        if self.trend_ma60_slope_min is not None:
            mask &= df["kg_ma60_slope5"].astype(float) >= self.trend_ma60_slope_min
        if self.trend_low20_distance_min is not None:
            mask &= df["kg_low20_distance"].astype(float) >= self.trend_low20_distance_min
        return mask.fillna(False)

    def _mode_mask(self, df: pd.DataFrame) -> pd.Series:
        momentum = df["kg_momentum"].astype(float)
        x_momentum = df["kg_x_momentum"].astype(float)
        score = df["kg_score"].astype(float)

        reversal = (momentum < self.reversal_momentum_max) & (x_momentum >= self.reversal_x_min)
        if self.require_score_for_reversal:
            if self.score_min is not None:
                reversal &= score >= self.score_min
            if self.score_max is not None:
                reversal &= score <= self.score_max

        momentum_rebound = (
            momentum.between(self.trend_momentum_min, self.trend_momentum_max)
            & x_momentum.between(self.trend_x_min, self.trend_x_max)
        )
        if self.score_min is not None:
            momentum_rebound &= score >= self.score_min
        if self.score_max is not None:
            momentum_rebound &= score <= self.score_max

        trend = momentum_rebound & self._trend_context_mask(df)

        mode = self.mode.lower()
        if mode == "reversal":
            mask = reversal
        elif mode in {"momentum_rebound", "rebound", "left_to_right_early"}:
            mask = momentum_rebound
        elif mode in {"trend", "trend_continue", "continuation"}:
            mask = trend
        elif mode == "both":
            mask = reversal | trend
        else:
            raise ValueError(f"Unsupported KG momentum mode: {self.mode}")

        if self.overheat_momentum_max is not None:
            mask &= momentum <= self.overheat_momentum_max
        if self.require_volume_confirm:
            mask &= df["kg_vol_ratio"].astype(float) >= self.volume_ratio_min
        return mask.fillna(False)

    def __call__(self, hist: pd.DataFrame) -> bool:
        if hist.empty:
            return False
        if "kg_pick" not in hist.columns:
            hist = self._compute(hist)
        return bool(self._mode_mask(hist).iloc[-1])

    def vec_mask(self, df: pd.DataFrame) -> np.ndarray:
        if "kg_pick" not in df.columns:
            df = self._compute(df)
        return self._mode_mask(df).to_numpy(dtype=bool)


# =============================================================================
# ── 具体 Selector 实现 ────────────────────────────────────────────────────────
# =============================================================================

def _apply_vec_filters(df: pd.DataFrame, filters: list) -> np.ndarray:
    """对列表中所有实现了 vec_mask 的 Filter 取交集，返回布尔数组。"""
    mask = np.ones(len(df), dtype=bool)
    for f in filters:
        mask &= f.vec_mask(df)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# B1Selector
# ─────────────────────────────────────────────────────────────────────────────

class B1Selector(PipelineSelector):
    """
    B1 选股器：
      ① KDJQuantileFilter    — J 值低位（< j_threshold 或历史低分位）
      ② ZXConditionFilter    — close > zxdkx，zxdq > zxdkx
      ③ WeeklyMABullFilter   — 周线多头排列

    ``prepare_df()`` 预计算 K/D/J、zxdq/zxdkx、wma_bull、_vec_pick，
    之后用 ``vec_picks_from_prepared()`` 批量获取选股日期。
    """

    def __init__(
        self,
        j_threshold:          float = -5.0,
        j_q_threshold:        float = 0.10,
        kdj_n:                int   = 9,
        zx_m1:                int   = 10,
        zx_m2:                int   = 50,
        zx_m3:                int   = 200,
        zx_m4:                int   = 300,
        zxdq_span:            int   = 10,
        require_close_gt_long: bool = True,
        require_short_gt_long: bool = True,
        wma_short:            int   = 10,
        wma_mid:              int   = 20,
        wma_long:             int   = 30,
        max_vol_lookback:     Optional[int] = 20,
        *,
        date_col:          str = "date",
        extra_bars_buffer: int = 20,
    ) -> None:
        self._kdj_filter = KDJQuantileFilter(
            j_threshold=j_threshold, j_q_threshold=j_q_threshold, kdj_n=kdj_n,
        )
        self._zx_filter = ZXConditionFilter(
            zx_m1=zx_m1, zx_m2=zx_m2, zx_m3=zx_m3, zx_m4=zx_m4,
            zxdq_span=zxdq_span,
            require_close_gt_long=require_close_gt_long,
            require_short_gt_long=require_short_gt_long,
        )
        self._wma_filter = WeeklyMABullFilter(
            wma_short=wma_short, wma_mid=wma_mid, wma_long=wma_long,
        )
        self._max_vol_filter:MaxVolNotBearishFilter = MaxVolNotBearishFilter(n=max_vol_lookback) 
        _b1_filters: list = [self._kdj_filter, self._zx_filter, self._wma_filter, self._max_vol_filter]
        super().__init__(
            filters=_b1_filters,
            date_col=date_col,
            min_bars=max(30, zx_m4),
            extra_bars_buffer=extra_bars_buffer,
        )
        # 保存参数供 prepare_df
        self.kdj_n    = kdj_n
        self.zx_m1, self.zx_m2, self.zx_m3, self.zx_m4 = zx_m1, zx_m2, zx_m3, zx_m4
        self.zxdq_span = zxdq_span
        self.wma_short, self.wma_mid, self.wma_long = wma_short, wma_mid, wma_long

    def prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """预计算知行线、KDJ、周线多头排列、向量化 pick mask。"""
        df = df.copy()
        # 知行线
        zs, zk = compute_zx_lines(
            df, self.zx_m1, self.zx_m2, self.zx_m3, self.zx_m4,
            zxdq_span=self.zxdq_span,
        )
        df["zxdq"] = zs; df["zxdkx"] = zk
        # KDJ
        kdj = compute_kdj(df, n=self.kdj_n)
        df["K"] = kdj["K"]; df["D"] = kdj["D"]; df["J"] = kdj["J"]
        # 周线多头排列
        df["wma_bull"] = compute_weekly_ma_bull(
            df, ma_periods=(self.wma_short, self.wma_mid, self.wma_long)
        ).to_numpy()
        # 向量化 pick
        _b1_vec_filters: list = [self._kdj_filter, self._zx_filter, self._wma_filter]
        if self._max_vol_filter is not None:
            _b1_vec_filters.append(self._max_vol_filter)
        df["_vec_pick"] = _apply_vec_filters(df, _b1_vec_filters)
        return df


# ─────────────────────────────────────────────────────────────────────────────
# BrickChartSelector
# ─────────────────────────────────────────────────────────────────────────────

class BrickChartSelector(PipelineSelector):
    """
    砖型图选股器，由以下四个独立模块组成：
      ① BrickPatternFilter   — 形态（红/绿柱 + 涨幅 + 连续绿柱数）
      ② ZXDQRatioFilter      — close < zxdq × ratio          [可选]
      ③ ZXConditionFilter     — zxdq > zxdkx                  [可选]
      ④ WeeklyMABullFilter   — 周线多头排列                    [可选]

    快速用法::

        sel = BrickChartSelector()
        pf  = sel.prepare_df(daily_df)          # 一次性预计算（O(N)）
        dates = sel.vec_picks_from_prepared(pf)  # 批量获取通过日期

    超参搜索时，先 prepare_df() 预热慢速列，然后对每组砖型图参数
    调用 prepare_df_brick_only()（快 3-10×）。
    """

    def __init__(
        self,
        *,
        # ── 砖型图形态参数 ──
        daily_return_threshold: float = 0.05,
        brick_growth_ratio:     float = 1.0,
        min_prior_green_bars:   int   = 1,
        # ── 砖型图计算参数 ──
        n:      int   = 4,  m1: int = 4, m2: int = 6, m3: int = 6,
        t:      float = 4.0, shift1: float = 90.0, shift2: float = 100.0,
        sma_w1: int   = 1,   sma_w2: int = 1, sma_w3: int = 1,
        # ── 知行线参数 ──
        zxdq_span:  int = 10,
        zxdkx_m1: int = 14, zxdkx_m2: int = 28,
        zxdkx_m3: int = 57, zxdkx_m4: int = 114,
        # ── 条件开关 ──
        zxdq_ratio:             Optional[float] = 1.0,
        require_zxdq_gt_zxdkx: bool = True,
        require_weekly_ma_bull: bool = True,
        # ── 周线参数 ──
        wma_short: int = 20, wma_mid: int = 60, wma_long: int = 120,
        # ── 基类参数 ──
        date_col:          str = "date",
        extra_bars_buffer: int = 10,
    ) -> None:
        self._bp = BrickComputeParams(
            n=n, m1=m1, m2=m2, m3=m3,
            t=t, shift1=shift1, shift2=shift2,
            sma_w1=sma_w1, sma_w2=sma_w2, sma_w3=sma_w3,
        )
        self._pattern_filter = BrickPatternFilter(
            daily_return_threshold=daily_return_threshold,
            brick_growth_ratio=brick_growth_ratio,
            min_prior_green_bars=min_prior_green_bars,
            brick_params=self._bp,
        )
        self._zxdq_ratio_filter: Optional[ZXDQRatioFilter] = (
            ZXDQRatioFilter(
                zxdq_ratio=zxdq_ratio, zxdq_span=zxdq_span,
                zxdkx_m1=zxdkx_m1, zxdkx_m2=zxdkx_m2,
                zxdkx_m3=zxdkx_m3, zxdkx_m4=zxdkx_m4,
            ) if zxdq_ratio is not None else None
        )
        self._zxdq_gt_filter: Optional[ZXConditionFilter] = (
            ZXConditionFilter(
                zx_m1=zxdkx_m1, zx_m2=zxdkx_m2,
                zx_m3=zxdkx_m3, zx_m4=zxdkx_m4,
                zxdq_span=zxdq_span,
                require_close_gt_long=False,
                require_short_gt_long=True,
            ) if require_zxdq_gt_zxdkx else None
        )
        self._wma_filter: Optional[WeeklyMABullFilter] = (
            WeeklyMABullFilter(wma_short=wma_short, wma_mid=wma_mid, wma_long=wma_long)
            if require_weekly_ma_bull else None
        )

        # 传给基类的 filters（用于 _passes / passes_hist）
        _filters: list = [self._pattern_filter]
        if self._zxdq_ratio_filter is not None: _filters.append(self._zxdq_ratio_filter)
        if self._zxdq_gt_filter    is not None: _filters.append(self._zxdq_gt_filter)
        if self._wma_filter        is not None: _filters.append(self._wma_filter)

        super().__init__(
            _filters, date_col=date_col,
            min_bars=max(n + 3, 1 + min_prior_green_bars + 1, zxdkx_m4, wma_long * 5),
            extra_bars_buffer=extra_bars_buffer,
        )
        # 保存参数供 prepare_df
        self.zxdq_span  = zxdq_span
        self.zxdkx_m1, self.zxdkx_m2 = zxdkx_m1, zxdkx_m2
        self.zxdkx_m3, self.zxdkx_m4 = zxdkx_m3, zxdkx_m4
        self.wma_short, self.wma_mid, self.wma_long = wma_short, wma_mid, wma_long
        self.require_weekly_ma_bull = require_weekly_ma_bull

    # ── 预计算辅助 ─────────────────────────────────────────────────────────

    def _precompute_zx_wma(self, df: pd.DataFrame) -> None:
        """就地写入 zxdq / zxdkx / wma_bull。"""
        zs, zk = compute_zx_lines(
            df, self.zxdkx_m1, self.zxdkx_m2, self.zxdkx_m3, self.zxdkx_m4,
            zxdq_span=self.zxdq_span,
        )
        df["zxdq"] = zs; df["zxdkx"] = zk
        if self.require_weekly_ma_bull:
            df["wma_bull"] = compute_weekly_ma_bull(
                df, ma_periods=(self.wma_short, self.wma_mid, self.wma_long)
            ).to_numpy()

    def _precompute_brick(self, df: pd.DataFrame) -> None:
        """就地写入 brick / brick_growth。"""
        bv   = self._bp.compute_arr(df)
        bp_  = np.empty_like(bv); bp_[0] = np.nan; bp_[1:] = bv[:-1]
        abp  = np.abs(bp_)
        safe = np.where(abp > 0, abp, 1.0)    # 分母置 1 避免除零警告
        df["brick"]        = bv
        df["brick_growth"] = np.where(abp > 0, bv / safe, bv)

    def _compute_vec_pick(self, df: pd.DataFrame) -> np.ndarray:
        fs: list = [self._pattern_filter]
        if self._zxdq_ratio_filter is not None: fs.append(self._zxdq_ratio_filter)
        if self._zxdq_gt_filter    is not None: fs.append(self._zxdq_gt_filter)
        if self._wma_filter        is not None: fs.append(self._wma_filter)
        return _apply_vec_filters(df, fs)

    # ── 公开接口 ───────────────────────────────────────────────────────────

    def prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        完整预计算：brick + zxdq/zxdkx + wma_bull + brick_growth + _vec_pick。
        返回 df 副本。
        """
        df = df.copy()
        self._precompute_zx_wma(df)
        self._precompute_brick(df)
        df["_vec_pick"] = self._compute_vec_pick(df)
        return df

    def prepare_df_brick_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        轻量版 prepare_df：假设 zxdq / zxdkx / wma_bull 列已存在，
        仅重新计算 brick / brick_growth / _vec_pick。
        **就地修改 df**（不 copy），超参搜索内层循环专用，速度快 3-10×。
        """
        self._precompute_brick(df)
        df["_vec_pick"] = self._compute_vec_pick(df)
        return df

    def brick_growth_on_date(self, df: pd.DataFrame, date: pd.Timestamp) -> float:
        """
        返回指定日期的砖型图增长倍数（top-k 排序用）。
        优先读预计算列 'brick_growth'。
        """
        hist = self._get_hist(df, date)
        if len(hist) < 3:
            return -np.inf
        if "brick_growth" in hist.columns:
            val = float(hist["brick_growth"].iloc[-1])
            return val if np.isfinite(val) else -np.inf
        return float(self._pattern_filter.brick_growth_arr(hist)[-1])


class GoldenNeedleSelector(PipelineSelector):
    """黄金针选股器：完全复刻通达信 COND1-6，并保留中间指标列。"""

    def __init__(
        self,
        *,
        n1: int = 3,
        n2: int = 21,
        include_platinum: bool = True,
        cond1_long_min: float = 79.0,
        cond1_short_max: float = 30.0,
        cond2_gap_min: float = 58.0,
        cond2_long_min: float = 70.0,
        cond6_count_window: int = 8,
        cond6_count_min: int = 4,
        cond6_count_short_max: float = 75.0,
        cond6_long_window: int = 9,
        cond6_long_min: float = 85.0,
        cond6_short_max: float = 60.0,
        date_col: str = "date",
        extra_bars_buffer: int = 0,
    ) -> None:
        self.n1 = int(n1)
        self.n2 = int(n2)
        self.include_platinum = bool(include_platinum)
        self.cond1_long_min = float(cond1_long_min)
        self.cond1_short_max = float(cond1_short_max)
        self.cond2_gap_min = float(cond2_gap_min)
        self.cond2_long_min = float(cond2_long_min)
        self.cond6_count_window = int(cond6_count_window)
        self.cond6_count_min = int(cond6_count_min)
        self.cond6_count_short_max = float(cond6_count_short_max)
        self.cond6_long_window = int(cond6_long_window)
        self.cond6_long_min = float(cond6_long_min)
        self.cond6_short_max = float(cond6_short_max)
        self._pattern_filter = GoldenNeedlePatternFilter(
            include_platinum=self.include_platinum,
            n1=self.n1,
            n2=self.n2,
            cond1_long_min=self.cond1_long_min,
            cond1_short_max=self.cond1_short_max,
            cond2_gap_min=self.cond2_gap_min,
            cond2_long_min=self.cond2_long_min,
            cond6_count_window=self.cond6_count_window,
            cond6_count_min=self.cond6_count_min,
            cond6_count_short_max=self.cond6_count_short_max,
            cond6_long_window=self.cond6_long_window,
            cond6_long_min=self.cond6_long_min,
            cond6_short_max=self.cond6_short_max,
        )

        super().__init__(
            [self._pattern_filter],
            date_col=date_col,
            min_bars=max(self.n1, self.n2 + self.cond6_long_window - 1, 6) + 1,
            extra_bars_buffer=extra_bars_buffer,
        )

    def prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = compute_golden_needle(
            df,
            n1=self.n1,
            n2=self.n2,
            cond1_long_min=self.cond1_long_min,
            cond1_short_max=self.cond1_short_max,
            cond2_gap_min=self.cond2_gap_min,
            cond2_long_min=self.cond2_long_min,
            cond6_count_window=self.cond6_count_window,
            cond6_count_min=self.cond6_count_min,
            cond6_count_short_max=self.cond6_count_short_max,
            cond6_long_window=self.cond6_long_window,
            cond6_long_min=self.cond6_long_min,
            cond6_short_max=self.cond6_short_max,
        )
        df["_vec_pick"] = self._pattern_filter.vec_mask(df)
        return df


class KGMomentumSelector(PipelineSelector):
    """KG/JRX 动能选股器：识别洗盘反转与健康回调后的趋势延续。"""

    def __init__(
        self,
        *,
        mode: str = "both",
        reversal_momentum_max: float = 20.0,
        reversal_x_min: float = 50.0,
        trend_momentum_min: float = 30.0,
        trend_momentum_max: float = 50.0,
        trend_x_min: float = 30.0,
        trend_x_max: float = 50.0,
        score_min: Optional[float] = 10.0,
        score_max: Optional[float] = 50.0,
        overheat_momentum_max: Optional[float] = 60.0,
        require_volume_confirm: bool = True,
        volume_ratio_min: float = 0.9,
        require_score_for_reversal: bool = False,
        trend_require_close_above_ma20: bool = True,
        trend_ma20_slope_min: Optional[float] = 0.0,
        trend_require_close_above_ma60: bool = False,
        trend_ma60_slope_min: Optional[float] = None,
        trend_require_ma20_above_ma60: bool = False,
        trend_low20_distance_min: Optional[float] = None,
        date_col: str = "date",
        extra_bars_buffer: int = 0,
    ) -> None:
        self.mode = mode
        self._pattern_filter = KGMomentumPatternFilter(
            mode=mode,
            reversal_momentum_max=float(reversal_momentum_max),
            reversal_x_min=float(reversal_x_min),
            trend_momentum_min=float(trend_momentum_min),
            trend_momentum_max=float(trend_momentum_max),
            trend_x_min=float(trend_x_min),
            trend_x_max=float(trend_x_max),
            score_min=None if score_min is None else float(score_min),
            score_max=None if score_max is None else float(score_max),
            overheat_momentum_max=None if overheat_momentum_max is None else float(overheat_momentum_max),
            require_volume_confirm=bool(require_volume_confirm),
            volume_ratio_min=float(volume_ratio_min),
            require_score_for_reversal=bool(require_score_for_reversal),
            trend_require_close_above_ma20=bool(trend_require_close_above_ma20),
            trend_ma20_slope_min=None if trend_ma20_slope_min is None else float(trend_ma20_slope_min),
            trend_require_close_above_ma60=bool(trend_require_close_above_ma60),
            trend_ma60_slope_min=None if trend_ma60_slope_min is None else float(trend_ma60_slope_min),
            trend_require_ma20_above_ma60=bool(trend_require_ma20_above_ma60),
            trend_low20_distance_min=None if trend_low20_distance_min is None else float(trend_low20_distance_min),
        )
        super().__init__(
            [self._pattern_filter],
            date_col=date_col,
            min_bars=65,
            extra_bars_buffer=extra_bars_buffer,
        )

    def prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = compute_kg_momentum(df)
        df["_vec_pick"] = self._pattern_filter.vec_mask(df)
        return df


# =============================================================================
# AnySelector Protocol（外部类型提示用）
# =============================================================================

class AnySelector(Protocol):
    """外部代码面向接口编程时使用的 Protocol 类型。"""
    def passes_df_on_date(self, df: pd.DataFrame, date: pd.Timestamp) -> bool: ...
    def prepare_df(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def vec_picks_from_prepared(
        self, df: pd.DataFrame,
        start: Optional[pd.Timestamp] = None,
        end:   Optional[pd.Timestamp] = None,
    ) -> List[pd.Timestamp]: ...
