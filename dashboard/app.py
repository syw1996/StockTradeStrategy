"""
dashboard/app.py
AgentTrader · 单票看盘 — Streamlit 主入口（单页）

启动方式：
    cd <项目根>
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

# ── 路径修正 ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
_DASH = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_DASH))

# ── 辅助加载 ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def _load_cfg() -> dict:
    p = _ROOT / "config" / "dashboard.yaml"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _resolve_project_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (_ROOT / p)

@st.cache_data(ttl=30)
def _load_candidates_map() -> dict[str, dict]:
    cfg = _load_cfg()
    rel = cfg.get("paths", {}).get("candidates_latest", "data/candidates/candidates_latest.json")
    p = _ROOT / rel
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {c["code"]: c for c in data.get("candidates", [])}


@st.cache_data(show_spinner=False)
def _load_raw(code: str) -> pd.DataFrame:
    cfg = _load_cfg()
    raw_dir = _resolve_project_path(cfg.get("paths", {}).get("raw_data_dir", "data/raw"))
    csv = raw_dir / f"{code}.csv"
    if not csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ── 页面配置 ─────────────────────────────────────────────────────────────────
cfg = _load_cfg()
page_title = cfg.get("server", {}).get("title", "AgentTrader · 看盘")

st.set_page_config(
    page_title=page_title,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

_css_path = _DASH / "assets" / "style.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True)

# ── 组件导入 ─────────────────────────────────────────────────────────────────
from components.charts import make_daily_chart, make_weekly_chart

# ── 图表参数 ─────────────────────────────────────────────────────────────────
chart_cfg        = cfg.get("chart", {})
weekly_ma_wins   = chart_cfg.get("weekly_ma_windows", [5, 10, 20, 60])
weekly_ma_colors = {int(k): v for k, v in chart_cfg.get("weekly_ma_colors", {}).items()}
vol_up   = chart_cfg.get("volume_up_color",  "rgba(220,53,69,0.7)")
vol_down = chart_cfg.get("volume_down_color", "rgba(40,167,69,0.7)")

candidates_map  = _load_candidates_map()
candidate_codes = sorted(candidates_map.keys())

# ── 侧边栏 ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 AgentTrader")
    st.markdown("---")
    st.markdown("### 🔍 选择股票")

    if candidate_codes:
        st.markdown("**今日候选股票**")
        quick_code = st.selectbox("快速选择候选", ["— 手动输入 —"] + candidate_codes)
    else:
        quick_code = "— 手动输入 —"

    manual_code = st.text_input("手动输入代码（6位）", placeholder="例：600519")

    if manual_code.strip():
        active_code = manual_code.strip().zfill(6)
    elif quick_code and quick_code != "— 手动输入 —":
        active_code = quick_code
    else:
        active_code = None

    st.markdown("---")
    st.markdown("### ⚙️ 图表设置")

    bars_options = {"近60根": 60, "近120根": 120, "近250根": 250, "全部": 0}
    bars_label = st.selectbox("显示K线数量", list(bars_options.keys()), index=1)
    bars = bars_options[bars_label]

    weekly_ma_sel = st.multiselect(
        "周线均线",
        options=weekly_ma_wins,
        default=weekly_ma_wins,
        format_func=lambda x: f"MA{x}",
    )

# ── 主体 ─────────────────────────────────────────────────────────────────────
st.markdown("## 🔍 单票看盘")

if not active_code:
    st.info("👈 请在左侧选择或输入一只股票代码。")
    st.stop()

st.markdown(f"### {active_code}")

candidate = candidates_map.get(active_code)
if candidate:
    bg_val = candidate.get("brick_growth")
    bg_str = f"{bg_val:.3f}x" if bg_val is not None else "—"
    strat  = candidate.get("strategy", "")
    badge_cls = f"strategy-{strat}"
    st.markdown(
        f"<span class='candidate-strategy {badge_cls}'>{strat}</span>"
        f"&nbsp;&nbsp;收盘 <b>{candidate.get('close', 0):.2f}</b>"        
        f"&nbsp;·&nbsp; 选股日 {candidate.get('date','—')}",
        unsafe_allow_html=True,
    )
else:
    st.caption("（不在今日候选列表中）")

df_raw = _load_raw(active_code)

if df_raw.empty:
    st.error(f"❌ 未找到 `{active_code}.csv`，请检查日线数据目录配置。")
    st.stop()

# ── 图表 ─────────────────────────────────────────────────────────────────────
with st.spinner("加载图表..."):
    fig_daily = make_daily_chart(
        df_raw, active_code,
        volume_up_color=vol_up,
        volume_down_color=vol_down,
        bars=bars,
        height=720,
    )
    st.plotly_chart(fig_daily, use_container_width=True, config={"scrollZoom": True})

    fig_weekly = make_weekly_chart(
        df_raw, active_code,
        ma_windows=weekly_ma_sel or weekly_ma_wins,
        ma_colors=weekly_ma_colors,
        volume_up_color=vol_up,
        volume_down_color=vol_down,
        height=520,
    )
    st.plotly_chart(fig_weekly, use_container_width=True, config={"scrollZoom": True})
