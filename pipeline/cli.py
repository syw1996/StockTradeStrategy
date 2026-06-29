"""
pipeline/cli.py
统一命令行入口。

用法：
  python -m pipeline.cli preselect
  python -m pipeline.cli preselect --date 2025-12-31
  python -m pipeline.cli preselect --config config/rules_preselect.yaml --data data/raw

子命令：
  preselect   运行量化初选，写入 data/candidates/
"""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path

# 将 pipeline 目录加入 path（直接用 python cli.py 时需要）
sys.path.insert(0, str(Path(__file__).parent))

from select_stock import run_preselect, resolve_preselect_output_dir
from schemas import CandidateRun
from pipeline_io import save_candidates

# ── 日志配置 ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cli")


def _add_log_file(log_dir: str, pick_date: str) -> None:
    """可选：追加文件日志到 data/logs/pipeline_YYYY-MM-DD.log。"""
    p = Path(log_dir)
    p.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(p / f"pipeline_{pick_date}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)


# =============================================================================
# preselect 子命令
# =============================================================================

def cmd_preselect(args: argparse.Namespace) -> None:
    logger.info("===== 量化初选开始 =====")

    pick_ts, candidates = run_preselect(
        config_path=args.config or None,
        data_dir=args.data or None,
        end_date=args.end_date or None,
        pick_date=args.date or None,
    )

    pick_date_str = pick_ts.strftime("%Y-%m-%d")
    run_date_str = datetime.date.today().isoformat()

    # 可选日志文件
    if args.log_dir:
        _add_log_file(args.log_dir, pick_date_str)

    run = CandidateRun(
        run_date=run_date_str,
        pick_date=pick_date_str,
        candidates=candidates,
        meta={
            "config": args.config,
            "data_dir": args.data,
            "total": len(candidates),
        },
    )

    resolved_output_dir = resolve_preselect_output_dir(
        config_path=args.config or None,
        output_dir=args.output or None,
    )

    paths = save_candidates(
        run,
        candidates_dir=resolved_output_dir,
    )

    logger.info("===== 初选完成 =====")
    logger.info("选股日期  : %s", pick_date_str)
    logger.info("候选数量  : %d 只", len(candidates))
    for key, path in paths.items():
        logger.info("%-8s → %s", key, path)

    # 终端摘要
    if candidates:
        print(f"\n{'代码':>8}  {'策略':>18}  {'收盘价':>8}  {'砖型增长':>10}  {'信号':>16}")
        print("-" * 74)
        for c in candidates:
            bg = f"{c.brick_growth:.2f}x" if c.brick_growth is not None else "  —"
            signal = ""
            if c.extra:
                signal = str(
                    c.extra.get("needle_signal")
                    or c.extra.get("golden_needle_extra", {}).get("needle_signal")
                    or c.extra.get("kg_signal")
                    or c.extra.get("kg_momentum_extra", {}).get("kg_signal")
                    or ""
                )
            print(f"{c.code:>8}  {c.strategy:>18}  {c.close:>8.2f}  {bg:>10}  {signal:>16}")
    else:
        print("\n(今日无候选股票)")


# =============================================================================
# CLI 解析
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline.cli",
        description="AgentTrader 量化初选 CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preselect", help="运行量化初选")
    p.add_argument("--config", default=None, help="rules_preselect.yaml 路径")
    p.add_argument("--data",   default=None, help="CSV 数据目录（覆盖配置文件）")
    p.add_argument("--date",   default=None, help="选股基准日期 YYYY-MM-DD（默认最新）")
    p.add_argument("--end-date", dest="end_date", default=None,
                   help="数据截断日期（回测用）")
    p.add_argument("--output", default=None, help="候选输出目录（默认 data/candidates/）")
    p.add_argument("--log-dir", dest="log_dir", default=None,
                   help="流水日志目录（默认 data/logs/）")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "preselect":
        cmd_preselect(args)
    else:
        parser.print_help()
        sys.exit(1)
        
def test():
    """简单测试函数，验证 CLI 逻辑（不依赖外部数据）。"""
    class Args:
        command = "preselect"
        config = None
        data = None
        date = None
        end_date = None
        output = "./data/candidates"
        log_dir = "./data/logs"

    args = Args()
    cmd_preselect(args)


if __name__ == "__main__":
    main()  
    # test()
