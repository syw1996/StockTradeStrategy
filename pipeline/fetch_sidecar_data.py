from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "pit_metadata"
DEFAULT_STOCKLIST = PROJECT_ROOT / "pipeline" / "stocklist.csv"

DAILY_BASIC_FIELDS = (
    "ts_code,trade_date,total_mv,circ_mv,turnover_rate,volume_ratio,pe,pb,close"
)
DAILY_MARKET_FIELDS = (
    "ts_code,trade_date,open,high,low,close,pct_chg,vol,amount"
)
STOCK_ST_FIELDS = "ts_code,name,trade_date,type,type_name"
FINA_FIELDS = "ts_code,ann_date,end_date,profit_dedt,debt_to_assets"
CASHFLOW_FIELDS = "ts_code,ann_date,end_date,n_cashflow_act"
ADJ_FACTOR_FIELDS = "ts_code,trade_date,adj_factor"
MONEYFLOW_DC_FIELDS = (
    "trade_date,ts_code,name,pct_change,close,net_amount,net_amount_rate,"
    "buy_elg_amount,buy_elg_amount_rate,buy_lg_amount,buy_lg_amount_rate,"
    "buy_md_amount,buy_md_amount_rate,buy_sm_amount,buy_sm_amount_rate"
)


def _import_tushare():
    try:
        import tushare as ts
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "tushare is not installed in this Python. Use the auto-quant-gpu "
            "Python or install requirements.txt."
        ) from exc
    return ts


def _resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _normalize_date(value: str) -> str:
    value = str(value).strip()
    if value.lower() == "today":
        value = datetime.now().strftime("%Y%m%d")
    value = value.replace("-", "")[:8]
    if len(value) != 8 or not value.isdigit():
        raise ValueError(f"Invalid date: {value}. Use YYYYMMDD, YYYY-MM-DD, or today.")
    return value


def _symbol_from_ts_code(ts_code: object) -> str:
    return str(ts_code).split(".", 1)[0].zfill(6)


def _ts_code_from_symbol(symbol: object) -> str:
    code = str(symbol).split(".", 1)[0].zfill(6)
    if code.startswith(("60", "68", "9")):
        return f"{code}.SH"
    if code.startswith(("4", "8")):
        return f"{code}.BJ"
    return f"{code}.SZ"


def _write_csv(df: pd.DataFrame, path: Path, columns: Iterable[str] | None = None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        columns = list(columns)
        for column in columns:
            if column not in df.columns:
                df[column] = pd.NA
        df = df[columns]
    df.to_csv(path, index=False)
    return int(len(df))


def _read_codes(stocklist: Path, limit_codes: int = 0) -> list[str]:
    if not stocklist.exists():
        raise FileNotFoundError(f"stocklist not found: {stocklist}")
    df = pd.read_csv(stocklist, dtype={"ts_code": str, "symbol": str})
    if "ts_code" in df.columns:
        codes = df["ts_code"].dropna().astype(str).tolist()
    elif "symbol" in df.columns:
        codes = [_ts_code_from_symbol(v) for v in df["symbol"].dropna().astype(str)]
    else:
        raise ValueError(f"stocklist must contain ts_code or symbol: {stocklist}")
    deduped = list(dict.fromkeys(codes))
    return deduped[:limit_codes] if limit_codes and limit_codes > 0 else deduped


@dataclass
class TushareClient:
    retries: int = 3
    retry_wait: float = 0.8

    def __post_init__(self) -> None:
        token = os.environ.get("TUSHARE_TOKEN", "").strip()
        if not token:
            raise RuntimeError("TUSHARE_TOKEN is not set.")
        ts = _import_tushare()
        ts.set_token(token)
        self.pro = ts.pro_api(token)

    def call(self, method_name: str, **kwargs) -> pd.DataFrame:
        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                df = getattr(self.pro, method_name)(**kwargs)
                return df if df is not None else pd.DataFrame()
            except Exception as exc:
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(self.retry_wait * attempt)
        raise RuntimeError(f"{method_name} failed after {self.retries} attempts: {last_exc}")


def find_trade_date(client: TushareClient, as_of: str) -> str:
    end_dt = datetime.strptime(as_of, "%Y%m%d")
    start = end_dt.replace(day=1).strftime("%Y%m%d")
    cal = client.call(
        "trade_cal",
        exchange="SSE",
        start_date=start,
        end_date=as_of,
        is_open="1",
        fields="cal_date,is_open",
    )
    if cal.empty:
        return as_of
    return str(cal["cal_date"].max())


def load_trade_dates(client: TushareClient, start: str, end: str) -> list[str]:
    cal = client.call(
        "trade_cal",
        exchange="SSE",
        start_date=start,
        end_date=end,
        is_open="1",
        fields="cal_date,is_open",
    )
    if cal.empty:
        return []
    return sorted(str(v) for v in cal["cal_date"].dropna().tolist())


def fetch_daily_sidecars(
    client: TushareClient,
    *,
    trade_date: str,
    out_dir: Path,
    incremental: bool,
) -> dict[str, object]:
    outputs: dict[str, object] = {"trade_date": trade_date}

    daily_basic_path = out_dir / "daily_basic" / f"{trade_date}.csv"
    if incremental and daily_basic_path.exists():
        daily_basic = pd.read_csv(daily_basic_path, dtype={"ts_code": str})
    else:
        daily_basic = client.call(
            "daily_basic",
            trade_date=trade_date,
            fields=DAILY_BASIC_FIELDS,
        )
        _write_csv(daily_basic, daily_basic_path, DAILY_BASIC_FIELDS.split(","))
    outputs["daily_basic_rows"] = int(len(daily_basic))
    outputs["daily_basic_path"] = str(daily_basic_path)

    daily_market_path = out_dir / "daily_market" / f"{trade_date}.csv"
    if incremental and daily_market_path.exists():
        daily_market = pd.read_csv(daily_market_path, dtype={"ts_code": str})
    else:
        daily_market = client.call(
            "daily",
            trade_date=trade_date,
            fields=DAILY_MARKET_FIELDS,
        )
        _write_csv(daily_market, daily_market_path, DAILY_MARKET_FIELDS.split(","))
    outputs["daily_market_rows"] = int(len(daily_market))
    outputs["daily_market_path"] = str(daily_market_path)

    stock_st_path = out_dir / "stock_st" / f"{trade_date}.csv"
    if incremental and stock_st_path.exists():
        stock_st = pd.read_csv(stock_st_path, dtype={"ts_code": str})
    else:
        stock_st = client.call(
            "stock_st",
            trade_date=trade_date,
            fields=STOCK_ST_FIELDS,
        )
        _write_csv(stock_st, stock_st_path, STOCK_ST_FIELDS.split(","))
    outputs["stock_st_rows"] = int(len(stock_st))
    outputs["stock_st_path"] = str(stock_st_path)

    moneyflow_path = out_dir / "moneyflow" / f"{trade_date}.csv"
    if incremental and moneyflow_path.exists():
        moneyflow = pd.read_csv(moneyflow_path, dtype={"ts_code": str})
    else:
        moneyflow = client.call("moneyflow", trade_date=trade_date)
        moneyflow = enrich_moneyflow(moneyflow, daily_market)
        _write_csv(moneyflow, moneyflow_path)
    outputs["moneyflow_rows"] = int(len(moneyflow))
    outputs["moneyflow_path"] = str(moneyflow_path)

    moneyflow_dc_path = out_dir / "moneyflow_dc" / f"{trade_date}.csv"
    if incremental and moneyflow_dc_path.exists():
        moneyflow_dc = pd.read_csv(moneyflow_dc_path, dtype={"ts_code": str})
    else:
        try:
            moneyflow_dc = client.call(
                "moneyflow_dc",
                trade_date=trade_date,
                fields=MONEYFLOW_DC_FIELDS,
            )
            moneyflow_dc = enrich_moneyflow_dc(moneyflow_dc)
        except Exception as exc:
            outputs["moneyflow_dc_error"] = str(exc)
            moneyflow_dc = pd.DataFrame(
                columns=[
                    *MONEYFLOW_DC_FIELDS.split(","),
                    "symbol",
                    "main_net_amount",
                    "main_net_ratio_pct",
                ]
            )
        _write_csv(moneyflow_dc, moneyflow_dc_path)
    outputs["moneyflow_dc_rows"] = int(len(moneyflow_dc))
    outputs["moneyflow_dc_path"] = str(moneyflow_dc_path)

    return outputs


def enrich_moneyflow_dc(moneyflow_dc: pd.DataFrame) -> pd.DataFrame:
    if moneyflow_dc.empty:
        return moneyflow_dc
    out = moneyflow_dc.copy()
    out["symbol"] = out["ts_code"].map(_symbol_from_ts_code)
    if "net_amount" in out.columns:
        out["main_net_amount"] = pd.to_numeric(out["net_amount"], errors="coerce")
    if "net_amount_rate" in out.columns:
        out["main_net_ratio_pct"] = pd.to_numeric(out["net_amount_rate"], errors="coerce")
    return out


def enrich_moneyflow(moneyflow: pd.DataFrame, daily_market: pd.DataFrame) -> pd.DataFrame:
    if moneyflow.empty:
        return moneyflow
    out = moneyflow.copy()
    out["symbol"] = out["ts_code"].map(_symbol_from_ts_code)

    for column in [
        "buy_lg_amount",
        "buy_elg_amount",
        "sell_lg_amount",
        "sell_elg_amount",
        "net_mf_amount",
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    if {"buy_lg_amount", "buy_elg_amount", "sell_lg_amount", "sell_elg_amount"}.issubset(out.columns):
        out["main_net_amount"] = (
            out["buy_lg_amount"]
            + out["buy_elg_amount"]
            - out["sell_lg_amount"]
            - out["sell_elg_amount"]
        )
    elif "net_mf_amount" in out.columns:
        out["main_net_amount"] = out["net_mf_amount"]
    else:
        out["main_net_amount"] = pd.NA

    if not daily_market.empty and "amount" in daily_market.columns:
        amount = daily_market[["ts_code", "amount"]].copy()
        amount["amount"] = pd.to_numeric(amount["amount"], errors="coerce")
        out = out.merge(amount, on="ts_code", how="left", suffixes=("", "_daily"))
        # Tushare daily.amount is in thousand CNY; moneyflow amount fields are commonly in 10k CNY.
        out["main_net_ratio_pct"] = out["main_net_amount"] * 10.0 / out["amount"].replace(0, pd.NA) * 100.0
    return out


def fetch_limit_up_ytd(
    client: TushareClient,
    *,
    trade_date: str,
    out_dir: Path,
    incremental: bool,
) -> dict[str, object]:
    start = f"{trade_date[:4]}0101"
    trade_dates = load_trade_dates(client, start, trade_date)
    raw_dir = out_dir / "limit_list_d"
    frames: list[pd.DataFrame] = []

    for idx, day in enumerate(trade_dates, 1):
        raw_path = raw_dir / f"{day}.csv"
        if incremental and raw_path.exists():
            df = pd.read_csv(raw_path, dtype={"ts_code": str})
        else:
            df = client.call("limit_list_d", trade_date=day)
            _write_csv(df, raw_path)
        if not df.empty:
            frames.append(df)
        print(f"[limit {idx}/{len(trade_dates)}] {day} rows={len(df)}")

    if frames:
        all_limits = pd.concat(frames, ignore_index=True)
        if "limit" in all_limits.columns:
            up = all_limits[all_limits["limit"].astype(str).str.upper().eq("U")].copy()
        else:
            up = all_limits.copy()
        summary = (
            up.groupby("ts_code", as_index=False)
            .agg(limit_up_days_ytd=("trade_date", "nunique"))
            .sort_values("limit_up_days_ytd", ascending=False)
        )
        summary["symbol"] = summary["ts_code"].map(_symbol_from_ts_code)
    else:
        summary = pd.DataFrame(columns=["ts_code", "limit_up_days_ytd", "symbol"])

    out_path = out_dir / "limit_up_ytd" / f"{trade_date}.csv"
    _write_csv(summary, out_path, ["ts_code", "symbol", "limit_up_days_ytd"])
    return {
        "trade_date": trade_date,
        "trade_dates_scanned": len(trade_dates),
        "limit_up_rows": int(len(summary)),
        "limit_up_ytd_path": str(out_path),
    }


def _latest_report(df: pd.DataFrame, as_of: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for column in ("ann_date", "end_date"):
        if column in out.columns:
            out[column] = out[column].astype(str)
    if "ann_date" in out.columns:
        out = out[out["ann_date"].le(as_of)]
    if out.empty:
        return out
    return out.sort_values(["end_date", "ann_date"], ascending=[False, False]).head(1)


def fetch_one_financial(
    client: TushareClient,
    ts_code: str,
    *,
    as_of: str,
    raw_dir: Path,
    refresh: bool,
) -> dict[str, object]:
    fina_path = raw_dir / "fina_indicator" / f"{ts_code}.csv"
    cashflow_path = raw_dir / "cashflow" / f"{ts_code}.csv"

    if fina_path.exists() and not refresh:
        fina = pd.read_csv(fina_path, dtype={"ts_code": str, "ann_date": str, "end_date": str})
    else:
        fina = client.call("fina_indicator", ts_code=ts_code, fields=FINA_FIELDS)
        _write_csv(fina, fina_path, FINA_FIELDS.split(","))

    if cashflow_path.exists() and not refresh:
        cashflow = pd.read_csv(cashflow_path, dtype={"ts_code": str, "ann_date": str, "end_date": str})
    else:
        cashflow = client.call("cashflow", ts_code=ts_code, fields=CASHFLOW_FIELDS)
        _write_csv(cashflow, cashflow_path, CASHFLOW_FIELDS.split(","))

    latest_fina = _latest_report(fina, as_of)
    latest_cashflow = _latest_report(cashflow, as_of)

    row: dict[str, object] = {"ts_code": ts_code, "symbol": _symbol_from_ts_code(ts_code)}
    if not latest_fina.empty:
        r = latest_fina.iloc[0]
        row.update(
            {
                "ann_date": r.get("ann_date", ""),
                "end_date": r.get("end_date", ""),
                "profit_dedt": r.get("profit_dedt", pd.NA),
                "debt_to_assets": r.get("debt_to_assets", pd.NA),
            }
        )
    if not latest_cashflow.empty:
        r = latest_cashflow.iloc[0]
        row["cashflow_ann_date"] = r.get("ann_date", "")
        row["cashflow_end_date"] = r.get("end_date", "")
        row["n_cashflow_act"] = r.get("n_cashflow_act", pd.NA)
    return row


def fetch_financial_latest(
    client: TushareClient,
    *,
    as_of: str,
    out_dir: Path,
    stocklist: Path,
    workers: int,
    limit_codes: int,
    refresh: bool,
) -> dict[str, object]:
    codes = _read_codes(stocklist, limit_codes=limit_codes)
    raw_dir = out_dir / "financial_raw"
    latest_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                fetch_one_financial,
                client,
                ts_code,
                as_of=as_of,
                raw_dir=raw_dir,
                refresh=refresh,
            ): ts_code
            for ts_code in codes
        }
        for idx, future in enumerate(as_completed(futures), 1):
            ts_code = futures[future]
            try:
                latest_rows.append(future.result())
                print(f"[financial {idx}/{len(codes)}] {ts_code}")
            except Exception as exc:
                failures.append({"ts_code": ts_code, "error": str(exc)})
                print(f"[financial {idx}/{len(codes)}] failed {ts_code}: {exc}")

    latest = pd.DataFrame(latest_rows)
    out_path = out_dir / "financial_latest" / f"{as_of}.csv"
    _write_csv(latest, out_path)

    if failures:
        fail_path = out_dir / "financial_latest" / f"{as_of}_failures.csv"
        _write_csv(pd.DataFrame(failures), fail_path)
    return {
        "as_of": as_of,
        "codes_requested": len(codes),
        "financial_rows": int(len(latest)),
        "financial_failures": len(failures),
        "financial_latest_path": str(out_path),
    }


def fetch_one_adj_factor(
    client: TushareClient,
    ts_code: str,
    *,
    start: str,
    end: str,
    raw_dir: Path,
    refresh: bool,
) -> dict[str, object]:
    symbol = _symbol_from_ts_code(ts_code)
    path = raw_dir / f"{symbol}.csv"
    if path.exists() and not refresh:
        cached = pd.read_csv(path, dtype={"ts_code": str, "trade_date": str})
        cached_dates = cached.get("trade_date", pd.Series(dtype=str)).astype(str)
        if (
            not cached.empty
            and str(cached_dates.min()) <= start
            and str(cached_dates.max()) >= end
        ):
            return {"ts_code": ts_code, "symbol": symbol, "rows": int(len(cached)), "cached": True}

    df = client.call(
        "adj_factor",
        ts_code=ts_code,
        start_date=start,
        end_date=end,
        fields=ADJ_FACTOR_FIELDS,
    )
    _write_csv(df, path, ADJ_FACTOR_FIELDS.split(","))
    return {"ts_code": ts_code, "symbol": symbol, "rows": int(len(df)), "cached": False}


def fetch_adj_factors_by_code(
    client: TushareClient,
    *,
    start: str,
    end: str,
    out_dir: Path,
    stocklist: Path,
    workers: int,
    limit_codes: int,
    refresh: bool,
) -> dict[str, object]:
    codes = _read_codes(stocklist, limit_codes=limit_codes)
    raw_dir = out_dir / "adj_factor_by_code"
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                fetch_one_adj_factor,
                client,
                ts_code,
                start=start,
                end=end,
                raw_dir=raw_dir,
                refresh=refresh,
            ): ts_code
            for ts_code in codes
        }
        for idx, future in enumerate(as_completed(futures), 1):
            ts_code = futures[future]
            try:
                row = future.result()
                rows.append(row)
                print(f"[adj_factor {idx}/{len(codes)}] {ts_code} rows={row['rows']}")
            except Exception as exc:
                failures.append({"ts_code": ts_code, "error": str(exc)})
                print(f"[adj_factor {idx}/{len(codes)}] failed {ts_code}: {exc}")

    summary_path = out_dir / "adj_factor_by_code_summary" / f"{end}.csv"
    _write_csv(pd.DataFrame(rows), summary_path)
    if failures:
        fail_path = out_dir / "adj_factor_by_code_summary" / f"{end}_failures.csv"
        _write_csv(pd.DataFrame(failures), fail_path)
    return {
        "start": start,
        "end": end,
        "codes_requested": len(codes),
        "adj_factor_files": len(rows),
        "adj_factor_failures": len(failures),
        "adj_factor_dir": str(raw_dir),
        "adj_factor_summary_path": str(summary_path),
    }


def parse_tasks(value: str) -> set[str]:
    tasks = {part.strip().lower() for part in value.split(",") if part.strip()}
    if "all" in tasks:
        return {"daily", "limit", "financial"}
    valid = {"daily", "limit", "financial", "adj_factor"}
    invalid = tasks - valid
    if invalid:
        raise ValueError(f"Invalid task(s): {', '.join(sorted(invalid))}")
    return tasks or {"daily"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch StockTradebyZ sidecar data into data/pit_metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tasks", default="daily", help="daily,limit,financial,adj_factor,all")
    parser.add_argument("--as-of", default="today", help="YYYYMMDD, YYYY-MM-DD, or today")
    parser.add_argument("--start", default="19900101", help="YYYYMMDD start date for adj_factor")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--stocklist", default=str(DEFAULT_STOCKLIST))
    parser.add_argument("--incremental", action="store_true", default=True)
    parser.add_argument("--no-incremental", dest="incremental", action="store_false")
    parser.add_argument("--refresh-financial", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-codes", type=int, default=0, help="debug helper for financial/adj_factor tasks")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-wait", type=float, default=0.8)
    args = parser.parse_args()

    tasks = parse_tasks(args.tasks)
    out_dir = _resolve_project_path(args.out)
    stocklist = _resolve_project_path(args.stocklist)
    client = TushareClient(retries=args.retries, retry_wait=args.retry_wait)
    as_of = _normalize_date(args.as_of)
    start = _normalize_date(args.start)
    trade_date = find_trade_date(client, as_of)

    started = time.perf_counter()
    summary: dict[str, object] = {
        "as_of": as_of,
        "trade_date": trade_date,
        "tasks": sorted(tasks),
        "out_dir": str(out_dir),
    }

    if "daily" in tasks:
        summary["daily"] = fetch_daily_sidecars(
            client,
            trade_date=trade_date,
            out_dir=out_dir,
            incremental=args.incremental,
        )

    if "limit" in tasks:
        summary["limit"] = fetch_limit_up_ytd(
            client,
            trade_date=trade_date,
            out_dir=out_dir,
            incremental=args.incremental,
        )

    if "financial" in tasks:
        summary["financial"] = fetch_financial_latest(
            client,
            as_of=trade_date,
            out_dir=out_dir,
            stocklist=stocklist,
            workers=args.workers,
            limit_codes=args.limit_codes,
            refresh=args.refresh_financial,
        )

    if "adj_factor" in tasks:
        summary["adj_factor"] = fetch_adj_factors_by_code(
            client,
            start=start,
            end=trade_date,
            out_dir=out_dir,
            stocklist=stocklist,
            workers=args.workers,
            limit_codes=args.limit_codes,
            refresh=not args.incremental,
        )

    summary["seconds"] = round(time.perf_counter() - started, 3)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_suffix = "_".join(sorted(tasks))
    summary_path = out_dir / f"sidecar_fetch_summary_{trade_date}_{task_suffix}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved sidecar summary: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
