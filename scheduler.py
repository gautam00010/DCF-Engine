"""Autonomous Equity Research Pipeline – Weekly Scheduler."""

from __future__ import annotations

import logging
import os
import shutil
import sys
from datetime import date, datetime

import pandas as pd

from valuation.report_generator import run_scenario_analysis

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USA_UNIVERSE_FILE = "universe_usa.csv"
_INDIA_UNIVERSE_FILE = "universe_india.csv"
_REPORTS_DIR = "reports"

_DEFAULT_USA_TICKERS: list[str] = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]
_DEFAULT_INDIA_TICKERS: list[str] = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

def _load_or_create_universe(path: str, default_tickers: list[str]) -> pd.DataFrame:
    if os.path.exists(path):
        logger.info("Loading universe from '%s'", path)
        df = pd.read_csv(path, dtype={"ticker": str, "last_analyzed": str})
        df.columns = [c.strip().lower() for c in df.columns]
        if "ticker" not in df.columns:
            raise ValueError(f"'{path}' must contain a 'ticker' column.")
        if "last_analyzed" not in df.columns:
            df["last_analyzed"] = ""
        df["last_analyzed"] = df["last_analyzed"].fillna("").replace("nan", "")
        return df

    logger.warning("'%s' not found – creating default universe.", path)
    df = pd.DataFrame({
        "ticker": default_tickers,
        "last_analyzed": [""] * len(default_tickers),
    })
    df.to_csv(path, index=False)
    return df


def _select_target(df: pd.DataFrame) -> str | None:
    def _parse_date(val: str) -> date:
        if not val or not val.strip():
            return date.min
        try:
            return datetime.strptime(val.strip(), "%Y-%m-%d").date()
        except ValueError:
            return date.min

    if df.empty:
        return None

    df = df.copy()
    df["_parsed_date"] = df["last_analyzed"].apply(_parse_date)
    idx = df["_parsed_date"].idxmin()
    target = df.loc[idx, "ticker"]
    return str(target)


def _file_report(src_path: str, ticker: str, region: str, run_date: date) -> str:
    dest_dir = os.path.join(_REPORTS_DIR, region, ticker)
    os.makedirs(dest_dir, exist_ok=True)
    dest_filename = f"{ticker}_DCF_Report_{run_date.isoformat()}.xlsx"
    dest_path = os.path.join(dest_dir, dest_filename)
    shutil.copy2(src_path, dest_path)
    logger.info("Report filed to '%s'", dest_path)
    return dest_path


def _update_universe(df: pd.DataFrame, ticker: str, run_date: date, path: str) -> None:
    mask = df["ticker"] == ticker
    df.loc[mask, "last_analyzed"] = run_date.isoformat()
    df.to_csv(path, index=False)


def main() -> None:
    logger.info("=" * 60)
    logger.info("DCF Engine – Autonomous Global Weekly Scheduler starting")
    logger.info("=" * 60)

    run_date = date.today()
    
    universes = [
        (_INDIA_UNIVERSE_FILE, _DEFAULT_INDIA_TICKERS, "INDIA"),
        (_USA_UNIVERSE_FILE, _DEFAULT_USA_TICKERS, "USA")
    ]

    success_count = 0

    for path, defaults, region in universes:
        logger.info(f"--- Processing Region: {region} ---")
        try:
            df = _load_or_create_universe(path, defaults)
            target = _select_target(df)
            
            if not target:
                continue
                
            logger.info("INFO     Running scenario analysis for '%s' …", target)
            src_path = run_scenario_analysis(target)
            _file_report(src_path, target, region, run_date)
            _update_universe(df, target, run_date, path)
            success_count += 1
            
        except Exception as exc:
            logger.critical(f"Valuation failed for '{target}' in {region}: {exc}")

    logger.info("=" * 60)
    if success_count == 0:
        logger.critical("All regional valuations failed this run.")
        sys.exit(1)
    else:
        logger.info(f"Weekly run complete. Successfully processed {success_count} region(s).")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

