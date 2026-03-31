"""Autonomous Equity Research Pipeline – Weekly Scheduler.

This script acts as the orchestration layer for the DCF Engine.  Each time it
runs it:

1. Loads (or auto-creates) a ticker universe from ``universe.csv``.
2. Selects the ticker that has not been analysed most recently.
3. Calls :func:`valuation.report_generator.run_scenario_analysis` to produce
   a DCF valuation workbook.
4. Files the report under ``reports/{ticker}/{ticker}_DCF_Report_{date}.xlsx``.
5. Updates ``last_analyzed`` in ``universe.csv`` so the same ticker is not
   re-selected next week.

If the valuation engine raises any exception the CSV is **not** updated, so the
same ticker will be retried on the next run.

Usage::

    python scheduler.py

Environment variable ``FMP_API_KEY`` must be set (or passed via the
``api_key`` argument inside the code).
"""

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

_UNIVERSE_FILE = "universe.csv"
_REPORTS_DIR = "reports"
_DEFAULT_TICKERS: list[str] = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
]


# ---------------------------------------------------------------------------
# Universe helpers
# ---------------------------------------------------------------------------


def _load_or_create_universe(path: str = _UNIVERSE_FILE) -> pd.DataFrame:
    """Load the ticker universe CSV, or create it with defaults.

    The CSV has two columns:

    * ``ticker``        – equity symbol (str)
    * ``last_analyzed`` – ISO-8601 date of the most recent analysis, or blank

    Parameters
    ----------
    path:
        Path to the CSV file.  Defaults to ``universe.csv`` in the current
        working directory.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``["ticker", "last_analyzed"]``.
    """
    if os.path.exists(path):
        logger.info("Loading universe from '%s'", path)
        df = pd.read_csv(path, dtype={"ticker": str, "last_analyzed": str})
        # Normalise column names (strip whitespace, lower-case)
        df.columns = [c.strip().lower() for c in df.columns]
        if "ticker" not in df.columns:
            raise ValueError(
                f"'{path}' must contain a 'ticker' column; "
                f"found: {list(df.columns)}"
            )
        if "last_analyzed" not in df.columns:
            df["last_analyzed"] = ""
        # Replace NaN / "nan" strings with empty string for uniform handling
        df["last_analyzed"] = df["last_analyzed"].fillna("").replace("nan", "")
        return df

    logger.warning(
        "'%s' not found – creating default universe with %d tickers.",
        path,
        len(_DEFAULT_TICKERS),
    )
    df = pd.DataFrame(
        {"ticker": _DEFAULT_TICKERS, "last_analyzed": [""] * len(_DEFAULT_TICKERS)}
    )
    df.to_csv(path, index=False)
    logger.info("Default universe saved to '%s'", path)
    return df


def _select_target(df: pd.DataFrame) -> str:
    """Return the ticker with the oldest (or missing) ``last_analyzed`` date.

    Blank / null dates are treated as the oldest possible date so they are
    always prioritised over tickers that have been analysed at least once.

    Parameters
    ----------
    df:
        DataFrame with columns ``["ticker", "last_analyzed"]``.

    Returns
    -------
    str
        Ticker symbol chosen for this run.
    """

    def _parse_date(val: str) -> date:
        """Parse an ISO-8601 date string; return ``date.min`` for blanks."""
        if not val or not val.strip():
            return date.min
        try:
            return datetime.strptime(val.strip(), "%Y-%m-%d").date()
        except ValueError:
            logger.warning("Could not parse date '%s'; treating as never analysed.", val)
            return date.min

    df = df.copy()
    df["_parsed_date"] = df["last_analyzed"].apply(_parse_date)
    idx = df["_parsed_date"].idxmin()
    target = df.loc[idx, "ticker"]
    logger.info(
        "Target ticker selected: %s  (last_analyzed: '%s')",
        target,
        df.loc[idx, "last_analyzed"],
    )
    return str(target)


def _file_report(src_path: str, ticker: str, run_date: date) -> str:
    """Copy the generated Excel file into the ``reports/{ticker}/`` folder.

    Parameters
    ----------
    src_path:
        Absolute path to the Excel file produced by
        :func:`~valuation.report_generator.run_scenario_analysis`.
    ticker:
        The ticker symbol (used for folder and file naming).
    run_date:
        The date to embed in the destination file name.

    Returns
    -------
    str
        Absolute path of the filed report.
    """
    dest_dir = os.path.join(_REPORTS_DIR, ticker)
    os.makedirs(dest_dir, exist_ok=True)
    dest_filename = f"{ticker}_DCF_Report_{run_date.isoformat()}.xlsx"
    dest_path = os.path.join(dest_dir, dest_filename)
    shutil.copy2(src_path, dest_path)
    logger.info("Report filed to '%s'", dest_path)
    return dest_path


def _update_universe(
    df: pd.DataFrame,
    ticker: str,
    run_date: date,
    path: str = _UNIVERSE_FILE,
) -> None:
    """Write today's date for *ticker* back to the universe CSV.

    Parameters
    ----------
    df:
        The in-memory universe DataFrame.
    ticker:
        The ticker that was successfully analysed.
    run_date:
        The date to record.
    path:
        Destination CSV path.
    """
    mask = df["ticker"] == ticker
    df.loc[mask, "last_analyzed"] = run_date.isoformat()
    df.to_csv(path, index=False)
    logger.info(
        "Universe updated: %s  →  last_analyzed = %s",
        ticker,
        run_date.isoformat(),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(universe_path: str = _UNIVERSE_FILE) -> None:
    """Run one iteration of the weekly equity research pipeline.

    Parameters
    ----------
    universe_path:
        Path to the universe CSV file.  Defaults to ``universe.csv``.
    """
    logger.info("=" * 60)
    logger.info("DCF Engine – Autonomous Weekly Scheduler starting")
    logger.info("=" * 60)

    # 1. Load / create the universe
    df = _load_or_create_universe(universe_path)

    # 2. Select the target ticker
    target = _select_target(df)
    run_date = date.today()

    # 3. Run the valuation engine (fail-safe: no CSV update on error)
    try:
        logger.info("Running scenario analysis for '%s' …", target)
        src_path = run_scenario_analysis(target)
        logger.info("Scenario analysis complete.  Excel output: '%s'", src_path)
    except Exception as exc:
        logger.critical(
            "Valuation engine failed for '%s': %s – universe.csv NOT updated; "
            "this ticker will be retried next run.",
            target,
            exc,
            exc_info=True,
        )
        sys.exit(1)

    # 4. File the report into reports/{ticker}/
    _file_report(src_path, target, run_date)

    # 5. Update the universe CSV with today's date
    _update_universe(df, target, run_date, universe_path)

    logger.info("=" * 60)
    logger.info("Weekly run complete.  Target: %s", target)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
