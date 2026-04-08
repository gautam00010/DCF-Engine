"""Autonomous Equity Research Pipeline – Weekly Scheduler.

This script acts as the orchestration layer for the DCF Engine.  Each time it
runs it:

1. Loads (or auto-creates) two ticker universes: ``universe_usa.csv`` and
   ``universe_india.csv``.
2. Selects the stalest ticker from each universe independently.
3. Calls :func:`valuation.report_generator.run_scenario_analysis` for both
   tickers; if one fails the other is still attempted.
4. Files reports under ``reports/USA/{ticker}/`` or ``reports/INDIA/{ticker}/``
   respectively.
5. Updates ``last_analyzed`` in each universe CSV only on success.

Usage::

    python scheduler.py

Environment variable ``FMP_API_KEY`` must be set for US stock valuations.
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

_USA_UNIVERSE_FILE = "universe_usa.csv"
_INDIA_UNIVERSE_FILE = "universe_india.csv"
_REPORTS_DIR = "reports"

_DEFAULT_USA_TICKERS: list[str] = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]
_DEFAULT_INDIA_TICKERS: list[str] = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
]

# Backward-compatibility alias
_DEFAULT_TICKERS: list[str] = _DEFAULT_INDIA_TICKERS


# ---------------------------------------------------------------------------
# Universe helpers
# ---------------------------------------------------------------------------


def _load_or_create_universe(
    path: str, default_tickers: list[str] | None = None
) -> pd.DataFrame:
    """Load the ticker universe CSV, or create it with defaults.

    The CSV has two columns:

    * ``ticker``        – equity symbol (str)
    * ``last_analyzed`` – ISO-8601 date of the most recent analysis, or blank

    Parameters
    ----------
    path:
        Path to the CSV file.
    default_tickers:
        Tickers to use when creating a new file.  Defaults to
        ``_DEFAULT_INDIA_TICKERS`` when ``None``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``["ticker", "last_analyzed"]``.
    """
    if default_tickers is None:
        default_tickers = _DEFAULT_INDIA_TICKERS

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
        len(default_tickers),
    )
    df = pd.DataFrame(
        {
            "ticker": default_tickers,
            "last_analyzed": [""] * len(default_tickers),
        }
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


def _is_indian_ticker(ticker: str) -> bool:
    """Return ``True`` if *ticker* is listed on an Indian exchange."""
    upper = ticker.upper()
    return upper.endswith(".NS") or upper.endswith(".BO")


def _file_report(src_path: str, ticker: str, run_date: date) -> str:
    """Copy the generated Excel file into the appropriate regional folder.

    US tickers are filed under ``reports/USA/{ticker}/``; Indian tickers
    (``*.NS`` / ``*.BO``) under ``reports/INDIA/{ticker}/``.

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
    region = "INDIA" if _is_indian_ticker(ticker) else "USA"
    dest_dir = os.path.join(_REPORTS_DIR, region, ticker)
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
    path: str,
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


def main(
    usa_universe_path: str = _USA_UNIVERSE_FILE,
    india_universe_path: str = _INDIA_UNIVERSE_FILE,
) -> None:
    """Run one iteration of the weekly equity research pipeline.

    Selects the stalest ticker from the USA universe and the stalest ticker
    from the India universe, then runs both valuations independently.  If one
    valuation fails the other is still attempted.  Exits with code 1 only if
    both valuations fail.

    Parameters
    ----------
    usa_universe_path:
        Path to the US ticker universe CSV.  Defaults to ``universe_usa.csv``.
    india_universe_path:
        Path to the India ticker universe CSV.  Defaults to
        ``universe_india.csv``.
    """
    logger.info("=" * 60)
    logger.info("DCF Engine – Autonomous Weekly Scheduler starting")
    logger.info("=" * 60)

    run_date = date.today()
    usa_success = False
    india_success = False

    # ------------------------------------------------------------------
    # USA valuation
    # ------------------------------------------------------------------
    df_usa = _load_or_create_universe(usa_universe_path, _DEFAULT_USA_TICKERS)
    usa_target = _select_target(df_usa)

    try:
        logger.info("INFO     Running scenario analysis for '%s' …", usa_target)
        src_path = run_scenario_analysis(usa_target)
        logger.info("INFO     Scenario analysis complete.  Excel output: '%s'", src_path)
        _file_report(src_path, usa_target, run_date)
        _update_universe(df_usa, usa_target, run_date, usa_universe_path)
        usa_success = True
    except Exception as exc:
        logger.critical(
            "Valuation engine failed for '%s': %s – %s NOT updated; "
            "this ticker will be retried next run.",
            usa_target,
            exc,
            usa_universe_path,
            exc_info=True,
        )

    # ------------------------------------------------------------------
    # India valuation
    # ------------------------------------------------------------------
    df_india = _load_or_create_universe(india_universe_path, _DEFAULT_INDIA_TICKERS)
    india_target = _select_target(df_india)

    try:
        logger.info("INFO     Running scenario analysis for '%s' …", india_target)
        src_path = run_scenario_analysis(india_target)
        logger.info("INFO     Scenario analysis complete.  Excel output: '%s'", src_path)
        _file_report(src_path, india_target, run_date)
        _update_universe(df_india, india_target, run_date, india_universe_path)
        india_success = True
    except Exception as exc:
        logger.critical(
            "Valuation engine failed for '%s': %s – %s NOT updated; "
            "this ticker will be retried next run.",
            india_target,
            exc,
            india_universe_path,
            exc_info=True,
        )

    # ------------------------------------------------------------------
    # Final status
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    if usa_success or india_success:
        logger.info(
            "Weekly run complete.  USA: %s  |  India: %s",
            usa_target if usa_success else "FAILED",
            india_target if india_success else "FAILED",
        )
    else:
        logger.critical("Both USA and India valuations failed this run.")
        sys.exit(1)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

