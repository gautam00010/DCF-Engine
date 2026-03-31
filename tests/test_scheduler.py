"""Unit tests for scheduler.py.

All tests are fully offline – no real HTTP calls, no real file-system
side-effects outside of temporary directories managed by the test harness.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(**rows) -> pd.DataFrame:
    """Build a minimal universe DataFrame from keyword args.

    Usage::

        _make_df(AAPL="2024-01-01", TSLA="", MSFT="2023-06-15")
    """
    tickers = list(rows.keys())
    dates = list(rows.values())
    return pd.DataFrame({"ticker": tickers, "last_analyzed": dates})


# ---------------------------------------------------------------------------
# _load_or_create_universe
# ---------------------------------------------------------------------------

from scheduler import (
    _DEFAULT_INDIA_TICKERS,
    _DEFAULT_TICKERS,
    _DEFAULT_USA_TICKERS,
    _file_report,
    _load_or_create_universe,
    _select_target,
    _update_universe,
    main,
)


class TestLoadOrCreateUniverse(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _path(self, filename: str = "universe.csv") -> str:
        return os.path.join(self.tmpdir, filename)

    # --- File does not exist → auto-create ---

    def test_creates_file_when_missing(self) -> None:
        path = self._path()
        _load_or_create_universe(path)
        self.assertTrue(os.path.isfile(path))

    def test_created_file_has_default_tickers(self) -> None:
        path = self._path()
        df = _load_or_create_universe(path)
        self.assertEqual(list(df["ticker"]), _DEFAULT_TICKERS)

    def test_created_file_has_blank_dates(self) -> None:
        path = self._path()
        df = _load_or_create_universe(path)
        self.assertTrue((df["last_analyzed"] == "").all())

    def test_created_file_has_two_columns(self) -> None:
        path = self._path()
        df = _load_or_create_universe(path)
        self.assertEqual(list(df.columns), ["ticker", "last_analyzed"])

    # --- File exists → load it ---

    def test_loads_existing_csv(self) -> None:
        path = self._path()
        pd.DataFrame(
            {"ticker": ["X", "Y"], "last_analyzed": ["2024-01-01", ""]}
        ).to_csv(path, index=False)
        df = _load_or_create_universe(path)
        self.assertEqual(list(df["ticker"]), ["X", "Y"])

    def test_nan_dates_normalised_to_empty_string(self) -> None:
        path = self._path()
        # Write a CSV that has an empty cell (pandas reads this as NaN)
        with open(path, "w") as f:
            f.write("ticker,last_analyzed\nAAPL,\nTSLA,2024-01-01\n")
        df = _load_or_create_universe(path)
        self.assertEqual(df.loc[df["ticker"] == "AAPL", "last_analyzed"].iloc[0], "")

    def test_missing_last_analyzed_column_added(self) -> None:
        path = self._path()
        pd.DataFrame({"ticker": ["A", "B"]}).to_csv(path, index=False)
        df = _load_or_create_universe(path)
        self.assertIn("last_analyzed", df.columns)

    def test_raises_if_no_ticker_column(self) -> None:
        path = self._path()
        pd.DataFrame({"symbol": ["X"]}).to_csv(path, index=False)
        with self.assertRaises(ValueError):
            _load_or_create_universe(path)


# ---------------------------------------------------------------------------
# _select_target
# ---------------------------------------------------------------------------


class TestSelectTarget(unittest.TestCase):
    def test_blank_date_selected_over_dated(self) -> None:
        df = _make_df(AAPL="2024-01-01", TSLA="")
        self.assertEqual(_select_target(df), "TSLA")

    def test_oldest_date_selected(self) -> None:
        df = _make_df(AAPL="2024-06-01", TSLA="2023-01-01", MSFT="2024-01-01")
        self.assertEqual(_select_target(df), "TSLA")

    def test_all_blank_selects_first(self) -> None:
        df = _make_df(AAPL="", TSLA="", MSFT="")
        # All tie at date.min; idxmin returns the first occurrence
        self.assertEqual(_select_target(df), "AAPL")

    def test_single_ticker(self) -> None:
        df = _make_df(AAPL="2024-01-01")
        self.assertEqual(_select_target(df), "AAPL")

    def test_all_same_date_selects_first(self) -> None:
        df = _make_df(A="2024-01-01", B="2024-01-01", C="2024-01-01")
        self.assertEqual(_select_target(df), "A")

    def test_unparseable_date_treated_as_oldest(self) -> None:
        df = _make_df(AAPL="not-a-date", TSLA="2024-01-01")
        # "not-a-date" falls back to date.min, which is older than TSLA's date
        self.assertEqual(_select_target(df), "AAPL")

    def test_returns_string(self) -> None:
        df = _make_df(AAPL="2024-01-01")
        result = _select_target(df)
        self.assertIsInstance(result, str)

    def test_original_df_not_mutated(self) -> None:
        df = _make_df(AAPL="2024-01-01", TSLA="")
        original_cols = list(df.columns)
        _select_target(df)
        self.assertEqual(list(df.columns), original_cols)


# ---------------------------------------------------------------------------
# _file_report
# ---------------------------------------------------------------------------


class TestFileReport(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        # Create a fake Excel source file
        self.src = os.path.join(self.tmpdir, "source.xlsx")
        with open(self.src, "wb") as f:
            f.write(b"fake-excel-content")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run(self, ticker: str = "AAPL", run_date: date = date(2024, 7, 1)) -> str:
        # Patch _REPORTS_DIR to use the temp dir
        with patch("scheduler._REPORTS_DIR", self.tmpdir):
            return _file_report(self.src, ticker, run_date)

    def test_dest_file_created(self) -> None:
        dest = self._run()
        self.assertTrue(os.path.isfile(dest))

    def test_dest_content_matches_source(self) -> None:
        dest = self._run()
        with open(dest, "rb") as f:
            self.assertEqual(f.read(), b"fake-excel-content")

    def test_dest_filename_follows_convention(self) -> None:
        dest = self._run(ticker="TSLA", run_date=date(2024, 8, 15))
        self.assertEqual(os.path.basename(dest), "TSLA_DCF_Report_2024-08-15.xlsx")

    def test_dest_directory_created_if_missing(self) -> None:
        dest = self._run(ticker="NEWCO")
        self.assertTrue(os.path.isdir(os.path.dirname(dest)))

    def test_returns_string_path(self) -> None:
        result = self._run()
        self.assertIsInstance(result, str)

    def test_ticker_subfolder_used(self) -> None:
        ticker = "HDFCBANK.NS"
        dest = self._run(ticker=ticker)
        self.assertIn(ticker, dest)


# ---------------------------------------------------------------------------
# _update_universe
# ---------------------------------------------------------------------------


class TestUpdateUniverse(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _path(self) -> str:
        return os.path.join(self.tmpdir, "universe.csv")

    def test_updates_correct_ticker(self) -> None:
        df = _make_df(AAPL="", TSLA="2024-01-01")
        path = self._path()
        _update_universe(df, "AAPL", date(2024, 7, 1), path)
        saved = pd.read_csv(path)
        self.assertEqual(
            saved.loc[saved["ticker"] == "AAPL", "last_analyzed"].iloc[0],
            "2024-07-01",
        )

    def test_other_tickers_unchanged(self) -> None:
        df = _make_df(AAPL="", TSLA="2024-01-01")
        path = self._path()
        _update_universe(df, "AAPL", date(2024, 7, 1), path)
        saved = pd.read_csv(path)
        self.assertEqual(
            saved.loc[saved["ticker"] == "TSLA", "last_analyzed"].iloc[0],
            "2024-01-01",
        )

    def test_file_is_overwritten(self) -> None:
        df = _make_df(AAPL="")
        path = self._path()
        _update_universe(df, "AAPL", date(2024, 7, 1), path)
        self.assertTrue(os.path.isfile(path))

    def test_csv_has_no_index_column(self) -> None:
        df = _make_df(AAPL="")
        path = self._path()
        _update_universe(df, "AAPL", date(2024, 7, 1), path)
        saved = pd.read_csv(path)
        # If index was written, there would be an 'Unnamed: 0' column
        self.assertNotIn("Unnamed: 0", saved.columns)

    def test_in_memory_df_also_updated(self) -> None:
        df = _make_df(AAPL="")
        _update_universe(df, "AAPL", date(2024, 7, 1), self._path())
        self.assertEqual(
            df.loc[df["ticker"] == "AAPL", "last_analyzed"].iloc[0], "2024-07-01"
        )


# ---------------------------------------------------------------------------
# main() – integration (fully mocked)
# ---------------------------------------------------------------------------


class TestMain(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.universe_path = os.path.join(self.tmpdir, "universe_usa.csv")
        self.india_universe_path = os.path.join(self.tmpdir, "universe_india.csv")
        # Create a fake Excel file that run_scenario_analysis "returns"
        self.fake_excel = os.path.join(self.tmpdir, "fake_DCF_Valuation.xlsx")
        with open(self.fake_excel, "wb") as f:
            f.write(b"fake")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_main(self, mock_run_scenario) -> None:
        """Call main() with all filesystem side-effects redirected to tmpdir."""
        with (
            patch("scheduler._REPORTS_DIR", os.path.join(self.tmpdir, "reports")),
            patch(
                "scheduler.run_scenario_analysis",
                mock_run_scenario,
            ),
        ):
            main(self.universe_path, self.india_universe_path)

    def test_main_creates_universe_if_missing(self) -> None:
        mock_rsa = MagicMock(return_value=self.fake_excel)
        self._run_main(mock_rsa)
        self.assertTrue(os.path.isfile(self.universe_path))

    def test_main_calls_run_scenario_analysis(self) -> None:
        """run_scenario_analysis is called once for USA and once for India."""
        mock_rsa = MagicMock(return_value=self.fake_excel)
        self._run_main(mock_rsa)
        self.assertEqual(mock_rsa.call_count, 2)

    def test_main_updates_universe_csv_on_success(self) -> None:
        mock_rsa = MagicMock(return_value=self.fake_excel)
        self._run_main(mock_rsa)
        df = pd.read_csv(self.universe_path)
        today = date.today().isoformat()
        # The first (oldest/blank) USA ticker should now have today's date
        first_ticker_date = df.iloc[0]["last_analyzed"]
        self.assertEqual(first_ticker_date, today)

    def test_main_does_not_update_csv_on_failure(self) -> None:
        # Pre-create the USA universe so we can check it wasn't modified
        pd.DataFrame(
            {"ticker": ["A", "B"], "last_analyzed": ["", ""]}
        ).to_csv(self.universe_path, index=False)

        # Both USA and India fail → sys.exit(1)
        mock_rsa = MagicMock(side_effect=RuntimeError("API timeout"))
        with self.assertRaises(SystemExit):
            self._run_main(mock_rsa)

        df = pd.read_csv(self.universe_path)
        # Dates should still be blank / NaN
        self.assertTrue(df["last_analyzed"].isna().all() or (df["last_analyzed"] == "").all())

    def test_main_files_usa_report_to_usa_folder(self) -> None:
        mock_rsa = MagicMock(return_value=self.fake_excel)
        reports_dir = os.path.join(self.tmpdir, "reports")
        with (
            patch("scheduler._REPORTS_DIR", reports_dir),
            patch("scheduler.run_scenario_analysis", mock_rsa),
        ):
            main(self.universe_path, self.india_universe_path)

        # First USA default ticker is AAPL → reports/USA/AAPL/
        target_dir = os.path.join(reports_dir, "USA", "AAPL")
        self.assertTrue(os.path.isdir(target_dir))
        xlsx_files = [f for f in os.listdir(target_dir) if f.endswith(".xlsx")]
        self.assertTrue(len(xlsx_files) > 0)

    def test_main_files_india_report_to_india_folder(self) -> None:
        mock_rsa = MagicMock(return_value=self.fake_excel)
        reports_dir = os.path.join(self.tmpdir, "reports")
        with (
            patch("scheduler._REPORTS_DIR", reports_dir),
            patch("scheduler.run_scenario_analysis", mock_rsa),
        ):
            main(self.universe_path, self.india_universe_path)

        # First India default ticker is RELIANCE.NS → reports/INDIA/RELIANCE.NS/
        target_dir = os.path.join(reports_dir, "INDIA", "RELIANCE.NS")
        self.assertTrue(os.path.isdir(target_dir))
        xlsx_files = [f for f in os.listdir(target_dir) if f.endswith(".xlsx")]
        self.assertTrue(len(xlsx_files) > 0)

    def test_main_report_filename_convention(self) -> None:
        mock_rsa = MagicMock(return_value=self.fake_excel)
        reports_dir = os.path.join(self.tmpdir, "reports")
        with (
            patch("scheduler._REPORTS_DIR", reports_dir),
            patch("scheduler.run_scenario_analysis", mock_rsa),
        ):
            main(self.universe_path, self.india_universe_path)

        today = date.today().isoformat()
        expected_name = f"RELIANCE.NS_DCF_Report_{today}.xlsx"
        target_dir = os.path.join(reports_dir, "INDIA", "RELIANCE.NS")
        self.assertIn(expected_name, os.listdir(target_dir))

    def test_main_selects_oldest_ticker(self) -> None:
        """When a CSV exists, the ticker with the oldest date is chosen."""
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        old_date = "2020-01-01"
        pd.DataFrame(
            {
                "ticker": ["AAPL", "TSLA", "MSFT"],
                "last_analyzed": [yesterday, old_date, yesterday],
            }
        ).to_csv(self.universe_path, index=False)

        mock_rsa = MagicMock(return_value=self.fake_excel)
        reports_dir = os.path.join(self.tmpdir, "reports")
        with (
            patch("scheduler._REPORTS_DIR", reports_dir),
            patch("scheduler.run_scenario_analysis", mock_rsa),
        ):
            main(self.universe_path, self.india_universe_path)

        # TSLA should have been selected from the USA universe
        call_args = [str(call[0][0]) for call in mock_rsa.call_args_list]
        self.assertIn("TSLA", call_args)

    def test_main_exits_nonzero_on_engine_failure(self) -> None:
        """sys.exit(1) when both USA and India valuations fail."""
        mock_rsa = MagicMock(side_effect=ConnectionError("network down"))
        with self.assertRaises(SystemExit) as ctx:
            self._run_main(mock_rsa)
        self.assertNotEqual(ctx.exception.code, 0)

    def test_main_india_failure_does_not_block_usa(self) -> None:
        """If India fails but USA succeeds, no SystemExit is raised."""
        def _side_effect(ticker):
            from scheduler import _is_indian_ticker
            if _is_indian_ticker(ticker):
                raise RuntimeError("India API down")
            return self.fake_excel

        mock_rsa = MagicMock(side_effect=_side_effect)
        # Should not raise SystemExit – USA succeeds
        self._run_main(mock_rsa)
        # USA CSV should be updated
        df = pd.read_csv(self.universe_path)
        today = date.today().isoformat()
        self.assertEqual(df.iloc[0]["last_analyzed"], today)

    def test_main_usa_failure_does_not_block_india(self) -> None:
        """If USA fails but India succeeds, no SystemExit is raised."""
        def _side_effect(ticker):
            from scheduler import _is_indian_ticker
            if not _is_indian_ticker(ticker):
                raise RuntimeError("FMP API down")
            return self.fake_excel

        mock_rsa = MagicMock(side_effect=_side_effect)
        # Should not raise SystemExit – India succeeds
        self._run_main(mock_rsa)
        # India CSV should be updated
        df = pd.read_csv(self.india_universe_path)
        today = date.today().isoformat()
        self.assertEqual(df.iloc[0]["last_analyzed"], today)


if __name__ == "__main__":
    unittest.main()
