"""Unit tests for valuation.data_fetcher.

All tests are fully offline – no real HTTP calls are made.  The FMP API is
mocked via :mod:`unittest.mock`.  yfinance calls are also mocked.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from valuation.data_fetcher import (
    BaseDataFetcher,
    DataFetchError,
    FMPDataFetcher,
    FinancialDataFetcher,
    YFinanceDataFetcher,
    get_fetcher,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DUMMY_KEY = "test_api_key"
_TICKER = "AAPL"


def _make_fetcher(ticker: str = _TICKER) -> FinancialDataFetcher:
    return FinancialDataFetcher(ticker, api_key=_DUMMY_KEY)


def _mock_response(json_data, status_code: int = 200) -> MagicMock:
    """Return a mock :class:`requests.Response` object."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    if status_code >= 400:
        import requests

        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_resp
        )
    else:
        mock_resp.raise_for_status.return_value = None
    return mock_resp


# ---------------------------------------------------------------------------
# FinancialDataFetcher instantiation
# ---------------------------------------------------------------------------


class TestFinancialDataFetcherInit(unittest.TestCase):
    def test_init_with_explicit_key(self) -> None:
        fetcher = FinancialDataFetcher(_TICKER, api_key=_DUMMY_KEY)
        self.assertEqual(fetcher.ticker, _TICKER)

    def test_ticker_normalised_to_upper(self) -> None:
        fetcher = FinancialDataFetcher("aapl", api_key=_DUMMY_KEY)
        self.assertEqual(fetcher.ticker, "AAPL")

    def test_ticker_strips_whitespace(self) -> None:
        fetcher = FinancialDataFetcher("  MSFT  ", api_key=_DUMMY_KEY)
        self.assertEqual(fetcher.ticker, "MSFT")

    def test_init_reads_env_variable(self) -> None:
        with patch.dict("os.environ", {"FMP_API_KEY": "env_key"}):
            fetcher = FinancialDataFetcher(_TICKER)
        self.assertEqual(fetcher._api_key, "env_key")

    def test_init_raises_without_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            # Make sure FMP_API_KEY is not present
            import os

            os.environ.pop("FMP_API_KEY", None)
            with self.assertRaises(DataFetchError):
                FinancialDataFetcher(_TICKER)


# ---------------------------------------------------------------------------
# get_cash_flow_statement
# ---------------------------------------------------------------------------


class TestGetCashFlowStatement(unittest.TestCase):
    _SAMPLE = [
        {
            "date": "2023-09-30",
            "symbol": "AAPL",
            "freeCashFlow": 99584000000,
            "operatingCashFlow": 114301000000,
            "capitalExpenditure": -10959000000,
        }
    ]

    @patch("valuation.data_fetcher.requests.get")
    def test_returns_first_record(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response(self._SAMPLE)
        fetcher = _make_fetcher()
        result = fetcher.get_cash_flow_statement()
        self.assertEqual(result["freeCashFlow"], 99584000000)
        self.assertEqual(result["date"], "2023-09-30")

    @patch("valuation.data_fetcher.requests.get")
    def test_empty_list_raises(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response([])
        with self.assertRaises(DataFetchError):
            _make_fetcher().get_cash_flow_statement()

    @patch("valuation.data_fetcher.requests.get")
    def test_api_error_message_raises(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response({"Error Message": "Invalid ticker"})
        with self.assertRaises(DataFetchError):
            _make_fetcher().get_cash_flow_statement()

    @patch("valuation.data_fetcher.requests.get")
    def test_http_error_raises(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response({}, status_code=401)
        with self.assertRaises(DataFetchError):
            _make_fetcher().get_cash_flow_statement()

    @patch("valuation.data_fetcher.requests.get")
    def test_network_error_raises(self, mock_get: MagicMock) -> None:
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError("no connection")
        with self.assertRaises(DataFetchError):
            _make_fetcher().get_cash_flow_statement()

    @patch("valuation.data_fetcher.requests.get")
    def test_timeout_raises(self, mock_get: MagicMock) -> None:
        import requests

        mock_get.side_effect = requests.exceptions.Timeout("timed out")
        with self.assertRaises(DataFetchError):
            _make_fetcher().get_cash_flow_statement()


# ---------------------------------------------------------------------------
# get_enterprise_metrics
# ---------------------------------------------------------------------------


class TestGetEnterpriseMetrics(unittest.TestCase):
    _BS_SAMPLE = [
        {
            "date": "2023-09-30",
            "symbol": "AAPL",
            "totalDebt": 111088000000,
            "cashAndCashEquivalents": 29965000000,
        }
    ]
    _SHARES_SAMPLE = [
        {
            "symbol": "AAPL",
            "outstandingShares": 15634232000,
        }
    ]

    @patch("valuation.data_fetcher.requests.get")
    def test_returns_merged_dict(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = [
            _mock_response(self._BS_SAMPLE),
            _mock_response(self._SHARES_SAMPLE),
        ]
        result = _make_fetcher().get_enterprise_metrics()
        self.assertEqual(result["totalDebt"], 111088000000)
        self.assertEqual(result["cashAndCashEquivalents"], 29965000000)
        self.assertEqual(result["sharesOutstanding"], 15634232000)
        self.assertEqual(result["date"], "2023-09-30")

    @patch("valuation.data_fetcher.requests.get")
    def test_falls_back_to_longterm_plus_shortterm_debt(
        self, mock_get: MagicMock
    ) -> None:
        bs_no_totaldebt = [
            {
                "date": "2023-09-30",
                "symbol": "AAPL",
                "longTermDebt": 95000000000,
                "shortTermDebt": 10000000000,
                "cashAndCashEquivalents": 20000000000,
            }
        ]
        mock_get.side_effect = [
            _mock_response(bs_no_totaldebt),
            _mock_response(self._SHARES_SAMPLE),
        ]
        result = _make_fetcher().get_enterprise_metrics()
        self.assertEqual(result["totalDebt"], 105000000000)

    @patch("valuation.data_fetcher.requests.get")
    def test_zero_total_debt_not_overridden(self, mock_get: MagicMock) -> None:
        """totalDebt=0 must not trigger the long/short fallback."""
        bs_zero_debt = [
            {
                "date": "2023-09-30",
                "symbol": "AAPL",
                "totalDebt": 0,
                "longTermDebt": 50000000000,
                "shortTermDebt": 10000000000,
                "cashAndCashEquivalents": 30000000000,
            }
        ]
        mock_get.side_effect = [
            _mock_response(bs_zero_debt),
            _mock_response(self._SHARES_SAMPLE),
        ]
        result = _make_fetcher().get_enterprise_metrics()
        self.assertEqual(result["totalDebt"], 0)

    @patch("valuation.data_fetcher.requests.get")
    def test_empty_balance_sheet_raises(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response([])
        with self.assertRaises(DataFetchError):
            _make_fetcher().get_enterprise_metrics()


# ---------------------------------------------------------------------------
# get_wacc
# ---------------------------------------------------------------------------


class TestGetWacc(unittest.TestCase):
    _SAMPLE = [
        {
            "symbol": "AAPL",
            "wacc": 0.0912,
            "costOfEquity": 0.0985,
            "costOfDebt": 0.0305,
            "taxRate": 0.1607,
            "equityWeight": 0.926,
            "debtWeight": 0.074,
        }
    ]

    @patch("valuation.data_fetcher.requests.get")
    def test_returns_wacc_dict(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response(self._SAMPLE)
        result = _make_fetcher().get_wacc()
        self.assertAlmostEqual(result["wacc"], 0.0912)
        self.assertAlmostEqual(result["costOfEquity"], 0.0985)
        self.assertAlmostEqual(result["debtWeight"], 0.074)

    @patch("valuation.data_fetcher.requests.get")
    def test_missing_wacc_field_raises(self, mock_get: MagicMock) -> None:
        # Payload present but 'wacc' key absent
        mock_get.return_value = _mock_response([{"symbol": "AAPL"}])
        with self.assertRaises(DataFetchError):
            _make_fetcher().get_wacc()

    @patch("valuation.data_fetcher.requests.get")
    def test_empty_response_raises(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response([])
        with self.assertRaises(DataFetchError):
            _make_fetcher().get_wacc()

    @patch("valuation.data_fetcher.requests.get")
    def test_http_500_raises(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response({}, status_code=500)
        with self.assertRaises(DataFetchError):
            _make_fetcher().get_wacc()


# ---------------------------------------------------------------------------
# DataFetchError
# ---------------------------------------------------------------------------


class TestDataFetchError(unittest.TestCase):
    def test_is_exception_subclass(self) -> None:
        self.assertTrue(issubclass(DataFetchError, Exception))

    def test_can_be_raised_and_caught(self) -> None:
        with self.assertRaises(DataFetchError) as ctx:
            raise DataFetchError("something went wrong")
        self.assertIn("something went wrong", str(ctx.exception))


# ---------------------------------------------------------------------------
# BaseDataFetcher / class hierarchy
# ---------------------------------------------------------------------------


class TestBaseDataFetcher(unittest.TestCase):
    def test_fmp_is_subclass_of_base(self) -> None:
        self.assertTrue(issubclass(FMPDataFetcher, BaseDataFetcher))

    def test_yfinance_is_subclass_of_base(self) -> None:
        self.assertTrue(issubclass(YFinanceDataFetcher, BaseDataFetcher))

    def test_financial_data_fetcher_alias(self) -> None:
        """FinancialDataFetcher must remain an alias for FMPDataFetcher."""
        self.assertIs(FinancialDataFetcher, FMPDataFetcher)

    def test_base_is_abstract(self) -> None:
        """BaseDataFetcher cannot be instantiated directly."""
        import abc

        self.assertTrue(issubclass(BaseDataFetcher, abc.ABC))


# ---------------------------------------------------------------------------
# get_fetcher router
# ---------------------------------------------------------------------------


class TestGetFetcher(unittest.TestCase):
    def test_ns_ticker_returns_yfinance_fetcher(self) -> None:
        fetcher = get_fetcher("RELIANCE.NS")
        self.assertIsInstance(fetcher, YFinanceDataFetcher)

    def test_bo_ticker_returns_yfinance_fetcher(self) -> None:
        fetcher = get_fetcher("RELIANCE.BO")
        self.assertIsInstance(fetcher, YFinanceDataFetcher)

    def test_lowercase_ns_ticker_returns_yfinance_fetcher(self) -> None:
        fetcher = get_fetcher("tcs.ns")
        self.assertIsInstance(fetcher, YFinanceDataFetcher)

    def test_us_ticker_returns_fmp_fetcher(self) -> None:
        fetcher = get_fetcher("AAPL", api_key=_DUMMY_KEY)
        self.assertIsInstance(fetcher, FMPDataFetcher)

    def test_us_ticker_without_suffix_returns_fmp_fetcher(self) -> None:
        fetcher = get_fetcher("MSFT", api_key=_DUMMY_KEY)
        self.assertIsInstance(fetcher, FMPDataFetcher)

    def test_fmp_fetcher_has_api_key(self) -> None:
        fetcher = get_fetcher("AAPL", api_key=_DUMMY_KEY)
        self.assertIsInstance(fetcher, FMPDataFetcher)
        self.assertEqual(fetcher._api_key, _DUMMY_KEY)

    def test_yfinance_ticker_normalised_to_upper(self) -> None:
        fetcher = get_fetcher("reliance.ns")
        self.assertEqual(fetcher.ticker, "RELIANCE.NS")


# ---------------------------------------------------------------------------
# YFinanceDataFetcher
# ---------------------------------------------------------------------------


def _make_yf_fetcher(ticker: str = "RELIANCE.NS") -> YFinanceDataFetcher:
    return YFinanceDataFetcher(ticker)


def _make_mock_cf_df(
    free_cf: float = 1_000_000_000,
    op_cf: float = 1_200_000_000,
    capex: float = -200_000_000,
) -> MagicMock:
    """Build a mock yfinance cash_flow DataFrame with one column."""
    import pandas as pd

    date_idx = pd.Timestamp("2023-03-31")
    data = {
        date_idx: {
            "Free Cash Flow": free_cf,
            "Operating Cash Flow": op_cf,
            "Capital Expenditure": capex,
        }
    }
    return pd.DataFrame(data)


class TestYFinanceDataFetcherInit(unittest.TestCase):
    def test_ticker_normalised_to_upper(self) -> None:
        fetcher = YFinanceDataFetcher("reliance.ns")
        self.assertEqual(fetcher.ticker, "RELIANCE.NS")

    def test_ticker_strips_whitespace(self) -> None:
        fetcher = YFinanceDataFetcher("  TCS.NS  ")
        self.assertEqual(fetcher.ticker, "TCS.NS")


class TestYFinanceGetCashFlowStatement(unittest.TestCase):
    def _fetcher_with_cf(self, cf_df) -> YFinanceDataFetcher:
        fetcher = _make_yf_fetcher()
        mock_stock = MagicMock()
        mock_stock.cash_flow = cf_df
        fetcher._stock = MagicMock(return_value=mock_stock)
        return fetcher

    def test_returns_correct_keys(self) -> None:
        fetcher = self._fetcher_with_cf(_make_mock_cf_df())
        result = fetcher.get_cash_flow_statement()
        for key in ("freeCashFlow", "operatingCashFlow", "capitalExpenditure", "date"):
            self.assertIn(key, result)

    def test_free_cash_flow_value(self) -> None:
        fetcher = self._fetcher_with_cf(_make_mock_cf_df(free_cf=5_000_000_000))
        result = fetcher.get_cash_flow_statement()
        self.assertAlmostEqual(result["freeCashFlow"], 5_000_000_000)

    def test_fcf_computed_from_op_minus_capex_when_missing(self) -> None:
        import pandas as pd

        date_idx = pd.Timestamp("2023-03-31")
        # No 'Free Cash Flow' row – should fall back to op + capex
        cf_df = pd.DataFrame(
            {date_idx: {"Operating Cash Flow": 1_200_000_000, "Capital Expenditure": -200_000_000}}
        )
        fetcher = self._fetcher_with_cf(cf_df)
        result = fetcher.get_cash_flow_statement()
        self.assertAlmostEqual(result["freeCashFlow"], 1_000_000_000)

    def test_empty_dataframe_raises(self) -> None:
        import pandas as pd

        fetcher = self._fetcher_with_cf(pd.DataFrame())
        with self.assertRaises(DataFetchError):
            fetcher.get_cash_flow_statement()

    def test_none_cash_flow_raises(self) -> None:
        fetcher = self._fetcher_with_cf(None)
        with self.assertRaises(DataFetchError):
            fetcher.get_cash_flow_statement()


class TestYFinanceGetEnterpriseMetrics(unittest.TestCase):
    _SAMPLE_INFO = {
        "totalDebt": 500_000_000,
        "totalCash": 200_000_000,
        "sharesOutstanding": 6_760_000_000,
    }

    def _fetcher_with_info(self, info: dict) -> YFinanceDataFetcher:
        fetcher = _make_yf_fetcher()
        mock_stock = MagicMock()
        mock_stock.info = info
        fetcher._stock = MagicMock(return_value=mock_stock)
        return fetcher

    def test_returns_correct_keys(self) -> None:
        fetcher = self._fetcher_with_info(self._SAMPLE_INFO)
        result = fetcher.get_enterprise_metrics()
        for key in ("totalDebt", "cashAndCashEquivalents", "sharesOutstanding", "date"):
            self.assertIn(key, result)

    def test_correct_values(self) -> None:
        fetcher = self._fetcher_with_info(self._SAMPLE_INFO)
        result = fetcher.get_enterprise_metrics()
        self.assertAlmostEqual(result["totalDebt"], 500_000_000)
        self.assertAlmostEqual(result["cashAndCashEquivalents"], 200_000_000)
        self.assertAlmostEqual(result["sharesOutstanding"], 6_760_000_000)

    def test_missing_fields_default_to_zero(self) -> None:
        fetcher = self._fetcher_with_info({"symbol": "RELIANCE.NS"})
        result = fetcher.get_enterprise_metrics()
        self.assertEqual(result["totalDebt"], 0.0)
        self.assertEqual(result["cashAndCashEquivalents"], 0.0)
        self.assertEqual(result["sharesOutstanding"], 0.0)

    def test_empty_info_raises(self) -> None:
        fetcher = self._fetcher_with_info({})
        with self.assertRaises(DataFetchError):
            fetcher.get_enterprise_metrics()


class TestYFinanceGetWacc(unittest.TestCase):
    _SAMPLE_INFO = {
        "beta": 0.9,
        "marketCap": 20_000_000_000,
        "totalDebt": 5_000_000_000,
    }

    def _fetcher_with_info(self, info: dict) -> YFinanceDataFetcher:
        fetcher = _make_yf_fetcher()
        mock_stock = MagicMock()
        mock_stock.info = info
        fetcher._stock = MagicMock(return_value=mock_stock)
        return fetcher

    def test_returns_correct_keys(self) -> None:
        fetcher = self._fetcher_with_info(self._SAMPLE_INFO)
        result = fetcher.get_wacc()
        for key in ("wacc", "costOfEquity", "costOfDebt", "taxRate", "equityWeight", "debtWeight"):
            self.assertIn(key, result)

    def test_capm_cost_of_equity(self) -> None:
        fetcher = self._fetcher_with_info(self._SAMPLE_INFO)
        result = fetcher.get_wacc()
        # CAPM: 0.07 + 0.9 * (0.12 - 0.07) = 0.115
        self.assertAlmostEqual(result["costOfEquity"], 0.115, places=6)

    def test_default_cost_of_debt(self) -> None:
        fetcher = self._fetcher_with_info(self._SAMPLE_INFO)
        result = fetcher.get_wacc()
        self.assertAlmostEqual(result["costOfDebt"], 0.10, places=6)

    def test_default_tax_rate(self) -> None:
        fetcher = self._fetcher_with_info(self._SAMPLE_INFO)
        result = fetcher.get_wacc()
        self.assertAlmostEqual(result["taxRate"], 0.25, places=6)

    def test_beta_none_defaults_to_one(self) -> None:
        info = dict(self._SAMPLE_INFO)
        info["beta"] = None
        fetcher = self._fetcher_with_info(info)
        result = fetcher.get_wacc()
        # With beta=1: cost_of_equity = 0.07 + 1 * 0.05 = 0.12
        self.assertAlmostEqual(result["costOfEquity"], 0.12, places=6)

    def test_weights_sum_to_one(self) -> None:
        fetcher = self._fetcher_with_info(self._SAMPLE_INFO)
        result = fetcher.get_wacc()
        self.assertAlmostEqual(result["equityWeight"] + result["debtWeight"], 1.0, places=6)

    def test_no_debt_equity_weight_is_one(self) -> None:
        info = {"beta": 1.0, "marketCap": 10_000_000_000, "totalDebt": 0}
        fetcher = self._fetcher_with_info(info)
        result = fetcher.get_wacc()
        self.assertAlmostEqual(result["equityWeight"], 1.0, places=6)
        self.assertAlmostEqual(result["debtWeight"], 0.0, places=6)

    def test_wacc_positive(self) -> None:
        fetcher = self._fetcher_with_info(self._SAMPLE_INFO)
        result = fetcher.get_wacc()
        self.assertGreater(result["wacc"], 0)


class TestYFinanceGetCurrentPrice(unittest.TestCase):
    def _fetcher_with_info(self, info: dict) -> YFinanceDataFetcher:
        fetcher = _make_yf_fetcher()
        mock_stock = MagicMock()
        mock_stock.info = info
        fetcher._stock = MagicMock(return_value=mock_stock)
        return fetcher

    def test_returns_current_price(self) -> None:
        fetcher = self._fetcher_with_info({"currentPrice": 2500.0, "symbol": "RELIANCE.NS"})
        result = fetcher.get_current_price()
        self.assertAlmostEqual(result["price"], 2500.0)
        self.assertEqual(result["symbol"], "RELIANCE.NS")

    def test_falls_back_to_regular_market_price(self) -> None:
        fetcher = self._fetcher_with_info({"regularMarketPrice": 1800.0})
        result = fetcher.get_current_price()
        self.assertAlmostEqual(result["price"], 1800.0)

    def test_falls_back_to_previous_close(self) -> None:
        fetcher = self._fetcher_with_info({"previousClose": 1750.0})
        result = fetcher.get_current_price()
        self.assertAlmostEqual(result["price"], 1750.0)

    def test_price_is_float(self) -> None:
        fetcher = self._fetcher_with_info({"currentPrice": 2500})
        result = fetcher.get_current_price()
        self.assertIsInstance(result["price"], float)

    def test_no_price_raises(self) -> None:
        fetcher = self._fetcher_with_info({"symbol": "RELIANCE.NS"})
        with self.assertRaises(DataFetchError):
            fetcher.get_current_price()

    def test_empty_info_raises(self) -> None:
        fetcher = self._fetcher_with_info({})
        with self.assertRaises(DataFetchError):
            fetcher.get_current_price()


class TestYFinanceGetHistoricalFcfGrowth(unittest.TestCase):
    def _fetcher_with_cf(self, cf_df) -> YFinanceDataFetcher:
        fetcher = _make_yf_fetcher()
        mock_stock = MagicMock()
        mock_stock.cash_flow = cf_df
        fetcher._stock = MagicMock(return_value=mock_stock)
        return fetcher

    def _make_multi_year_cf(self, fcf_values: list) -> "pd.DataFrame":
        """Build a DataFrame with FCF values (newest first in columns)."""
        import pandas as pd

        dates = pd.date_range("2023-03-31", periods=len(fcf_values), freq="-1YE")
        data = {d: {"Free Cash Flow": v} for d, v in zip(dates, fcf_values)}
        return pd.DataFrame(data)

    def test_returns_float(self) -> None:
        cf = self._make_multi_year_cf([1000, 900, 800, 700, 600])
        fetcher = self._fetcher_with_cf(cf)
        result = fetcher.get_historical_fcf_growth()
        self.assertIsInstance(result, float)

    def test_cagr_positive_values(self) -> None:
        # oldest=600, newest=1000, n=4 → CAGR = (1000/600)^(1/4) - 1
        cf = self._make_multi_year_cf([1000, 900, 800, 700, 600])
        fetcher = self._fetcher_with_cf(cf)
        result = fetcher.get_historical_fcf_growth()
        expected = (1000 / 600) ** (1 / 4) - 1
        self.assertAlmostEqual(result, expected, places=6)

    def test_single_column_returns_default(self) -> None:
        cf = self._make_multi_year_cf([1000])
        fetcher = self._fetcher_with_cf(cf)
        result = fetcher.get_historical_fcf_growth()
        self.assertAlmostEqual(result, 0.05)

    def test_empty_df_returns_default(self) -> None:
        import pandas as pd

        fetcher = self._fetcher_with_cf(pd.DataFrame())
        result = fetcher.get_historical_fcf_growth()
        self.assertAlmostEqual(result, 0.05)

    def test_none_cash_flow_returns_default(self) -> None:
        fetcher = self._fetcher_with_cf(None)
        result = fetcher.get_historical_fcf_growth()
        self.assertAlmostEqual(result, 0.05)


if __name__ == "__main__":
    unittest.main()
