"""Unit tests for valuation.data_fetcher.

All tests are fully offline – no real HTTP calls are made.  The FMP API is
mocked via :mod:`unittest.mock`.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from valuation.data_fetcher import DataFetchError, FinancialDataFetcher

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


if __name__ == "__main__":
    unittest.main()
