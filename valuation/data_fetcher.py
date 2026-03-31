"""Data ingestion module for the DCF Engine.

Provides :class:`FinancialDataFetcher`, which wraps the Financial Modeling
Prep (FMP) REST API and returns clean, typed data structures consumed by the
downstream valuation pipeline.
"""

from __future__ import annotations

import os
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class DataFetchError(Exception):
    """Raised when data cannot be retrieved from the FMP API.

    This covers network failures, HTTP error responses, invalid tickers, and
    any unexpected payload shape returned by the API.
    """


# ---------------------------------------------------------------------------
# Main fetcher class
# ---------------------------------------------------------------------------

_BASE_V3 = "https://financialmodelingprep.com/api/v3"
_BASE_V4 = "https://financialmodelingprep.com/api/v4"

_REQUEST_TIMEOUT = 15  # seconds


class FinancialDataFetcher:
    """Fetches financial data for a given ticker from the FMP API.

    Parameters
    ----------
    ticker:
        The equity ticker symbol (e.g. ``"AAPL"``).
    api_key:
        A valid FMP API key.  Defaults to the value of the ``FMP_API_KEY``
        environment variable when *api_key* is ``None``.

    Raises
    ------
    DataFetchError
        If *api_key* is ``None`` and ``FMP_API_KEY`` is not set in the
        environment.

    Examples
    --------
    >>> fetcher = FinancialDataFetcher("AAPL")
    >>> cf = fetcher.get_cash_flow_statement()
    >>> cf["freeCashFlow"]
    12345678900
    """

    def __init__(
        self,
        ticker: str,
        api_key: str | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("FMP_API_KEY")
        if not resolved_key:
            raise DataFetchError(
                "FMP API key is required.  Pass api_key= or set the "
                "FMP_API_KEY environment variable."
            )
        self.ticker: str = ticker.upper().strip()
        self._api_key: str = resolved_key

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        """Perform a GET request and return the parsed JSON payload.

        Parameters
        ----------
        url:
            The fully-qualified endpoint URL.
        params:
            Optional query-string parameters (the API key is always appended).

        Returns
        -------
        Any
            The decoded JSON value (usually a ``list`` or ``dict``).

        Raises
        ------
        DataFetchError
            On any network error, non-2xx HTTP status, or if the API returns
            an error message instead of the expected payload.
        """
        merged: dict[str, Any] = {"apikey": self._api_key}
        if params:
            merged.update(params)

        try:
            response = requests.get(url, params=merged, timeout=_REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise DataFetchError(
                f"Network error while contacting FMP API: {exc}"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise DataFetchError(
                f"Request timed out after {_REQUEST_TIMEOUT}s: {exc}"
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise DataFetchError(
                f"HTTP {response.status_code} error from FMP API "
                f"for ticker '{self.ticker}': {exc}"
            ) from exc

        payload = response.json()

        # FMP signals invalid tickers / quota errors inside a 200 response
        # via {"Error Message": "..."} or {"message": "..."}.
        if isinstance(payload, dict):
            for error_key in ("Error Message", "message", "error"):
                if error_key in payload:
                    raise DataFetchError(
                        f"FMP API error for ticker '{self.ticker}': "
                        f"{payload[error_key]}"
                    )

        return payload

    def _require_list(self, payload: Any, endpoint: str) -> list[dict[str, Any]]:
        """Validate that *payload* is a non-empty list.

        Raises
        ------
        DataFetchError
            If the payload is not a list or is empty.
        """
        if not isinstance(payload, list) or len(payload) == 0:
            raise DataFetchError(
                f"No data returned from '{endpoint}' for ticker "
                f"'{self.ticker}'.  The ticker may be invalid or the FMP "
                f"plan may not cover this endpoint."
            )
        return payload  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_cash_flow_statement(self, limit: int = 1) -> dict[str, Any]:
        """Return the most recent annual cash flow statement.

        Fetches the ``/api/v3/cash-flow-statement`` endpoint and returns the
        latest record, which includes the company's **free cash flow** as well
        as operating and capital-expenditure figures.

        Parameters
        ----------
        limit:
            Number of annual periods to request from FMP.  Defaults to ``1``
            (most recent fiscal year only).

        Returns
        -------
        dict[str, Any]
            A dictionary with FMP cash-flow fields.  Key fields used
            downstream:

            * ``freeCashFlow`` – operating cash flow minus capex (int, USD).
            * ``operatingCashFlow`` – net cash from operations.
            * ``capitalExpenditure`` – capital expenditures (usually negative).
            * ``date`` – fiscal year end date (``"YYYY-MM-DD"``).

        Raises
        ------
        DataFetchError
            If the API call fails or no data is available for the ticker.
        """
        url = f"{_BASE_V3}/cash-flow-statement/{self.ticker}"
        payload = self._get(url, params={"period": "annual", "limit": limit})
        records = self._require_list(payload, f"cash-flow-statement/{self.ticker}")
        return records[0]

    def get_enterprise_metrics(self) -> dict[str, Any]:
        """Return key enterprise-value metrics for the company.

        Combines data from the balance-sheet statement (total debt, cash and
        short-term investments) and the shares-outstanding endpoint to build
        a single consolidated dictionary.

        Returns
        -------
        dict[str, Any]
            A flat dictionary with the following keys:

            * ``totalDebt`` – total financial debt (long + short term, USD).
            * ``cashAndCashEquivalents`` – cash & short-term investments (USD).
            * ``sharesOutstanding`` – diluted shares outstanding (int).
            * ``date`` – balance-sheet date (``"YYYY-MM-DD"``).

        Raises
        ------
        DataFetchError
            If either underlying API call fails or no data is returned.
        """
        # --- Balance sheet (debt + cash) ---
        bs_url = f"{_BASE_V3}/balance-sheet-statement/{self.ticker}"
        bs_payload = self._get(bs_url, params={"period": "annual", "limit": 1})
        bs_records = self._require_list(
            bs_payload, f"balance-sheet-statement/{self.ticker}"
        )
        bs = bs_records[0]

        # FMP uses different field names across plan tiers; normalise gracefully.
        # Use an explicit None check so that a legitimate 0-debt company is not
        # incorrectly overridden by the long/short-term fallback sum.
        total_debt_raw = bs.get("totalDebt")
        total_debt: float = (
            total_debt_raw
            if total_debt_raw is not None
            else (bs.get("longTermDebt") or 0) + (bs.get("shortTermDebt") or 0)
        )
        cash: float = bs.get("cashAndCashEquivalents") or bs.get(
            "cashAndShortTermInvestments", 0
        )

        # shares_float is preferred over the balance-sheet sharesOutstanding
        # field because it reflects the most up-to-date float (including
        # recent buybacks or issuances) rather than the last filed value.
        shares_url = f"{_BASE_V3}/shares_float"
        shares_payload = self._get(shares_url, params={"symbol": self.ticker})
        shares_records = self._require_list(shares_payload, f"shares_float?symbol={self.ticker}")
        shares_outstanding: float = shares_records[0].get("outstandingShares", 0)

        return {
            "totalDebt": total_debt,
            "cashAndCashEquivalents": cash,
            "sharesOutstanding": shares_outstanding,
            "date": bs.get("date", ""),
        }

    def get_wacc(self) -> dict[str, Any]:
        """Return the Weighted Average Cost of Capital (WACC) for the company.

        Uses the FMP ``/api/v4/advanced_discounted_cash_flow`` endpoint, which
        provides a pre-computed WACC alongside the component inputs (cost of
        equity, cost of debt, tax rate, etc.).

        Returns
        -------
        dict[str, Any]
            A dictionary with the following keys (all floats expressed as
            decimals, e.g. ``0.085`` for 8.5 %):

            * ``wacc`` – weighted average cost of capital.
            * ``costOfEquity`` – required return on equity (CAPM-based).
            * ``costOfDebt`` – pre-tax cost of debt.
            * ``taxRate`` – effective corporate tax rate.
            * ``equityWeight`` – weight of equity in the capital structure.
            * ``debtWeight`` – weight of debt in the capital structure.

        Raises
        ------
        DataFetchError
            If the API call fails, the ticker is invalid, or the WACC field
            is missing from the response.
        """
        url = f"{_BASE_V4}/advanced_discounted_cash_flow"
        payload = self._get(url, params={"symbol": self.ticker})
        records = self._require_list(payload, "advanced_discounted_cash_flow")
        record = records[0]

        wacc_value = record.get("wacc")
        if wacc_value is None:
            raise DataFetchError(
                f"WACC field missing in FMP response for ticker '{self.ticker}'."
            )

        return {
            "wacc": wacc_value,
            "costOfEquity": record.get("costOfEquity"),
            "costOfDebt": record.get("costOfDebt"),
            "taxRate": record.get("taxRate"),
            "equityWeight": record.get("equityWeight"),
            "debtWeight": record.get("debtWeight"),
        }

    def get_current_price(self) -> dict[str, Any]:
        """Return the current market price for the stock.

        Uses the FMP ``/api/v3/quote/{ticker}`` endpoint, which provides the
        latest trade price alongside volume and market-cap metadata.

        Returns
        -------
        dict[str, Any]
            A dictionary with the following keys:

            * ``price`` – latest trade price (float, USD).
            * ``symbol`` – normalised ticker symbol (str).

        Raises
        ------
        DataFetchError
            If the API call fails, no data is returned, or the ``price``
            field is absent from the response.
        """
        url = f"{_BASE_V3}/quote/{self.ticker}"
        payload = self._get(url)
        records = self._require_list(payload, f"quote/{self.ticker}")
        record = records[0]

        price = record.get("price")
        if price is None:
            raise DataFetchError(
                f"'price' field missing in FMP quote response for ticker "
                f"'{self.ticker}'."
            )

        return {
            "price": float(price),
            "symbol": record.get("symbol", self.ticker),
        }

    def get_historical_fcf_growth(self, years: int = 5) -> float:
        """Compute the historical average annual FCF growth rate.

        Fetches up to *years* annual cash-flow statements and returns the
        compound annual growth rate (CAGR) when both endpoints are positive.
        Falls back to the average of valid year-over-year growth rates, and
        ultimately to a 5 % default when no meaningful rate can be derived.

        Parameters
        ----------
        years:
            Number of historical annual periods to include.  Defaults to
            ``5``.

        Returns
        -------
        float
            Historical FCF growth rate as a decimal (e.g. ``0.08`` for 8 %).
        """
        url = f"{_BASE_V3}/cash-flow-statement/{self.ticker}"
        payload = self._get(url, params={"period": "annual", "limit": years})
        try:
            records = self._require_list(
                payload, f"cash-flow-statement/{self.ticker}"
            )
        except DataFetchError:
            return 0.05

        # FMP returns records newest-first; reverse to chronological order.
        fcf_values = [
            float(r["freeCashFlow"])
            for r in reversed(records)
            if r.get("freeCashFlow") is not None
        ]

        if len(fcf_values) < 2:
            return 0.05

        oldest, latest = fcf_values[0], fcf_values[-1]
        n = len(fcf_values) - 1

        # Prefer CAGR when both endpoints are strictly positive.
        if oldest > 0 and latest > 0:
            return (latest / oldest) ** (1.0 / n) - 1.0

        # Fallback: average the valid YoY growth rates.
        growth_rates = [
            curr / prev - 1.0
            for prev, curr in zip(fcf_values, fcf_values[1:])
            if prev > 0 and curr > 0
        ]
        return sum(growth_rates) / len(growth_rates) if growth_rates else 0.05
