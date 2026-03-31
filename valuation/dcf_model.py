"""Core mathematical engine for the DCF valuation pipeline.

Provides :class:`DCFModel`, which accepts pre-fetched financial data and
performs a multi-stage discounted cash flow analysis to arrive at an
intrinsic value per share.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class DCFModelError(Exception):
    """Raised when the DCF calculation cannot be completed.

    Common triggers include invalid inputs (e.g. WACC ≤ terminal growth rate,
    zero or missing shares outstanding) or data that would produce economically
    meaningless results.
    """


# ---------------------------------------------------------------------------
# DCF Model
# ---------------------------------------------------------------------------


class DCFModel:
    """Discounted Cash Flow valuation model.

    The model accepts the raw data dictionaries returned by
    :class:`~valuation.data_fetcher.FinancialDataFetcher` and exposes a
    single high-level method, :meth:`calculate_intrinsic_value`, that walks
    through every step of the classic DCF methodology.

    Parameters
    ----------
    cash_flow_data:
        Dictionary as returned by
        :meth:`~valuation.data_fetcher.FinancialDataFetcher.get_cash_flow_statement`.
        Must contain a numeric ``freeCashFlow`` key.
    enterprise_metrics:
        Dictionary as returned by
        :meth:`~valuation.data_fetcher.FinancialDataFetcher.get_enterprise_metrics`.
        Must contain ``totalDebt``, ``cashAndCashEquivalents``, and
        ``sharesOutstanding`` keys.

    Raises
    ------
    DCFModelError
        If required fields are absent or ``sharesOutstanding`` is zero.

    Examples
    --------
    >>> model = DCFModel(cash_flow_data, enterprise_metrics)
    >>> result = model.calculate_intrinsic_value(
    ...     fcf_growth_rate=0.08,
    ...     wacc=0.09,
    ...     terminal_growth_rate=0.025,
    ...     years=5,
    ... )
    >>> result["intrinsic_value_per_share"]
    172.34
    """

    def __init__(
        self,
        cash_flow_data: dict[str, Any],
        enterprise_metrics: dict[str, Any],
    ) -> None:
        # --- Validate and extract cash flow data ---
        if "freeCashFlow" not in cash_flow_data or cash_flow_data["freeCashFlow"] is None:
            raise DCFModelError(
                "cash_flow_data must contain a non-None 'freeCashFlow' field."
            )
        self.current_fcf: float = float(cash_flow_data["freeCashFlow"])

        # --- Validate and extract enterprise metrics ---
        for field in ("totalDebt", "cashAndCashEquivalents", "sharesOutstanding"):
            if field not in enterprise_metrics or enterprise_metrics[field] is None:
                raise DCFModelError(
                    f"enterprise_metrics must contain a non-None '{field}' field."
                )

        self.total_debt: float = float(enterprise_metrics["totalDebt"])
        self.cash: float = float(enterprise_metrics["cashAndCashEquivalents"])
        self.shares_outstanding: float = float(enterprise_metrics["sharesOutstanding"])

        if self.shares_outstanding == 0:
            raise DCFModelError(
                "sharesOutstanding must be greater than zero to compute a "
                "per-share intrinsic value."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_intrinsic_value(
        self,
        fcf_growth_rate: float,
        wacc: float,
        terminal_growth_rate: float = 0.025,
        years: int = 5,
    ) -> dict[str, Any]:
        """Compute the intrinsic value per share via a multi-stage DCF.

        The calculation proceeds in six steps:

        1. **Project FCF** – grow the current free cash flow forward by
           *fcf_growth_rate* for each of the *years* forecast periods.
        2. **Discount FCFs to PV** – divide each projected FCF by
           ``(1 + wacc) ** t`` where *t* is the year number.
        3. **Terminal Value** – apply the Gordon Growth Model at the end of
           the forecast horizon:
           ``TV = FCF_n × (1 + terminal_growth_rate) / (wacc − terminal_growth_rate)``
        4. **Discount TV to PV** – ``PV_TV = TV / (1 + wacc) ** years``
        5. **Enterprise Value** – ``EV = Σ PV(FCFs) + PV_TV``
        6. **Equity Value → Per-Share Value**:
           ``Equity Value = EV − Total Debt + Cash``
           ``Intrinsic Value Per Share = Equity Value / Shares Outstanding``

        Parameters
        ----------
        fcf_growth_rate:
            Expected annual growth rate of free cash flow during the explicit
            forecast period, expressed as a decimal (e.g. ``0.08`` for 8 %).
        wacc:
            Weighted average cost of capital used as the discount rate,
            expressed as a decimal (e.g. ``0.09`` for 9 %).
        terminal_growth_rate:
            Perpetual growth rate applied in the Gordon Growth Model after the
            forecast horizon, expressed as a decimal.  Defaults to ``0.025``
            (2.5 %).
        years:
            Number of years in the explicit FCF forecast period.
            Defaults to ``5``.

        Returns
        -------
        dict[str, Any]
            A comprehensive result dictionary with the following keys:

            * ``"projected_fcfs"`` – list of per-year dicts, each containing:
              ``year`` (int), ``projected_fcf`` (float), ``present_value`` (float).
            * ``"terminal_value"`` – undiscounted terminal value (float).
            * ``"pv_terminal_value"`` – terminal value discounted to today (float).
            * ``"sum_pv_fcfs"`` – sum of the discounted FCFs (float).
            * ``"enterprise_value"`` – implied enterprise value (float).
            * ``"total_debt"`` – total debt used in the bridge (float).
            * ``"cash"`` – cash used in the bridge (float).
            * ``"equity_value"`` – implied equity value (float).
            * ``"shares_outstanding"`` – shares used for the per-share calc (float).
            * ``"intrinsic_value_per_share"`` – final per-share intrinsic value (float).

        Raises
        ------
        DCFModelError
            If *wacc* ≤ *terminal_growth_rate* (Gordon Growth Model would
            produce a non-positive or infinite terminal value), if *years* is
            not a positive integer, or if *wacc* is not positive.
        """
        # --- Input validation ---
        if years < 1 or not isinstance(years, int):
            raise DCFModelError("'years' must be a positive integer.")
        if wacc <= 0:
            raise DCFModelError("'wacc' must be a positive value.")
        if wacc <= terminal_growth_rate:
            raise DCFModelError(
                f"'wacc' ({wacc}) must be strictly greater than "
                f"'terminal_growth_rate' ({terminal_growth_rate}) for the "
                "Gordon Growth Model to produce a finite, positive terminal value."
            )

        # --- Step 1 & 2: Project FCFs and discount to PV ---
        projected_fcfs: list[dict[str, Any]] = []
        fcf = self.current_fcf
        sum_pv_fcfs: float = 0.0

        for year in range(1, years + 1):
            fcf = fcf * (1.0 + fcf_growth_rate)
            pv = fcf / (1.0 + wacc) ** year
            sum_pv_fcfs += pv
            projected_fcfs.append(
                {
                    "year": year,
                    "projected_fcf": fcf,
                    "present_value": pv,
                }
            )

        # --- Step 3 & 4: Terminal Value (Gordon Growth Model) ---
        terminal_fcf = fcf * (1.0 + terminal_growth_rate)
        terminal_value: float = terminal_fcf / (wacc - terminal_growth_rate)
        pv_terminal_value: float = terminal_value / (1.0 + wacc) ** years

        # --- Step 5: Enterprise Value ---
        enterprise_value: float = sum_pv_fcfs + pv_terminal_value

        # --- Step 6: Equity Value and per-share intrinsic value ---
        equity_value: float = enterprise_value - self.total_debt + self.cash
        intrinsic_value_per_share: float = equity_value / self.shares_outstanding

        return {
            "projected_fcfs": projected_fcfs,
            "terminal_value": terminal_value,
            "pv_terminal_value": pv_terminal_value,
            "sum_pv_fcfs": sum_pv_fcfs,
            "enterprise_value": enterprise_value,
            "total_debt": self.total_debt,
            "cash": self.cash,
            "equity_value": equity_value,
            "shares_outstanding": self.shares_outstanding,
            "intrinsic_value_per_share": intrinsic_value_per_share,
        }
