"""Unit tests for valuation.dcf_model.

All tests are fully offline – no external calls are made.  The tests validate
the mathematical correctness of the DCF engine as well as every input-
validation code path.
"""

from __future__ import annotations

import math
import unittest

from valuation.dcf_model import DCFModel, DCFModelError

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Represents a company with:
#   Current FCF : $100 (convenient round number for math verification)
#   Total Debt  : $300
#   Cash        : $50
#   Shares      : 10
_CASH_FLOW_DATA: dict = {"freeCashFlow": 100.0, "date": "2023-09-30"}
_ENTERPRISE_METRICS: dict = {
    "totalDebt": 300.0,
    "cashAndCashEquivalents": 50.0,
    "sharesOutstanding": 10.0,
    "date": "2023-09-30",
}

# Valuation parameters
_GROWTH = 0.10       # 10 % FCF growth
_WACC = 0.09         # 9 % discount rate
_TERMINAL_G = 0.025  # 2.5 % perpetual growth
_YEARS = 5


def _make_model(**overrides) -> DCFModel:
    cf = dict(_CASH_FLOW_DATA)
    em = dict(_ENTERPRISE_METRICS)
    cf.update(overrides.get("cash_flow_data", {}))
    em.update(overrides.get("enterprise_metrics", {}))
    return DCFModel(cf, em)


# ---------------------------------------------------------------------------
# DCFModel instantiation
# ---------------------------------------------------------------------------


class TestDCFModelInit(unittest.TestCase):
    def test_basic_construction(self) -> None:
        model = _make_model()
        self.assertEqual(model.current_fcf, 100.0)
        self.assertEqual(model.total_debt, 300.0)
        self.assertEqual(model.cash, 50.0)
        self.assertEqual(model.shares_outstanding, 10.0)

    def test_missing_free_cash_flow_raises(self) -> None:
        with self.assertRaises(DCFModelError):
            DCFModel({"date": "2023-09-30"}, _ENTERPRISE_METRICS)

    def test_none_free_cash_flow_raises(self) -> None:
        with self.assertRaises(DCFModelError):
            DCFModel({"freeCashFlow": None}, _ENTERPRISE_METRICS)

    def test_missing_total_debt_raises(self) -> None:
        em = {k: v for k, v in _ENTERPRISE_METRICS.items() if k != "totalDebt"}
        with self.assertRaises(DCFModelError):
            DCFModel(_CASH_FLOW_DATA, em)

    def test_missing_cash_raises(self) -> None:
        em = {k: v for k, v in _ENTERPRISE_METRICS.items() if k != "cashAndCashEquivalents"}
        with self.assertRaises(DCFModelError):
            DCFModel(_CASH_FLOW_DATA, em)

    def test_missing_shares_outstanding_raises(self) -> None:
        em = {k: v for k, v in _ENTERPRISE_METRICS.items() if k != "sharesOutstanding"}
        with self.assertRaises(DCFModelError):
            DCFModel(_CASH_FLOW_DATA, em)

    def test_zero_shares_outstanding_raises(self) -> None:
        with self.assertRaises(DCFModelError):
            _make_model(enterprise_metrics={"sharesOutstanding": 0})

    def test_integer_fcf_converted_to_float(self) -> None:
        model = DCFModel({"freeCashFlow": 1_000_000}, _ENTERPRISE_METRICS)
        self.assertIsInstance(model.current_fcf, float)


# ---------------------------------------------------------------------------
# calculate_intrinsic_value – input validation
# ---------------------------------------------------------------------------


class TestCalculateInputValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _make_model()

    def test_wacc_equal_to_terminal_growth_raises(self) -> None:
        with self.assertRaises(DCFModelError):
            self.model.calculate_intrinsic_value(0.08, wacc=0.025, terminal_growth_rate=0.025)

    def test_wacc_less_than_terminal_growth_raises(self) -> None:
        with self.assertRaises(DCFModelError):
            self.model.calculate_intrinsic_value(0.08, wacc=0.02, terminal_growth_rate=0.025)

    def test_non_positive_wacc_raises(self) -> None:
        with self.assertRaises(DCFModelError):
            self.model.calculate_intrinsic_value(0.08, wacc=0.0)

    def test_negative_wacc_raises(self) -> None:
        with self.assertRaises(DCFModelError):
            self.model.calculate_intrinsic_value(0.08, wacc=-0.05)

    def test_zero_years_raises(self) -> None:
        with self.assertRaises(DCFModelError):
            self.model.calculate_intrinsic_value(0.08, wacc=0.09, years=0)

    def test_negative_years_raises(self) -> None:
        with self.assertRaises(DCFModelError):
            self.model.calculate_intrinsic_value(0.08, wacc=0.09, years=-1)

    def test_float_years_raises(self) -> None:
        with self.assertRaises(DCFModelError):
            self.model.calculate_intrinsic_value(0.08, wacc=0.09, years=5.0)  # type: ignore


# ---------------------------------------------------------------------------
# calculate_intrinsic_value – return-value structure
# ---------------------------------------------------------------------------


class TestCalculateReturnStructure(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _make_model()
        self.result = self.model.calculate_intrinsic_value(
            _GROWTH, _WACC, _TERMINAL_G, _YEARS
        )

    def test_result_is_dict(self) -> None:
        self.assertIsInstance(self.result, dict)

    def test_required_keys_present(self) -> None:
        expected_keys = {
            "projected_fcfs",
            "terminal_value",
            "pv_terminal_value",
            "sum_pv_fcfs",
            "enterprise_value",
            "total_debt",
            "cash",
            "equity_value",
            "shares_outstanding",
            "intrinsic_value_per_share",
        }
        self.assertEqual(expected_keys, set(self.result.keys()))

    def test_projected_fcfs_has_correct_count(self) -> None:
        self.assertEqual(len(self.result["projected_fcfs"]), _YEARS)

    def test_per_year_dict_has_expected_keys(self) -> None:
        for entry in self.result["projected_fcfs"]:
            self.assertIn("year", entry)
            self.assertIn("projected_fcf", entry)
            self.assertIn("present_value", entry)

    def test_year_numbers_are_sequential(self) -> None:
        years = [entry["year"] for entry in self.result["projected_fcfs"]]
        self.assertEqual(years, list(range(1, _YEARS + 1)))

    def test_enterprise_metrics_echoed_in_result(self) -> None:
        self.assertEqual(self.result["total_debt"], 300.0)
        self.assertEqual(self.result["cash"], 50.0)
        self.assertEqual(self.result["shares_outstanding"], 10.0)


# ---------------------------------------------------------------------------
# calculate_intrinsic_value – mathematical correctness
# ---------------------------------------------------------------------------


class TestCalculateMath(unittest.TestCase):
    """Hand-verify every intermediate step against known-good values."""

    def setUp(self) -> None:
        # Use simple numbers so we can verify by hand
        # FCF = 100, growth = 10 %, WACC = 9 %, terminal_g = 2.5 %, years = 5
        self.model = _make_model()
        self.result = self.model.calculate_intrinsic_value(
            fcf_growth_rate=_GROWTH,
            wacc=_WACC,
            terminal_growth_rate=_TERMINAL_G,
            years=_YEARS,
        )

    def _expected_fcf(self, year: int) -> float:
        return 100.0 * (1.10 ** year)

    def _expected_pv(self, year: int) -> float:
        return self._expected_fcf(year) / (1.09 ** year)

    def test_projected_fcf_values(self) -> None:
        for entry in self.result["projected_fcfs"]:
            year = entry["year"]
            self.assertAlmostEqual(
                entry["projected_fcf"],
                self._expected_fcf(year),
                places=6,
                msg=f"Projected FCF mismatch for year {year}",
            )

    def test_present_value_of_fcfs(self) -> None:
        for entry in self.result["projected_fcfs"]:
            year = entry["year"]
            self.assertAlmostEqual(
                entry["present_value"],
                self._expected_pv(year),
                places=6,
                msg=f"PV of FCF mismatch for year {year}",
            )

    def test_sum_pv_fcfs(self) -> None:
        expected_sum = sum(self._expected_pv(y) for y in range(1, _YEARS + 1))
        self.assertAlmostEqual(self.result["sum_pv_fcfs"], expected_sum, places=6)

    def test_terminal_value(self) -> None:
        # FCF in year 5 grown by terminal_g, then divided by (wacc - terminal_g)
        fcf_year5 = self._expected_fcf(5)
        expected_tv = fcf_year5 * (1.0 + _TERMINAL_G) / (_WACC - _TERMINAL_G)
        self.assertAlmostEqual(self.result["terminal_value"], expected_tv, places=6)

    def test_pv_terminal_value(self) -> None:
        fcf_year5 = self._expected_fcf(5)
        tv = fcf_year5 * (1.0 + _TERMINAL_G) / (_WACC - _TERMINAL_G)
        expected_pv_tv = tv / (1.09 ** 5)
        self.assertAlmostEqual(self.result["pv_terminal_value"], expected_pv_tv, places=6)

    def test_enterprise_value(self) -> None:
        expected_ev = self.result["sum_pv_fcfs"] + self.result["pv_terminal_value"]
        self.assertAlmostEqual(self.result["enterprise_value"], expected_ev, places=6)

    def test_equity_value(self) -> None:
        # EV - debt + cash
        expected_eq = self.result["enterprise_value"] - 300.0 + 50.0
        self.assertAlmostEqual(self.result["equity_value"], expected_eq, places=6)

    def test_intrinsic_value_per_share(self) -> None:
        expected_ivps = self.result["equity_value"] / 10.0
        self.assertAlmostEqual(
            self.result["intrinsic_value_per_share"], expected_ivps, places=6
        )

    def test_all_pv_fcfs_are_finite(self) -> None:
        for entry in self.result["projected_fcfs"]:
            self.assertTrue(math.isfinite(entry["present_value"]))

    def test_intrinsic_value_is_finite(self) -> None:
        self.assertTrue(math.isfinite(self.result["intrinsic_value_per_share"]))


# ---------------------------------------------------------------------------
# calculate_intrinsic_value – edge cases
# ---------------------------------------------------------------------------


class TestCalculateEdgeCases(unittest.TestCase):
    def test_negative_fcf_yields_negative_equity_value(self) -> None:
        """A company with deeply negative FCF should produce a negative intrinsic value."""
        model = DCFModel(
            {"freeCashFlow": -500.0},
            _ENTERPRISE_METRICS,
        )
        result = model.calculate_intrinsic_value(0.05, 0.09)
        self.assertLess(result["intrinsic_value_per_share"], 0)

    def test_zero_growth_rate(self) -> None:
        """A 0 % growth rate should produce flat FCF projections."""
        model = _make_model()
        result = model.calculate_intrinsic_value(0.0, _WACC, _TERMINAL_G, _YEARS)
        for entry in result["projected_fcfs"]:
            self.assertAlmostEqual(entry["projected_fcf"], 100.0, places=6)

    def test_single_year_forecast(self) -> None:
        model = _make_model()
        result = model.calculate_intrinsic_value(0.10, _WACC, _TERMINAL_G, years=1)
        self.assertEqual(len(result["projected_fcfs"]), 1)
        self.assertTrue(math.isfinite(result["intrinsic_value_per_share"]))

    def test_longer_forecast_horizon(self) -> None:
        model = _make_model()
        result = model.calculate_intrinsic_value(0.10, _WACC, _TERMINAL_G, years=10)
        self.assertEqual(len(result["projected_fcfs"]), 10)

    def test_default_terminal_growth_rate(self) -> None:
        """Calling without terminal_growth_rate should use 2.5 % default."""
        model = _make_model()
        result_default = model.calculate_intrinsic_value(0.08, _WACC)
        result_explicit = model.calculate_intrinsic_value(0.08, _WACC, terminal_growth_rate=0.025)
        self.assertAlmostEqual(
            result_default["intrinsic_value_per_share"],
            result_explicit["intrinsic_value_per_share"],
            places=10,
        )

    def test_zero_debt_and_cash(self) -> None:
        """With zero debt and cash, equity value should equal enterprise value."""
        model = DCFModel(
            _CASH_FLOW_DATA,
            {"totalDebt": 0.0, "cashAndCashEquivalents": 0.0, "sharesOutstanding": 10.0},
        )
        result = model.calculate_intrinsic_value(0.08, _WACC, _TERMINAL_G, _YEARS)
        self.assertAlmostEqual(
            result["equity_value"], result["enterprise_value"], places=10
        )


# ---------------------------------------------------------------------------
# DCFModelError
# ---------------------------------------------------------------------------


class TestDCFModelError(unittest.TestCase):
    def test_is_exception_subclass(self) -> None:
        self.assertTrue(issubclass(DCFModelError, Exception))

    def test_can_be_raised_and_caught(self) -> None:
        with self.assertRaises(DCFModelError) as ctx:
            raise DCFModelError("invalid wacc")
        self.assertIn("invalid wacc", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
