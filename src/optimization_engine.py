"""
Capital-stack optimisation for the SDRB transaction.

This is the part a structurer would use to check whether the same collateral
pool can be financed cheaper without breaking senior subordination, OC, or
investor yield constraints. It optimises tranche thickness and coupons, then
reports tranche IRR, Macaulay duration, and equity VaR from simulated cashflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

from src.data_engine import SimulationResult
from src.structural_waterfall import StructuredWaterfallEngine, WaterfallResult


@dataclass(frozen=True)
class OptimizationResult:
    senior_thickness: float
    mezzanine_thickness: float
    equity_thickness: float
    senior_coupon: float
    mezzanine_coupon: float
    optimized_wacc: float
    success: bool
    message: str
    tranche_metrics: pd.DataFrame
    optimizer_diagnostics: Dict[str, float]


class CapitalStructureOptimizer:
    """Minimises issuer WACC while respecting the SDRB capital-stack rules."""

    def __init__(
        self,
        issuance_amount: float,
        horizon_months: int,
        optimization_config: Dict[str, float | list[float]],
        waterfall_config: Dict[str, float | bool],
        senior_yield_hurdle: float,
        mezzanine_yield_hurdle: float,
        equity_target_irr: float,
        confidence_level: float = 0.99,
    ) -> None:
        self.issuance_amount = float(issuance_amount)
        self.horizon_months = int(horizon_months)
        self.optimization_config = optimization_config
        self.waterfall_config = waterfall_config
        self.senior_yield_hurdle = float(senior_yield_hurdle)
        self.mezzanine_yield_hurdle = float(mezzanine_yield_hurdle)
        self.equity_target_irr = float(equity_target_irr)
        self.confidence_level = float(confidence_level)
        self._validate()

    @classmethod
    def from_config(cls, config_path: str | Path = "config/deal_structure.yaml") -> "CapitalStructureOptimizer":
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        return cls(
            issuance_amount=config["deal"]["issuance_amount"],
            horizon_months=config["deal"]["horizon_months"],
            optimization_config=config["optimization"],
            waterfall_config=config["waterfall"],
            senior_yield_hurdle=config["tranches"]["senior"]["institutional_yield_hurdle"],
            mezzanine_yield_hurdle=config["tranches"]["mezzanine"]["institutional_yield_hurdle"],
            equity_target_irr=config["tranches"]["equity"]["target_irr"],
            confidence_level=config["optimization"].get("confidence_level", 0.99),
        )

    def _validate(self) -> None:
        if self.issuance_amount <= 0:
            raise ValueError("issuance_amount must be positive")
        if not 0.90 <= self.confidence_level < 1.0:
            raise ValueError("confidence_level should be between 0.90 and 1.0")

    def optimise(self, simulation: SimulationResult, x0: Optional[Iterable[float]] = None) -> OptimizationResult:
        bounds = [
            tuple(self.optimization_config["senior_thickness_bounds"]),
            tuple(self.optimization_config["mezzanine_thickness_bounds"]),
            tuple(self.optimization_config["senior_coupon_bounds"]),
            tuple(self.optimization_config["mezzanine_coupon_bounds"]),
        ]
        if x0 is None:
            x0 = np.array(
                [
                    np.mean(bounds[0]),
                    np.mean(bounds[1]),
                    max(self.senior_yield_hurdle, np.mean(bounds[2])),
                    max(self.mezzanine_yield_hurdle, np.mean(bounds[3])),
                ],
                dtype=float,
            )
        else:
            x0 = np.asarray(list(x0), dtype=float)
            if x0.shape != (4,):
                raise ValueError("x0 must contain [senior_thickness, mezz_thickness, senior_coupon, mezz_coupon]")

        constraints = [
            {"type": "ineq", "fun": lambda x: self._equity_thickness(x) - self.optimization_config["min_equity_thickness"]},
            {"type": "ineq", "fun": lambda x: self.optimization_config["max_equity_thickness"] - self._equity_thickness(x)},
            {"type": "ineq", "fun": lambda x: x[2] - self.senior_yield_hurdle},
            {"type": "ineq", "fun": lambda x: x[3] - self.mezzanine_yield_hurdle},
            {"type": "ineq", "fun": lambda x: self._senior_subordination(x) - self.waterfall_config["min_senior_subordination"]},
        ]

        result = minimize(
            fun=self._issuer_wacc_objective,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100, "ftol": 1e-8, "disp": False},
        )

        x = np.asarray(result.x, dtype=float)
        senior, mezz, senior_coupon, mezz_coupon = x
        equity = self._equity_thickness(x)
        wacc = self._issuer_wacc_objective(x)
        waterfall = self._build_waterfall(senior, mezz, equity, senior_coupon, mezz_coupon)
        waterfall_result = waterfall.run(simulation)
        metrics = self.compute_tranche_metrics(waterfall_result)
        return OptimizationResult(
            senior_thickness=float(senior),
            mezzanine_thickness=float(mezz),
            equity_thickness=float(equity),
            senior_coupon=float(senior_coupon),
            mezzanine_coupon=float(mezz_coupon),
            optimized_wacc=float(wacc),
            success=bool(result.success),
            message=str(result.message),
            tranche_metrics=metrics,
            optimizer_diagnostics={
                "objective_value": float(result.fun),
                "iterations": float(getattr(result, "nit", np.nan)),
                "senior_subordination": float(self._senior_subordination(x)),
            },
        )

    def _issuer_wacc_objective(self, x: np.ndarray) -> float:
        senior, mezz, senior_coupon, mezz_coupon = x
        equity = self._equity_thickness(x)
        if equity <= 0:
            return 1e3
        equity_cost = self.equity_target_irr
        wacc = senior * senior_coupon + mezz * mezz_coupon + equity * equity_cost
        penalty = 0.0
        max_wacc = float(self.optimization_config.get("max_wacc", 1.0))
        if wacc > max_wacc:
            penalty += (wacc - max_wacc) ** 2 * 1_000
        if self._senior_subordination(x) < self.waterfall_config["min_senior_subordination"]:
            penalty += 100.0
        return float(wacc + penalty)

    @staticmethod
    def _equity_thickness(x: np.ndarray) -> float:
        return float(1.0 - x[0] - x[1])

    @staticmethod
    def _senior_subordination(x: np.ndarray) -> float:
        return float(1.0 - x[0])

    def _build_waterfall(
        self,
        senior_thickness: float,
        mezzanine_thickness: float,
        equity_thickness: float,
        senior_coupon: float,
        mezzanine_coupon: float,
    ) -> StructuredWaterfallEngine:
        return StructuredWaterfallEngine(
            issuance_amount=self.issuance_amount,
            horizon_months=self.horizon_months,
            senior_thickness=senior_thickness,
            mezzanine_thickness=mezzanine_thickness,
            equity_thickness=equity_thickness,
            senior_coupon=senior_coupon,
            mezzanine_coupon=mezzanine_coupon,
            senior_servicing_fee_rate=self.waterfall_config["senior_servicing_fee_rate"],
            trustee_fee_rate=self.waterfall_config["trustee_fee_rate"],
            reserve_account_target_pct=self.waterfall_config["reserve_account_target_pct"],
            min_senior_oc_ratio=self.waterfall_config["min_senior_oc_ratio"],
            min_total_oc_ratio=self.waterfall_config["min_total_oc_ratio"],
            min_senior_subordination=self.waterfall_config["min_senior_subordination"],
            excess_spread_trap_trigger_oc_ratio=self.waterfall_config["excess_spread_trap_trigger_oc_ratio"],
            trapped_cash_release_oc_ratio=self.waterfall_config["trapped_cash_release_oc_ratio"],
            pdl_cure_before_equity_distribution=self.waterfall_config["pdl_cure_before_equity_distribution"],
        )

    def compute_tranche_metrics(self, result: WaterfallResult) -> pd.DataFrame:
        rows = []
        for tranche_name, cashflows, initial in [
            ("senior", result.senior_cashflows, result.initial_senior),
            ("mezzanine", result.mezzanine_cashflows, result.initial_mezzanine),
            ("equity", result.equity_cashflows, result.initial_equity),
        ]:
            path_irr = np.array([annualised_irr(np.r_[-initial, row]) for row in cashflows])
            avg_cashflow = cashflows.mean(axis=0)
            duration = macaulay_duration(avg_cashflow, initial, discount_rate=max(np.nanmean(path_irr), 0.0))
            principal_loss = np.maximum(initial - cashflows.sum(axis=1), 0.0)
            var_loss = np.quantile(principal_loss / max(initial, 1e-12), self.confidence_level)
            rows.append(
                {
                    "tranche": tranche_name,
                    "initial_notional": initial,
                    "mean_irr": float(np.nanmean(path_irr)),
                    "median_irr": float(np.nanmedian(path_irr)),
                    "macaulay_duration_years": float(duration),
                    "loss_var": float(var_loss),
                    "expected_cash_multiple": float(cashflows.sum(axis=1).mean() / initial),
                }
            )
        return pd.DataFrame(rows)


def annualised_irr(cashflows: np.ndarray, guess_bounds: tuple[float, float] = (-0.95, 2.00)) -> float:
    """Monthly IRR converted to annual. Bisection is slower than Newton but safer."""
    cf = np.asarray(cashflows, dtype=float)
    if cf.ndim != 1 or len(cf) < 2:
        raise ValueError("cashflows must be a 1D array with an initial outflow and later inflows")
    low, high = guess_bounds

    def npv(rate: float) -> float:
        months = np.arange(len(cf), dtype=float)
        return float(np.sum(cf / (1.0 + rate) ** months))

    f_low, f_high = npv(low), npv(high)
    if np.sign(f_low) == np.sign(f_high):
        return np.nan
    for _ in range(120):
        mid = (low + high) / 2.0
        f_mid = npv(mid)
        if abs(f_mid) < 1e-7:
            monthly = mid
            return float((1.0 + monthly) ** 12 - 1.0)
        if np.sign(f_mid) == np.sign(f_low):
            low, f_low = mid, f_mid
        else:
            high = mid
    monthly = (low + high) / 2.0
    return float((1.0 + monthly) ** 12 - 1.0)


def macaulay_duration(cashflows: np.ndarray, initial_notional: float, discount_rate: float) -> float:
    flows = np.asarray(cashflows, dtype=float)
    monthly_discount = (1.0 + max(discount_rate, 0.0)) ** (1.0 / 12.0) - 1.0
    months = np.arange(1, len(flows) + 1, dtype=float)
    pv = flows / (1.0 + monthly_discount) ** months
    pv_total = float(pv.sum())
    if pv_total <= 0 or initial_notional <= 0:
        return np.nan
    return float(np.sum((months / 12.0) * pv) / pv_total)
