"""
Monthly SDRB priority-of-payments engine.

The waterfall is deliberately strict. It separates collateral collections,
allocates credit losses through the capital stack, runs PDL ledgers, tests OC
triggers, traps excess spread when credit enhancement weakens, and produces
path-level tranche cash flows for pricing or optimisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

from src.data_engine import SimulationResult


@dataclass(frozen=True)
class WaterfallResult:
    senior_cashflows: np.ndarray
    mezzanine_cashflows: np.ndarray
    equity_cashflows: np.ndarray
    senior_balance: np.ndarray
    mezzanine_balance: np.ndarray
    equity_balance: np.ndarray
    senior_pdl: np.ndarray
    mezzanine_pdl: np.ndarray
    equity_pdl: np.ndarray
    reserve_account: np.ndarray
    trapped_cash: np.ndarray
    oc_ratio_senior: np.ndarray
    oc_ratio_total: np.ndarray
    trigger_breaches: pd.DataFrame
    payment_ledger: pd.DataFrame
    initial_senior: float
    initial_mezzanine: float
    initial_equity: float
    initial_collateral: float

    @property
    def note_cashflows(self) -> Dict[str, np.ndarray]:
        return {
            "senior": self.senior_cashflows,
            "mezzanine": self.mezzanine_cashflows,
            "equity": self.equity_cashflows,
        }

    def tranche_loss_rates(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "senior_loss_rate": self.senior_pdl[:, -1] / max(self.initial_senior, 1e-12),
                "mezzanine_loss_rate": self.mezzanine_pdl[:, -1] / max(self.initial_mezzanine, 1e-12),
                "equity_loss_rate": self.equity_pdl[:, -1] / max(self.initial_equity, 1e-12),
            }
        )


class StructuredWaterfallEngine:
    """Runs SDRB monthly cash flow allocation across senior, mezz, and equity."""

    def __init__(
        self,
        issuance_amount: float,
        horizon_months: int,
        senior_thickness: float,
        mezzanine_thickness: float,
        equity_thickness: float,
        senior_coupon: float,
        mezzanine_coupon: float,
        senior_servicing_fee_rate: float = 0.0012,
        trustee_fee_rate: float = 0.0003,
        reserve_account_target_pct: float = 0.0125,
        min_senior_oc_ratio: float = 1.215,
        min_total_oc_ratio: float = 1.060,
        min_senior_subordination: float = 0.185,
        excess_spread_trap_trigger_oc_ratio: float = 1.075,
        trapped_cash_release_oc_ratio: float = 1.110,
        pdl_cure_before_equity_distribution: bool = True,
    ) -> None:
        self.issuance_amount = float(issuance_amount)
        self.horizon_months = int(horizon_months)
        self.senior_thickness = float(senior_thickness)
        self.mezzanine_thickness = float(mezzanine_thickness)
        self.equity_thickness = float(equity_thickness)
        self.senior_coupon = float(senior_coupon)
        self.mezzanine_coupon = float(mezzanine_coupon)
        self.senior_servicing_fee_rate = float(senior_servicing_fee_rate)
        self.trustee_fee_rate = float(trustee_fee_rate)
        self.reserve_account_target_pct = float(reserve_account_target_pct)
        self.min_senior_oc_ratio = float(min_senior_oc_ratio)
        self.min_total_oc_ratio = float(min_total_oc_ratio)
        self.min_senior_subordination = float(min_senior_subordination)
        self.excess_spread_trap_trigger_oc_ratio = float(excess_spread_trap_trigger_oc_ratio)
        self.trapped_cash_release_oc_ratio = float(trapped_cash_release_oc_ratio)
        self.pdl_cure_before_equity_distribution = bool(pdl_cure_before_equity_distribution)
        self._validate()

    @classmethod
    def from_config(cls, config_path: str | Path = "config/deal_structure.yaml") -> "StructuredWaterfallEngine":
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        tranches = config["tranches"]
        wf = config["waterfall"]
        return cls(
            issuance_amount=config["deal"]["issuance_amount"],
            horizon_months=config["deal"]["horizon_months"],
            senior_thickness=tranches["senior"]["initial_thickness"],
            mezzanine_thickness=tranches["mezzanine"]["initial_thickness"],
            equity_thickness=tranches["equity"]["initial_thickness"],
            senior_coupon=tranches["senior"]["coupon_rate"],
            mezzanine_coupon=tranches["mezzanine"]["coupon_rate"],
            senior_servicing_fee_rate=wf["senior_servicing_fee_rate"],
            trustee_fee_rate=wf["trustee_fee_rate"],
            reserve_account_target_pct=wf["reserve_account_target_pct"],
            min_senior_oc_ratio=wf["min_senior_oc_ratio"],
            min_total_oc_ratio=wf["min_total_oc_ratio"],
            min_senior_subordination=wf["min_senior_subordination"],
            excess_spread_trap_trigger_oc_ratio=wf["excess_spread_trap_trigger_oc_ratio"],
            trapped_cash_release_oc_ratio=wf["trapped_cash_release_oc_ratio"],
            pdl_cure_before_equity_distribution=wf["pdl_cure_before_equity_distribution"],
        )

    def _validate(self) -> None:
        if self.issuance_amount <= 0:
            raise ValueError("issuance_amount must be positive")
        if self.horizon_months <= 0:
            raise ValueError("horizon_months must be positive")
        stack_sum = self.senior_thickness + self.mezzanine_thickness + self.equity_thickness
        if not np.isclose(stack_sum, 1.0, atol=1e-8):
            raise ValueError(f"tranche thicknesses must sum to 1.0, got {stack_sum:.8f}")
        if min(self.senior_thickness, self.mezzanine_thickness, self.equity_thickness) <= 0:
            raise ValueError("all tranche thicknesses must be positive")
        if min(self.senior_coupon, self.mezzanine_coupon) < 0:
            raise ValueError("coupons cannot be negative")

    @property
    def initial_balances(self) -> Dict[str, float]:
        return {
            "senior": self.issuance_amount * self.senior_thickness,
            "mezzanine": self.issuance_amount * self.mezzanine_thickness,
            "equity": self.issuance_amount * self.equity_thickness,
        }

    def run(self, simulation: SimulationResult) -> WaterfallResult:
        interest = np.asarray(simulation.interest_collections, dtype=float)
        principal = np.asarray(simulation.principal_collections, dtype=float)
        recoveries = np.asarray(simulation.recovery_collections, dtype=float)
        gross_losses = np.asarray(simulation.gross_losses, dtype=float)
        collateral_balance = np.asarray(simulation.collateral_balance, dtype=float)
        self._validate_simulation_arrays(interest, principal, recoveries, gross_losses, collateral_balance)

        n_paths, months = interest.shape
        init = self.initial_balances
        senior_bal = np.full(n_paths, init["senior"], dtype=float)
        mezz_bal = np.full(n_paths, init["mezzanine"], dtype=float)
        equity_bal = np.full(n_paths, init["equity"], dtype=float)
        senior_pdl = np.zeros(n_paths, dtype=float)
        mezz_pdl = np.zeros(n_paths, dtype=float)
        equity_pdl = np.zeros(n_paths, dtype=float)
        reserve = np.zeros(n_paths, dtype=float)
        trapped_cash = np.zeros(n_paths, dtype=float)

        senior_cf = np.zeros((n_paths, months), dtype=float)
        mezz_cf = np.zeros((n_paths, months), dtype=float)
        equity_cf = np.zeros((n_paths, months), dtype=float)
        senior_bal_hist = np.zeros((n_paths, months), dtype=float)
        mezz_bal_hist = np.zeros((n_paths, months), dtype=float)
        equity_bal_hist = np.zeros((n_paths, months), dtype=float)
        senior_pdl_hist = np.zeros((n_paths, months), dtype=float)
        mezz_pdl_hist = np.zeros((n_paths, months), dtype=float)
        equity_pdl_hist = np.zeros((n_paths, months), dtype=float)
        reserve_hist = np.zeros((n_paths, months), dtype=float)
        trapped_hist = np.zeros((n_paths, months), dtype=float)
        senior_oc_hist = np.zeros((n_paths, months), dtype=float)
        total_oc_hist = np.zeros((n_paths, months), dtype=float)
        fee_paid_hist = np.zeros((n_paths, months), dtype=float)
        senior_interest_paid_hist = np.zeros((n_paths, months), dtype=float)
        mezz_interest_paid_hist = np.zeros((n_paths, months), dtype=float)
        senior_principal_paid_hist = np.zeros((n_paths, months), dtype=float)
        mezz_principal_paid_hist = np.zeros((n_paths, months), dtype=float)
        equity_dist_hist = np.zeros((n_paths, months), dtype=float)
        breach_rows: list[dict[str, float | int]] = []

        reserve_target = self.reserve_account_target_pct * self.issuance_amount
        total_note_start = init["senior"] + init["mezzanine"]

        for month in range(months):
            available = interest[:, month] + principal[:, month] + recoveries[:, month]
            loss = gross_losses[:, month]

            equity_loss = np.minimum(loss, equity_bal)
            equity_bal -= equity_loss
            equity_pdl += equity_loss
            remaining_loss = loss - equity_loss

            mezz_loss = np.minimum(remaining_loss, mezz_bal)
            mezz_bal -= mezz_loss
            mezz_pdl += mezz_loss
            remaining_loss -= mezz_loss

            senior_loss = np.minimum(remaining_loss, senior_bal)
            senior_bal -= senior_loss
            senior_pdl += senior_loss

            fee_due = (self.senior_servicing_fee_rate + self.trustee_fee_rate) / 12.0 * np.maximum(
                collateral_balance[:, month], senior_bal + mezz_bal
            )
            fee_paid = np.minimum(available, fee_due)
            available -= fee_paid

            senior_interest_due = senior_bal * self.senior_coupon / 12.0
            senior_interest_paid = np.minimum(available, senior_interest_due)
            available -= senior_interest_paid
            senior_interest_shortfall = senior_interest_due - senior_interest_paid
            senior_pdl += senior_interest_shortfall

            mezz_interest_due = mezz_bal * self.mezzanine_coupon / 12.0
            mezz_interest_paid = np.minimum(available, mezz_interest_due)
            available -= mezz_interest_paid
            mezz_interest_shortfall = mezz_interest_due - mezz_interest_paid
            mezz_pdl += mezz_interest_shortfall

            total_notes_now = np.maximum(senior_bal + mezz_bal, 1e-9)
            senior_oc = collateral_balance[:, month] / np.maximum(senior_bal, 1e-9)
            total_oc = collateral_balance[:, month] / total_notes_now
            senior_subordination = (mezz_bal + equity_bal + reserve + trapped_cash) / np.maximum(
                senior_bal + mezz_bal + equity_bal, 1e-9
            )
            trap_active = (
                (total_oc < self.excess_spread_trap_trigger_oc_ratio)
                | (senior_oc < self.min_senior_oc_ratio)
                | (senior_subordination < self.min_senior_subordination)
                | (senior_pdl > 1e-8)
                | (mezz_pdl > 1e-8)
            )

            cure_senior_pdl = np.minimum(available, senior_pdl)
            senior_pdl -= cure_senior_pdl
            available -= cure_senior_pdl
            senior_bal += cure_senior_pdl

            cure_mezz_pdl = np.minimum(available, mezz_pdl)
            mezz_pdl -= cure_mezz_pdl
            available -= cure_mezz_pdl
            mezz_bal += cure_mezz_pdl

            senior_target_balance = np.minimum(
                senior_bal,
                np.maximum(collateral_balance[:, month] / self.min_senior_oc_ratio, 0.0),
            )
            senior_principal_due = np.maximum(senior_bal - senior_target_balance, 0.0)
            senior_principal_paid = np.minimum(available, senior_principal_due)
            senior_bal -= senior_principal_paid
            available -= senior_principal_paid

            total_target_notes = np.minimum(
                senior_bal + mezz_bal,
                np.maximum(collateral_balance[:, month] / self.min_total_oc_ratio, 0.0),
            )
            mezz_target_balance = np.maximum(total_target_notes - senior_bal, 0.0)
            mezz_principal_due = np.maximum(mezz_bal - mezz_target_balance, 0.0)
            mezz_principal_paid = np.minimum(available, mezz_principal_due)
            mezz_bal -= mezz_principal_paid
            available -= mezz_principal_paid

            reserve_top_up = np.minimum(available, np.maximum(reserve_target - reserve, 0.0))
            reserve += reserve_top_up
            available -= reserve_top_up

            forced_trap = np.where(trap_active, available, 0.0)
            trapped_cash += forced_trap
            available -= forced_trap

            release_allowed = (total_oc > self.trapped_cash_release_oc_ratio) & (senior_pdl < 1e-8) & (mezz_pdl < 1e-8)
            release_cash = np.where(release_allowed, trapped_cash, 0.0)
            trapped_cash -= release_cash
            available += release_cash

            if self.pdl_cure_before_equity_distribution:
                cure_equity_pdl = np.minimum(available, equity_pdl)
                equity_pdl -= cure_equity_pdl
                equity_bal += cure_equity_pdl
                available -= cure_equity_pdl

            equity_distribution = np.maximum(available, 0.0)
            available -= equity_distribution

            senior_cf[:, month] = fee_paid * 0.0 + senior_interest_paid + senior_principal_paid + cure_senior_pdl
            mezz_cf[:, month] = mezz_interest_paid + mezz_principal_paid + cure_mezz_pdl
            equity_cf[:, month] = equity_distribution
            senior_bal_hist[:, month] = senior_bal
            mezz_bal_hist[:, month] = mezz_bal
            equity_bal_hist[:, month] = equity_bal
            senior_pdl_hist[:, month] = senior_pdl
            mezz_pdl_hist[:, month] = mezz_pdl
            equity_pdl_hist[:, month] = equity_pdl
            reserve_hist[:, month] = reserve
            trapped_hist[:, month] = trapped_cash
            senior_oc_hist[:, month] = senior_oc
            total_oc_hist[:, month] = total_oc
            fee_paid_hist[:, month] = fee_paid
            senior_interest_paid_hist[:, month] = senior_interest_paid
            mezz_interest_paid_hist[:, month] = mezz_interest_paid
            senior_principal_paid_hist[:, month] = senior_principal_paid
            mezz_principal_paid_hist[:, month] = mezz_principal_paid
            equity_dist_hist[:, month] = equity_distribution

            breach_mask = (senior_oc < self.min_senior_oc_ratio) | (total_oc < self.min_total_oc_ratio) | trap_active
            breach_count = int(breach_mask.sum())
            if breach_count:
                breach_rows.append(
                    {
                        "month": month + 1,
                        "breach_paths": breach_count,
                        "breach_rate": breach_count / n_paths,
                        "avg_senior_oc": float(np.mean(senior_oc)),
                        "avg_total_oc": float(np.mean(total_oc)),
                    }
                )

        months_idx = np.tile(np.arange(1, months + 1), n_paths)
        paths_idx = np.repeat(np.arange(n_paths), months)
        payment_ledger = pd.DataFrame(
            {
                "path": paths_idx,
                "month": months_idx,
                "fees_paid": fee_paid_hist.reshape(-1),
                "senior_interest_paid": senior_interest_paid_hist.reshape(-1),
                "mezzanine_interest_paid": mezz_interest_paid_hist.reshape(-1),
                "senior_principal_paid": senior_principal_paid_hist.reshape(-1),
                "mezzanine_principal_paid": mezz_principal_paid_hist.reshape(-1),
                "equity_distribution": equity_dist_hist.reshape(-1),
                "reserve_account": reserve_hist.reshape(-1),
                "trapped_cash": trapped_hist.reshape(-1),
                "senior_oc_ratio": senior_oc_hist.reshape(-1),
                "total_oc_ratio": total_oc_hist.reshape(-1),
            }
        )

        return WaterfallResult(
            senior_cashflows=senior_cf,
            mezzanine_cashflows=mezz_cf,
            equity_cashflows=equity_cf,
            senior_balance=senior_bal_hist,
            mezzanine_balance=mezz_bal_hist,
            equity_balance=equity_bal_hist,
            senior_pdl=senior_pdl_hist,
            mezzanine_pdl=mezz_pdl_hist,
            equity_pdl=equity_pdl_hist,
            reserve_account=reserve_hist,
            trapped_cash=trapped_hist,
            oc_ratio_senior=senior_oc_hist,
            oc_ratio_total=total_oc_hist,
            trigger_breaches=pd.DataFrame(breach_rows),
            payment_ledger=payment_ledger,
            initial_senior=init["senior"],
            initial_mezzanine=init["mezzanine"],
            initial_equity=init["equity"],
            initial_collateral=float(simulation.asset_pool["principal"].sum()),
        )

    def _validate_simulation_arrays(self, *arrays: np.ndarray) -> None:
        shape = arrays[0].shape
        if len(shape) != 2:
            raise ValueError("simulation arrays must be 2D: paths x months")
        if shape[1] != self.horizon_months:
            raise ValueError(
                f"simulation horizon {shape[1]} does not match waterfall horizon {self.horizon_months}"
            )
        for arr in arrays:
            if arr.shape != shape:
                raise ValueError("all simulation arrays must have identical shapes")
            if not np.all(np.isfinite(arr)):
                raise ValueError("simulation arrays contain NaN or infinite values")
            if np.any(arr < -1e-8):
                raise ValueError("simulation arrays cannot contain negative cash flow drivers")


def build_waterfall_from_dict(config: Dict[str, Any]) -> StructuredWaterfallEngine:
    tranches = config["tranches"]
    wf = config["waterfall"]
    return StructuredWaterfallEngine(
        issuance_amount=config["deal"]["issuance_amount"],
        horizon_months=config["deal"]["horizon_months"],
        senior_thickness=tranches["senior"]["initial_thickness"],
        mezzanine_thickness=tranches["mezzanine"]["initial_thickness"],
        equity_thickness=tranches["equity"]["initial_thickness"],
        senior_coupon=tranches["senior"]["coupon_rate"],
        mezzanine_coupon=tranches["mezzanine"]["coupon_rate"],
        senior_servicing_fee_rate=wf["senior_servicing_fee_rate"],
        trustee_fee_rate=wf["trustee_fee_rate"],
        reserve_account_target_pct=wf["reserve_account_target_pct"],
        min_senior_oc_ratio=wf["min_senior_oc_ratio"],
        min_total_oc_ratio=wf["min_total_oc_ratio"],
        min_senior_subordination=wf["min_senior_subordination"],
        excess_spread_trap_trigger_oc_ratio=wf["excess_spread_trap_trigger_oc_ratio"],
        trapped_cash_release_oc_ratio=wf["trapped_cash_release_oc_ratio"],
        pdl_cure_before_equity_distribution=wf["pdl_cure_before_equity_distribution"],
    )
