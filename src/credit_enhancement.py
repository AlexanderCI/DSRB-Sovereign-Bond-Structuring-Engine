"""
DSRB credit enhancement and guarantee pricing.

This module prices a first-loss guarantee around the senior tranche using Monte
Carlo expected loss, senior tail loss, and a fee load for capital usage. It also
tracks the simple overcollateralisation account created by issuing fewer notes
than collateral assets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml

from src.data_engine import SimulationResult, SovereignAssetPool
from src.structural_waterfall import StructuredWaterfallEngine, WaterfallResult


@dataclass(frozen=True)
class GuaranteePricingResult:
    guarantee_notional: float
    expected_senior_loss: float
    expected_guarantee_draw: float
    probability_of_draw: float
    senior_loss_p99: float
    senior_loss_p999: float
    fair_fee_bps: float
    loaded_fee_bps: float
    implied_wrapped_expected_loss_bps: float
    overcollateralization_amount: float
    overcollateralization_ratio: float
    diagnostics: pd.DataFrame


class OverCollateralizationAccount:
    """Simple OC account that measures excess collateral over funded notes."""

    def __init__(self, collateral_balance: float, note_balance: float) -> None:
        if collateral_balance <= 0 or note_balance <= 0:
            raise ValueError("collateral_balance and note_balance must be positive")
        self.collateral_balance = float(collateral_balance)
        self.note_balance = float(note_balance)

    @property
    def amount(self) -> float:
        return max(self.collateral_balance - self.note_balance, 0.0)

    @property
    def ratio(self) -> float:
        return self.collateral_balance / self.note_balance


class DSRBGuaranteePricer:
    """
    Prices a senior-tranche DSRB first-loss guarantee.

    The guarantee pays senior losses up to a cap. The price is expressed as an
    annual fee on guaranteed notional: expected discounted draw divided by the
    present-value annuity of the guarantee notional, plus capital/admin loads.
    """

    def __init__(
        self,
        guarantee_cap_pct_of_senior: float,
        target_senior_expected_loss_bps: float,
        capital_charge_bps: float,
        admin_margin_bps: float,
        discount_rate: float,
    ) -> None:
        self.guarantee_cap_pct_of_senior = float(guarantee_cap_pct_of_senior)
        self.target_senior_expected_loss_bps = float(target_senior_expected_loss_bps)
        self.capital_charge_bps = float(capital_charge_bps)
        self.admin_margin_bps = float(admin_margin_bps)
        self.discount_rate = float(discount_rate)
        self._validate()

    @classmethod
    def from_config(cls, config_path: str | Path = "config/deal_structure.yaml") -> "DSRBGuaranteePricer":
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        ce = config["credit_enhancement"]
        return cls(
            guarantee_cap_pct_of_senior=ce["dsrb_senior_guarantee_cap_pct_of_senior"],
            target_senior_expected_loss_bps=ce["target_senior_expected_loss_bps"],
            capital_charge_bps=ce["capital_charge_bps"],
            admin_margin_bps=ce["admin_margin_bps"],
            discount_rate=config["deal"]["discount_rate"],
        )

    def _validate(self) -> None:
        if not 0 < self.guarantee_cap_pct_of_senior <= 1:
            raise ValueError("guarantee cap must be in (0, 1]")
        if self.discount_rate < 0:
            raise ValueError("discount rate cannot be negative")
        if min(self.target_senior_expected_loss_bps, self.capital_charge_bps, self.admin_margin_bps) < 0:
            raise ValueError("fee and target expected loss values cannot be negative")

    def price_from_waterfall(self, result: WaterfallResult) -> GuaranteePricingResult:
        monthly_discount = (1.0 + self.discount_rate) ** (1.0 / 12.0) - 1.0
        discount_vector = 1.0 / (1.0 + monthly_discount) ** np.arange(1, result.senior_pdl.shape[1] + 1)

        senior_periodic_loss = np.diff(
            np.column_stack([np.zeros(result.senior_pdl.shape[0]), result.senior_pdl]), axis=1
        )
        senior_periodic_loss = np.maximum(senior_periodic_loss, 0.0)
        pv_senior_loss = senior_periodic_loss @ discount_vector
        senior_terminal_loss = result.senior_pdl[:, -1]

        guarantee_notional = result.initial_senior * self.guarantee_cap_pct_of_senior
        guarantee_draw = np.minimum(pv_senior_loss, guarantee_notional)
        expected_senior_loss = float(pv_senior_loss.mean())
        expected_draw = float(guarantee_draw.mean())
        probability_draw = float(np.mean(guarantee_draw > 1e-8))

        annuity_pv = guarantee_notional * float(discount_vector.sum()) / 12.0
        fair_fee_bps = 0.0 if annuity_pv <= 0 else expected_draw / annuity_pv * 10_000.0
        loaded_fee_bps = fair_fee_bps + self.capital_charge_bps + self.admin_margin_bps
        wrapped_expected_loss = np.maximum(pv_senior_loss - guarantee_notional, 0.0)
        implied_wrapped_el_bps = float(wrapped_expected_loss.mean() / result.initial_senior * 10_000.0)

        oc = OverCollateralizationAccount(result.initial_collateral, result.initial_senior + result.initial_mezzanine)
        diagnostics = pd.DataFrame(
            {
                "metric": [
                    "mean_pv_senior_loss",
                    "mean_guarantee_draw",
                    "probability_of_guarantee_draw",
                    "senior_terminal_loss_p99",
                    "senior_terminal_loss_p999",
                    "target_wrapped_senior_el_bps",
                    "implied_wrapped_senior_el_bps",
                ],
                "value": [
                    expected_senior_loss,
                    expected_draw,
                    probability_draw,
                    float(np.quantile(senior_terminal_loss, 0.99)),
                    float(np.quantile(senior_terminal_loss, 0.999)),
                    self.target_senior_expected_loss_bps,
                    implied_wrapped_el_bps,
                ],
            }
        )
        return GuaranteePricingResult(
            guarantee_notional=guarantee_notional,
            expected_senior_loss=expected_senior_loss,
            expected_guarantee_draw=expected_draw,
            probability_of_draw=probability_draw,
            senior_loss_p99=float(np.quantile(senior_terminal_loss, 0.99)),
            senior_loss_p999=float(np.quantile(senior_terminal_loss, 0.999)),
            fair_fee_bps=float(fair_fee_bps),
            loaded_fee_bps=float(loaded_fee_bps),
            implied_wrapped_expected_loss_bps=implied_wrapped_el_bps,
            overcollateralization_amount=oc.amount,
            overcollateralization_ratio=oc.ratio,
            diagnostics=diagnostics,
        )

    def run_monte_carlo_pricing(
        self,
        asset_pool: SovereignAssetPool,
        waterfall_engine: StructuredWaterfallEngine,
        macro_config: Dict[str, float],
        n_trials: int = 10_000,
        scenario: str = "base",
        chunk_size: int = 750,
    ) -> tuple[SimulationResult, WaterfallResult, GuaranteePricingResult]:
        if n_trials < 1_000:
            raise ValueError("use at least 1,000 trials for a guarantee price that is not noisy")
        simulation = asset_pool.simulate_portfolio(
            n_paths=n_trials,
            macro_config=macro_config,
            scenario=scenario,
            chunk_size=chunk_size,
        )
        waterfall = waterfall_engine.run(simulation)
        guarantee = self.price_from_waterfall(waterfall)
        return simulation, waterfall, guarantee


def guarantee_summary_frame(result: GuaranteePricingResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "guarantee_notional": result.guarantee_notional,
                "fair_fee_bps": result.fair_fee_bps,
                "loaded_fee_bps": result.loaded_fee_bps,
                "probability_of_draw": result.probability_of_draw,
                "senior_loss_p99": result.senior_loss_p99,
                "wrapped_expected_loss_bps": result.implied_wrapped_expected_loss_bps,
                "oc_amount": result.overcollateralization_amount,
                "oc_ratio": result.overcollateralization_ratio,
            }
        ]
    )
