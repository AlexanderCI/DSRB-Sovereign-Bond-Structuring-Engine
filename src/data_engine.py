"""
Synthetic sovereign asset-pool engine for the SDRB deal.

The point is not to pretend these are real DSRB assets. They are synthetic
loan exposures built with the same fields a front-office structuring analyst
would need before running tranche cash flow, credit enhancement, and WACC work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import yaml
from scipy.special import ndtr


@dataclass(frozen=True)
class SimulationResult:
    """Container for path-level asset cash flow and loss outputs."""

    asset_pool: pd.DataFrame
    interest_collections: np.ndarray
    principal_collections: np.ndarray
    recovery_collections: np.ndarray
    gross_losses: np.ndarray
    collateral_balance: np.ndarray
    macro_factors: Dict[str, np.ndarray]
    scenario_name: str

    @property
    def total_collections(self) -> np.ndarray:
        return self.interest_collections + self.principal_collections + self.recovery_collections

    @property
    def net_loss_rate(self) -> np.ndarray:
        start_balance = float(self.asset_pool["principal"].sum())
        if start_balance <= 0:
            raise ValueError("starting collateral balance must be positive")
        return self.gross_losses.sum(axis=1) / start_balance

    def as_monthly_frame(self) -> pd.DataFrame:
        n_paths, n_months = self.total_collections.shape
        months = np.arange(1, n_months + 1)
        return pd.DataFrame(
            {
                "month": np.tile(months, n_paths),
                "path": np.repeat(np.arange(n_paths), n_months),
                "interest_collections": self.interest_collections.reshape(-1),
                "principal_collections": self.principal_collections.reshape(-1),
                "recovery_collections": self.recovery_collections.reshape(-1),
                "gross_losses": self.gross_losses.reshape(-1),
                "collateral_balance": self.collateral_balance.reshape(-1),
            }
        )


class SovereignAssetPool:
    """
    Builds and simulates a diversified pool of sovereign-backed defence and
    resilience loans.

    The class uses monthly hazards, macro-linked default intensities, and a
    one-factor Gaussian copula. It returns aggregate path-level cash flows, not
    a giant loan-by-loan cube, because that is how the downstream waterfall
    engine actually needs to consume the pool.
    """

    def __init__(
        self,
        n_assets: int,
        target_collateral_balance: float,
        horizon_months: int,
        countries: Iterable[str],
        sectors: Iterable[str],
        base_pd_annual_range: tuple[float, float] = (0.0015, 0.02),
        lgd_range: tuple[float, float] = (0.18, 0.46),
        coupon_range: tuple[float, float] = (0.0375, 0.0725),
        maturity_range_months: tuple[int, int] = (36, 120),
        min_asset_principal: float = 25_000_000,
        max_asset_principal: float = 175_000_000,
        inflation_beta_range: tuple[float, float] = (0.35, 1.10),
        regional_vol_beta_range: tuple[float, float] = (0.50, 1.45),
        amortising_share: float = 0.72,
        random_seed: Optional[int] = None,
    ) -> None:
        self.n_assets = int(n_assets)
        self.target_collateral_balance = float(target_collateral_balance)
        self.horizon_months = int(horizon_months)
        self.countries = list(countries)
        self.sectors = list(sectors)
        self.base_pd_annual_range = tuple(map(float, base_pd_annual_range))
        self.lgd_range = tuple(map(float, lgd_range))
        self.coupon_range = tuple(map(float, coupon_range))
        self.maturity_range_months = tuple(map(int, maturity_range_months))
        self.min_asset_principal = float(min_asset_principal)
        self.max_asset_principal = float(max_asset_principal)
        self.inflation_beta_range = tuple(map(float, inflation_beta_range))
        self.regional_vol_beta_range = tuple(map(float, regional_vol_beta_range))
        self.amortising_share = float(amortising_share)
        self.rng = np.random.default_rng(random_seed)
        self._asset_pool: Optional[pd.DataFrame] = None
        self._validate_inputs()

    @classmethod
    def from_config(cls, config_path: str | Path = "config/deal_structure.yaml") -> "SovereignAssetPool":
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        pool = config["asset_pool"]
        deal = config["deal"]
        return cls(
            n_assets=pool["n_assets"],
            target_collateral_balance=deal["target_collateral_balance"],
            horizon_months=deal["horizon_months"],
            countries=pool["countries"],
            sectors=pool["sectors"],
            base_pd_annual_range=tuple(pool["base_pd_annual_range"]),
            lgd_range=tuple(pool["lgd_range"]),
            coupon_range=(pool["min_coupon"], pool["max_coupon"]),
            maturity_range_months=(pool["min_maturity_months"], pool["max_maturity_months"]),
            min_asset_principal=pool["min_asset_principal"],
            max_asset_principal=pool["max_asset_principal"],
            inflation_beta_range=tuple(pool["inflation_beta_range"]),
            regional_vol_beta_range=tuple(pool["regional_vol_beta_range"]),
            amortising_share=pool["amortising_share"],
            random_seed=deal.get("random_seed"),
        )

    def _validate_inputs(self) -> None:
        if self.n_assets < 50:
            raise ValueError("asset pool must contain at least 50 assets for this SDRB model")
        if self.target_collateral_balance <= 0:
            raise ValueError("target collateral balance must be positive")
        if self.horizon_months <= 0:
            raise ValueError("horizon_months must be positive")
        if not self.countries or not self.sectors:
            raise ValueError("countries and sectors cannot be empty")
        if not 0 <= self.amortising_share <= 1:
            raise ValueError("amortising_share must be between 0 and 1")
        for lower, upper, name in [
            (*self.base_pd_annual_range, "base_pd_annual_range"),
            (*self.lgd_range, "lgd_range"),
            (*self.coupon_range, "coupon_range"),
            (*self.maturity_range_months, "maturity_range_months"),
        ]:
            if lower < 0 or upper <= lower:
                raise ValueError(f"bad range for {name}: {(lower, upper)}")

    @property
    def asset_pool(self) -> pd.DataFrame:
        if self._asset_pool is None:
            self._asset_pool = self.generate_asset_pool()
        return self._asset_pool.copy()

    def generate_asset_pool(self) -> pd.DataFrame:
        """Create the static collateral tape used by the stochastic engine."""
        raw_weights = self.rng.dirichlet(np.full(self.n_assets, 1.35))
        principal = raw_weights * self.target_collateral_balance
        principal = np.clip(principal, self.min_asset_principal, self.max_asset_principal)
        principal *= self.target_collateral_balance / principal.sum()

        countries = self.rng.choice(self.countries, size=self.n_assets, replace=True)
        sectors = self.rng.choice(self.sectors, size=self.n_assets, replace=True)
        maturity = self.rng.integers(
            self.maturity_range_months[0],
            self.maturity_range_months[1] + 1,
            size=self.n_assets,
        )
        coupon = self.rng.uniform(self.coupon_range[0], self.coupon_range[1], self.n_assets)
        base_pd = self.rng.uniform(
            self.base_pd_annual_range[0], self.base_pd_annual_range[1], self.n_assets
        )
        lgd = self.rng.uniform(self.lgd_range[0], self.lgd_range[1], self.n_assets)
        inflation_beta = self.rng.uniform(
            self.inflation_beta_range[0], self.inflation_beta_range[1], self.n_assets
        )
        regional_vol_beta = self.rng.uniform(
            self.regional_vol_beta_range[0], self.regional_vol_beta_range[1], self.n_assets
        )
        is_amortising = self.rng.random(self.n_assets) <= self.amortising_share
        implied_rating_score = np.select(
            [base_pd < 0.004, base_pd < 0.009, base_pd < 0.015],
            ["AA-range", "A-range", "BBB-range"],
            default="BB-watch",
        )

        df = pd.DataFrame(
            {
                "asset_id": [f"SDRB-A{idx + 1:03d}" for idx in range(self.n_assets)],
                "country": countries,
                "sector": sectors,
                "principal": principal,
                "coupon_rate": coupon,
                "maturity_months": maturity,
                "base_pd_annual": base_pd,
                "lgd": lgd,
                "inflation_beta": inflation_beta,
                "regional_vol_beta": regional_vol_beta,
                "amortisation_type": np.where(is_amortising, "straight-line", "bullet"),
                "implied_rating_score": implied_rating_score,
            }
        )
        df["wal_months_proxy"] = np.where(
            df["amortisation_type"].eq("straight-line"),
            (df["maturity_months"] + 1) / 2,
            df["maturity_months"],
        )
        return df.sort_values("asset_id").reset_index(drop=True)

    def _simulate_macro_factors(
        self,
        n_paths: int,
        macro_config: Dict[str, float],
        scenario: str,
    ) -> Dict[str, np.ndarray]:
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")
        months = self.horizon_months
        rho = float(macro_config.get("factor_autocorrelation", 0.62))
        if not -0.99 < rho < 0.99:
            raise ValueError("factor_autocorrelation must sit inside (-0.99, 0.99)")

        infl_mean = float(macro_config.get("inflation_gap_mean", 0.0))
        vol_mean = float(macro_config.get("regional_vol_mean", 0.0))
        infl_sigma = float(macro_config.get("inflation_gap_vol", 0.0065))
        vol_sigma = float(macro_config.get("regional_vol_vol", 0.0100))
        if scenario.lower() in {"severe", "stress", "severe_stress"}:
            infl_mean += float(macro_config.get("severe_inflation_shift", 0.0250))
            vol_mean += float(macro_config.get("severe_regional_vol_shift", 0.0400))

        inflation_gap = np.empty((n_paths, months), dtype=float)
        regional_vol = np.empty((n_paths, months), dtype=float)
        inflation_gap[:, 0] = self.rng.normal(infl_mean, infl_sigma, n_paths)
        regional_vol[:, 0] = self.rng.normal(vol_mean, vol_sigma, n_paths)
        scale = np.sqrt(1 - rho**2)
        for month in range(1, months):
            inflation_gap[:, month] = (
                infl_mean + rho * (inflation_gap[:, month - 1] - infl_mean)
                + scale * self.rng.normal(0.0, infl_sigma, n_paths)
            )
            regional_vol[:, month] = (
                vol_mean + rho * (regional_vol[:, month - 1] - vol_mean)
                + scale * self.rng.normal(0.0, vol_sigma, n_paths)
            )
        return {"inflation_gap": inflation_gap, "regional_volatility": regional_vol}

    def simulate_portfolio(
        self,
        n_paths: int,
        macro_config: Dict[str, float],
        scenario: str = "base",
        chunk_size: int = 750,
    ) -> SimulationResult:
        """
        Run the loan pool as path-level monthly collections and losses.

        The implementation chunks Monte Carlo paths so 10,000 trials are usable
        on a normal laptop. Within each chunk, loan mechanics are vectorised by
        path and asset. Monthly default correlation comes from a Gaussian copula:
        idiosyncratic asset normals are blended with a systemic normal draw.
        """
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        assets = self.asset_pool
        principal = assets["principal"].to_numpy(dtype=float)
        coupon = assets["coupon_rate"].to_numpy(dtype=float)
        maturity = assets["maturity_months"].to_numpy(dtype=int)
        base_pd = assets["base_pd_annual"].to_numpy(dtype=float)
        lgd = assets["lgd"].to_numpy(dtype=float)
        inflation_beta = assets["inflation_beta"].to_numpy(dtype=float)
        regional_beta = assets["regional_vol_beta"].to_numpy(dtype=float)
        amortising = assets["amortisation_type"].eq("straight-line").to_numpy()

        months = self.horizon_months
        n_assets = len(assets)
        interest = np.zeros((n_paths, months), dtype=float)
        scheduled_principal = np.zeros((n_paths, months), dtype=float)
        recoveries = np.zeros((n_paths, months), dtype=float)
        losses = np.zeros((n_paths, months), dtype=float)
        collateral_balance = np.zeros((n_paths, months), dtype=float)
        macro_store = self._simulate_macro_factors(n_paths, macro_config, scenario)

        copula_rho = float(macro_config.get("gaussian_copula_rho", 0.34))
        if not 0 <= copula_rho < 0.999:
            raise ValueError("gaussian_copula_rho must be in [0, 0.999)")
        monthly_base_hazard = -np.log1p(-np.clip(base_pd, 1e-9, 0.999)) / 12.0
        copula_scale = np.sqrt(copula_rho)
        idio_scale = np.sqrt(1.0 - copula_rho)

        for start in range(0, n_paths, chunk_size):
            end = min(start + chunk_size, n_paths)
            size = end - start
            outstanding = np.broadcast_to(principal, (size, n_assets)).astype(float).copy()
            alive = np.ones((size, n_assets), dtype=bool)
            system_normals = self.rng.normal(0.0, 1.0, size=(size, months, 1))
            idio_normals = self.rng.normal(0.0, 1.0, size=(size, months, n_assets))
            uniform_draws = self._normal_cdf(copula_scale * system_normals + idio_scale * idio_normals)

            for month in range(months):
                month_index = month + 1
                active = alive & (outstanding > 1e-8) & (month_index <= maturity)
                infl = macro_store["inflation_gap"][start:end, month][:, None]
                reg = macro_store["regional_volatility"][start:end, month][:, None]
                stress_multiplier = np.exp(inflation_beta[None, :] * infl + regional_beta[None, :] * reg)
                monthly_hazard = np.clip(monthly_base_hazard[None, :] * stress_multiplier, 0.0, 0.55)
                default_probability = 1.0 - np.exp(-monthly_hazard)
                defaults = active & (uniform_draws[:, month, :] < default_probability)

                monthly_interest = np.where(active & ~defaults, outstanding * coupon[None, :] / 12.0, 0.0)
                amortising_due = np.where(
                    amortising[None, :],
                    principal[None, :] / np.maximum(maturity[None, :], 1),
                    0.0,
                )
                final_bullet = (~amortising)[None, :] & (month_index == maturity[None, :])
                principal_due = np.where(active & ~defaults, amortising_due, 0.0)
                principal_due = np.where(active & ~defaults & final_bullet, outstanding, principal_due)
                principal_due = np.minimum(principal_due, outstanding)

                default_exposure = np.where(defaults, outstanding, 0.0)
                default_loss = default_exposure * lgd[None, :]
                recovery_cash = default_exposure * (1.0 - lgd[None, :])

                interest[start:end, month] = monthly_interest.sum(axis=1)
                scheduled_principal[start:end, month] = principal_due.sum(axis=1)
                recoveries[start:end, month] = recovery_cash.sum(axis=1)
                losses[start:end, month] = default_loss.sum(axis=1)

                outstanding = np.maximum(outstanding - principal_due - default_exposure, 0.0)
                alive &= ~defaults
                collateral_balance[start:end, month] = outstanding.sum(axis=1)

        return SimulationResult(
            asset_pool=assets,
            interest_collections=interest,
            principal_collections=scheduled_principal,
            recovery_collections=recoveries,
            gross_losses=losses,
            collateral_balance=collateral_balance,
            macro_factors=macro_store,
            scenario_name=scenario,
        )

    @staticmethod
    def _normal_cdf(x: np.ndarray) -> np.ndarray:
        """Vectorised standard normal CDF for Gaussian copula uniforms."""
        return ndtr(x)

    def pool_summary(self) -> pd.DataFrame:
        pool = self.asset_pool
        total = pool["principal"].sum()
        rows = {
            "asset_count": len(pool),
            "collateral_balance": total,
            "weighted_average_coupon": np.average(pool["coupon_rate"], weights=pool["principal"]),
            "weighted_average_pd": np.average(pool["base_pd_annual"], weights=pool["principal"]),
            "weighted_average_lgd": np.average(pool["lgd"], weights=pool["principal"]),
            "weighted_average_life_months_proxy": np.average(pool["wal_months_proxy"], weights=pool["principal"]),
            "largest_asset_pct": pool["principal"].max() / total,
        }
        return pd.DataFrame([rows])


def load_deal_config(config_path: str | Path = "config/deal_structure.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"could not find config file: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
