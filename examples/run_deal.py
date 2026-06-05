from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.credit_enhancement import DSRBGuaranteePricer, guarantee_summary_frame
from src.data_engine import SovereignAssetPool, load_deal_config
from src.optimization_engine import CapitalStructureOptimizer
from src.structural_waterfall import StructuredWaterfallEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SDRB 2026-1 structuring engine.")
    parser.add_argument("--config", default="config/deal_structure.yaml")
    parser.add_argument("--paths", type=int, default=500)
    parser.add_argument("--scenario", choices=["base", "severe"], default="base")
    parser.add_argument("--chunk-size", type=int, default=500)
    args = parser.parse_args()

    config = load_deal_config(args.config)
    pool = SovereignAssetPool.from_config(args.config)
    waterfall = StructuredWaterfallEngine.from_config(args.config)
    pricer = DSRBGuaranteePricer.from_config(args.config)
    optimizer = CapitalStructureOptimizer.from_config(args.config)

    simulation = pool.simulate_portfolio(
        n_paths=args.paths,
        macro_config=config["macro"],
        scenario=args.scenario,
        chunk_size=args.chunk_size,
    )
    wf_result = waterfall.run(simulation)
    guarantee = pricer.price_from_waterfall(wf_result)
    opt = optimizer.optimise(simulation)

    print("\nPOOL SUMMARY")
    print(pool.pool_summary().to_string(index=False))

    print("\nGUARANTEE PRICE")
    print(guarantee_summary_frame(guarantee).to_string(index=False))

    print("\nBASE CAPITAL STACK METRICS")
    print(optimizer.compute_tranche_metrics(wf_result).to_string(index=False))

    print("\nOPTIMISED STACK")
    print(
        {
            "senior_thickness": round(opt.senior_thickness, 4),
            "mezzanine_thickness": round(opt.mezzanine_thickness, 4),
            "equity_thickness": round(opt.equity_thickness, 4),
            "senior_coupon": round(opt.senior_coupon, 4),
            "mezzanine_coupon": round(opt.mezzanine_coupon, 4),
            "wacc": round(opt.optimized_wacc, 5),
            "success": opt.success,
        }
    )

    print("\nOPTIMISED TRANCHE METRICS")
    print(opt.tranche_metrics.to_string(index=False))

    if not wf_result.trigger_breaches.empty:
        print("\nTRIGGER BREACH SUMMARY")
        print(wf_result.trigger_breaches.head(12).to_string(index=False))
    else:
        print("\nTRIGGER BREACH SUMMARY")
        print("No OC / trap breaches in this run.")


if __name__ == "__main__":
    main()
