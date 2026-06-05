from src.credit_enhancement import DSRBGuaranteePricer
from src.data_engine import SovereignAssetPool, load_deal_config
from src.structural_waterfall import StructuredWaterfallEngine


def test_guarantee_pricer_returns_fee_outputs():
    config = load_deal_config("config/deal_structure.yaml")
    pool = SovereignAssetPool.from_config("config/deal_structure.yaml")
    sim = pool.simulate_portfolio(n_paths=20, macro_config=config["macro"], scenario="base", chunk_size=10)
    wf = StructuredWaterfallEngine.from_config("config/deal_structure.yaml")
    wf_result = wf.run(sim)
    pricer = DSRBGuaranteePricer.from_config("config/deal_structure.yaml")
    result = pricer.price_from_waterfall(wf_result)
    assert result.guarantee_notional > 0
    assert result.loaded_fee_bps >= result.fair_fee_bps
    assert result.overcollateralization_ratio > 1.0
