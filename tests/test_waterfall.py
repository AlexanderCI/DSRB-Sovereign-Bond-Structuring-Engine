from src.data_engine import SovereignAssetPool, load_deal_config
from src.structural_waterfall import StructuredWaterfallEngine


def test_waterfall_runs_and_tracks_ledgers():
    config = load_deal_config("config/deal_structure.yaml")
    pool = SovereignAssetPool.from_config("config/deal_structure.yaml")
    sim = pool.simulate_portfolio(n_paths=15, macro_config=config["macro"], scenario="severe", chunk_size=5)
    wf = StructuredWaterfallEngine.from_config("config/deal_structure.yaml")
    result = wf.run(sim)
    assert result.senior_cashflows.shape == sim.interest_collections.shape
    assert result.mezzanine_cashflows.shape == sim.interest_collections.shape
    assert result.equity_cashflows.shape == sim.interest_collections.shape
    assert result.senior_pdl.min() >= 0
    assert result.mezzanine_pdl.min() >= 0
    assert result.equity_pdl.min() >= 0
    assert not result.payment_ledger.empty
