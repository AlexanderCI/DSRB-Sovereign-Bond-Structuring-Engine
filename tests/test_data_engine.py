from src.data_engine import SovereignAssetPool, load_deal_config


def test_asset_pool_generates_50_plus_assets():
    config = load_deal_config("config/deal_structure.yaml")
    pool = SovereignAssetPool.from_config("config/deal_structure.yaml")
    assets = pool.asset_pool
    assert len(assets) >= 50
    assert abs(assets["principal"].sum() - config["deal"]["target_collateral_balance"]) < 1e-2
    assert assets["base_pd_annual"].between(0, 1).all()
    assert assets["lgd"].between(0, 1).all()


def test_simulation_shapes_are_valid():
    config = load_deal_config("config/deal_structure.yaml")
    pool = SovereignAssetPool.from_config("config/deal_structure.yaml")
    sim = pool.simulate_portfolio(n_paths=25, macro_config=config["macro"], scenario="base", chunk_size=10)
    assert sim.interest_collections.shape == (25, config["deal"]["horizon_months"])
    assert sim.principal_collections.shape == sim.interest_collections.shape
    assert sim.recovery_collections.shape == sim.interest_collections.shape
    assert sim.gross_losses.shape == sim.interest_collections.shape
    assert sim.total_collections.min() >= 0
