# DSRB Sovereign Defence and Resilience Bond Structuring Engine

Hi everyone! In this repo, I built a synthetic front-office structuring "engine" for a **USD 5.0bn Sovereign Defence and Resilience Bond**, or **SDRB 2026-1**. The deal is written assuming a multilateral bank like the DSRB was arranging capital for allied sovereign defence, cyber, logistics, or dual-use infrastructure assets.

Obviously, this is not an official DSRB model, and is a purely synthetic deal model for testing tranche sizing, as well as guarantee pricing, waterfall behaviour, and issuer WACC under stress.

## Main summary

**Transaction:** SDRB 2026-1  
**Issuer concept:** multilateral-backed sovereign defence and resilience funding vehicle  
**Issue size:** USD 5.0bn  
**Collateral target:** USD 5.35bn synthetic sovereign-backed loans  
**Collateral type:** any defence logistics sites, dual-use infrastructure, cyber-resilience grids, secure data centres, allied supply-chain facilities, and satellite ground-station assets  
**Capital stack:** senior notes, mezzanine notes, and first-loss equity / excess spread notes  
**Credit enhancement:** overcollateralisation, excess spread trapping, reserve account, PDL mechanics, and a DSRB-style first-loss guarantee around the senior tranche  
**Main output:** tranche cashflows, expected loss, IRR, duration, equity VaR, guarantee fee, OC ratios, and trigger breach diagnostics

The basic idea is quite simple: pool sovereign-backed assets, then simulate defaults under macro stress, push the cash through a strict securitization waterfall, then optimize tranche thickness and coupons so the issuer funds itself cheaply without destroying the senior credit story.

## What the model is trying to do

DSRB is meant to sit between sovereign security needs and private capital markets. That makes the finance problem kind of unusual as it is not just treasury, and it is not just front-office DCM either. The actual job is to translate government-backed defence/resilience assets into something investors can EASILY buy and price and risk-manage, and syndicate.

So this repo is built around that exact idea/"bridge":

- **Treasury view:** funding cost, liquidity facility, OC account, WACC, note liability profile.
- **Risk view:** PD/LGD, macro shocks, correlated sovereign defaults, PDLs, tail loss.
- **Front-office structuring view:** tranche attachment, subordination, excess spread, senior guarantee fee, investor hurdle yields.
- **Actuarial/statistical view:** stochastic cashflow simulation, default intensity, Monte Carlo expected loss, VaR, duration.

## Flowchart of my work

```text
config/deal_structure.yaml
        |
        v
src/data_engine.py
SovereignAssetPool
- synthetic collateral tape
- monthly default intensities
- inflation / regional volatility shocks
- Gaussian copula joint sovereign default simulation
        |
        v
src/structural_waterfall.py
StructuredWaterfallEngine
- senior fees
- senior interest
- mezzanine interest
- senior principal amortisation
- mezzanine principal amortisation
- equity excess spread
- PDLs, OC tests, trapped cash
        |
        v
src/credit_enhancement.py
DSRBGuaranteePricer
- senior first-loss guarantee cap
- expected discounted guarantee draw
- fair fee and loaded fee in bps
- OC account diagnostics
        |
        v
src/optimization_engine.py
CapitalStructureOptimizer
- senior / mezz / equity thickness
- coupon optimisation
- issuer WACC minimisation
- tranche IRR, duration, equity VaR
        |
        v
notebooks/deal_structuring_sandbox.ipynb
- base vs severe stress cashflow plots
- tranche loss distributions
- guarantee fee and WACC output
```

## Mathematical highlights

The engine models asset cashflows by month. Each sovereign-backed loan has principal, coupon, maturity, base annual PD, LGD, country, sector, and macro sensitivity.

The monthly default intensity is linked to macro factors:

```math
\lambda_{i,t}=\lambda_{i,0}\exp(\beta_i^{\pi}\pi_t+\beta_i^{v}v_t)
```

where inflation gap and regional volatility shift the default intensity. Monthly default probabilities are then:

```math
q_{i,t}=1-\exp(-\lambda_{i,t})
```

Joint sovereign default behaviour is produced with a Gaussian copula:

```math
Z_{i,t}=\sqrt{\rho}S_t+\sqrt{1-\rho}\epsilon_{i,t}
```

The waterfall allocates losses bottom-up and cashflows top-down. Equity absorbs first loss, then mezzanine, then senior. Cash gets paid through fees, senior interest, mezzanine interest, senior principal, mezzanine principal, and only then equity excess spread.

## The repo layout

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── config/
│   └── deal_structure.yaml
├── docs/
│   ├── mathematical_framework.md
│   └── dsrb_alignment_brief.md
├── examples/
│   └── run_deal.py
├── notebooks/
│   └── deal_structuring_sandbox.ipynb
├── src/
│   ├── __init__.py
│   ├── data_engine.py
│   ├── structural_waterfall.py
│   ├── credit_enhancement.py
│   └── optimization_engine.py
└── tests/
    ├── test_data_engine.py
    ├── test_waterfall.py
    └── test_credit_enhancement.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the tests:

```bash
pytest -q
```

Run a small deal simulation:

```bash
python examples/run_deal.py --paths 500 --scenario base
```

Run a severe stress:

```bash
python examples/run_deal.py --paths 500 --scenario severe
```

For the full institutional run, use 10,000 paths:

```bash
python examples/run_deal.py --paths 10000 --scenario severe
```

## Main modules

### `src/data_engine.py`

Creates a synthetic pool of 50+ sovereign-backed assets. It generates asset-level coupons, maturities, annual PDs, LGDs, country/sector tags, and macro sensitivities. It then simulates monthly interest collections, principal collections, recoveries, gross losses, and collateral balance.

### `src/structural_waterfall.py`

Runs the payment waterfall. This is the securitisation engine. It tracks senior and mezzanine balances, equity first-loss position, PDLs, reserve account, trapped cash, senior OC ratio, total OC ratio, and trigger breaches.

### `src/credit_enhancement.py`

Prices a DSRB first-loss guarantee around the senior tranche. It uses expected discounted senior losses from Monte Carlo paths, caps the guarantee draw, computes a fair fee in basis points, and then adds capital/admin loading.

### `src/optimization_engine.py`

Optimises tranche thickness and coupons using `scipy.optimize`. The objective is issuer WACC, with constraints for equity thickness, senior subordination, and investor yield hurdles. It reports IRR, Macaulay duration, and VaR by tranche.

## Outputs worth looking at

The example runner prints:

- asset pool summary
- guarantee pricing table
- tranche IRR / duration / VaR table
- optimiser result
- trigger breach summary

The notebook adds charts for:

- base vs severe stress portfolio cashflows
- collateral balance roll-down
- senior / mezz / equity loss distribution
- equity tranche VaR

## The "realism" of my work

It goes without saying that the asset tape is synthetic, but the actual mechanics are meant to be taken serious. My model uses real structured-finance ideas like excess spread, OC ratio, PDL, subordination, first-loss wrap, tranche WAL/duration, investor hurdles, and, of course, issuer WACC.

There are still judgement calls worth considering. For example, recoveries are assumed to arrive in the same month as default, which is conservative for timing analysis only in some structures and generous in others. A REAL bank model would also have legal maturity, or perhaps delayed recovery curves, or rating-agency stresses, and country concentration caps, sanctions screens, defence procurement eligibility rules, and investor-by-investor syndication limits.
