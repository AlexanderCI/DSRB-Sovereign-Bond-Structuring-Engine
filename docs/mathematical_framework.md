# Mathematical framework

This file documents the quantitative backbone of the SDRB 2026-1 engine. It is written for the actual modelling work, not for a glossy investor deck. The model has four blocks: asset cashflows, stochastic default simulation, waterfall allocation, and capital stack optimisation.

## 1. Asset pool notation

Let there be \(N\) sovereign-backed loans and \(T\) monthly periods. Asset \(i\) has:

\[
P_i = \text{initial principal}, \qquad c_i = \text{annual coupon}, \qquad M_i = \text{maturity in months}
\]

The asset pool vector is:

\[
\mathbf{P}= (P_1, P_2,\ldots,P_N)^\top
\]

and the total collateral balance at closing is:

\[
C_0=\sum_{i=1}^{N}P_i
\]

For SDRB 2026-1, the target collateral pool is larger than funded notes:

\[
C_0 > L_0
\]

where \(L_0\) is the funded note balance. The difference is the overcollateralisation cushion:

\[
OC_0=C_0-L_0
\]

The OC ratio is:

\[
OCR_t=\frac{C_t}{L_t}
\]

where \(C_t\) is collateral outstanding and \(L_t\) is note balance outstanding.

## 2. Asset cashflow vectorisation

For month \(t\), scheduled interest on asset \(i\) is:

\[
I_{i,t}=\mathbb{1}_{i,t}^{alive}\frac{c_i}{12}B_{i,t-1}
\]

where \(B_{i,t-1}\) is outstanding principal before month \(t\), and \(\mathbb{1}_{i,t}^{alive}\) indicates the asset has not defaulted.

For a straight-line amortising loan:

\[
A_{i,t}=\mathbb{1}_{i,t}^{alive}\min\left(B_{i,t-1},\frac{P_i}{M_i}\right)
\]

For a bullet loan:

\[
A_{i,t}=\begin{cases}
B_{i,t-1}, & t=M_i \\
0, & t<M_i
\end{cases}
\]

The path-level portfolio collections are vectorised as:

\[
I_t=\sum_{i=1}^{N}I_{i,t}, \qquad A_t=\sum_{i=1}^{N}A_{i,t}, \qquad R_t=\sum_{i=1}^{N}R_{i,t}
\]

Total cash available before waterfall rules is:

\[
X_t=I_t+A_t+R_t
\]

## 3. Time-varying sovereign default intensity

Each asset has a base annual probability of default \(PD_i^{ann}\). It is converted into a monthly base hazard:

\[
\lambda_{i,0}=\frac{-\ln(1-PD_i^{ann})}{12}
\]

The model uses two macro factors:

- \(\pi_t\): inflation gap
- \(v_t\): regional volatility / escalation factor

The time-varying monthly hazard is:

\[
\lambda_{i,t}=\lambda_{i,0}\exp\left(\beta_i^{\pi}\pi_t+\beta_i^v v_t\right)
\]

Then the monthly conditional default probability is:

\[
q_{i,t}=1-\exp(-\lambda_{i,t})
\]

This is basically the actuarial survival model idea, moved into sovereign structured credit. The model is not trying to forecast countries. It is trying to stress a tranche structure.

## 4. Macro factor process

Each macro factor follows a simple AR(1) process:

\[
\pi_t=\mu_{\pi}+\phi(\pi_{t-1}-\mu_{\pi})+\sqrt{1-\phi^2}\sigma_{\pi}\epsilon_{\pi,t}
\]

\[
v_t=\mu_v+\phi(v_{t-1}-\mu_v)+\sqrt{1-\phi^2}\sigma_v\epsilon_{v,t}
\]

In the severe stress case, the means are shifted upward:

\[
\mu_{\pi}^{stress}=\mu_{\pi}+s_{\pi}, \qquad \mu_{v}^{stress}=\mu_v+s_v
\]

So severe stress is not a different model. It is the same model with worse macro state variables.

## 5. Gaussian copula for joint sovereign default

The model uses a one-factor Gaussian copula. For each path, month, and asset:

\[
Z_{i,t}=\sqrt{\rho}S_t+\sqrt{1-\rho}\epsilon_{i,t}
\]

where:

\[
S_t\sim N(0,1), \qquad \epsilon_{i,t}\sim N(0,1)
\]

The transformed uniform is:

\[
U_{i,t}=\Phi(Z_{i,t})
\]

The asset defaults in month \(t\) if:

\[
U_{i,t}<q_{i,t}
\]

This gives correlated default clustering without building a full country-by-country macroeconomic model. That is the point. We need tail dependence pressure on the tranche stack, not a political forecast essay.

## 6. LGD and recoveries

Each asset has loss given default \(LGD_i\). If default happens in month \(t\), the gross loss is:

\[
L_{i,t}=LGD_i B_{i,t-1}
\]

and recovery is:

\[
R_{i,t}=(1-LGD_i)B_{i,t-1}
\]

Path-level gross portfolio loss is:

\[
L_t=\sum_{i=1}^{N}L_{i,t}
\]

The current engine books recovery in the default month. That keeps the waterfall tight and easier to inspect. A delayed recovery curve can be added later by shifting \(R_{i,t}\) into a recovery lag vector.

## 7. Tranche structure

Let the funded note structure be:

\[
L_0=L_0^S+L_0^M+L_0^E
\]

where:

- \(L_0^S\): senior tranche
- \(L_0^M\): mezzanine tranche
- \(L_0^E\): equity / first-loss tranche

Senior subordination is:

\[
Sub_t^S=\frac{L_t^M+L_t^E+Reserve_t+TrappedCash_t}{L_t^S+L_t^M+L_t^E}
\]

This is the protection below the senior tranche before the DSRB guarantee is considered.

## 8. Loss allocation

Losses are allocated bottom-up:

\[
\Delta PDL_t^E=\min(L_t,L_{t-1}^E)
\]

\[
\Delta PDL_t^M=\min\left((L_t-\Delta PDL_t^E)^+,L_{t-1}^M\right)
\]

\[
\Delta PDL_t^S=\min\left((L_t-\Delta PDL_t^E-\Delta PDL_t^M)^+,L_{t-1}^S\right)
\]

where \(x^+=\max(x,0)\).

Principal Deficiency Ledgers are tracked by tranche:

\[
PDL_t^k=PDL_{t-1}^k+\Delta PDL_t^k-Cure_t^k, \qquad k\in\{S,M,E\}
\]

PDLs block or trap equity distributions. This is important because without PDL discipline, excess spread can leak out while credit support is actually deteriorating.

## 9. Priority of payments

Monthly cash \(X_t\) is paid in this order:

1. senior servicing and trustee fees
2. senior interest
3. mezzanine interest
4. senior PDL cure
5. mezzanine PDL cure
6. senior principal amortisation
7. mezzanine principal amortisation
8. reserve account top-up
9. trapped cash if OC or subordination tests fail
10. equity PDL cure, if allowed
11. equity excess spread distribution

The simplified waterfall equation is:

\[
X_t=Fees_t+Int_t^S+Int_t^M+Prin_t^S+Prin_t^M+Trap_t+Dist_t^E+Residual_t
\]

with each term constrained by available cash and current tranche balance.

## 10. Credit enhancement traps

Excess spread is trapped when credit enhancement weakens. The main tests are:

\[
OCR_t^S=\frac{C_t}{L_t^S}<OCR_{min}^S
\]

\[
OCR_t^{Total}=\frac{C_t}{L_t^S+L_t^M}<OCR_{min}^{Total}
\]

\[
Sub_t^S<Sub_{min}^S
\]

or when senior/mezzanine PDLs remain outstanding.

Trapped cash can later be released only if the structure cures above the release trigger and PDLs are cleared.

## 11. DSRB first-loss guarantee pricing

The senior guarantee has a cap:

\[
G_{cap}=\alpha_G L_0^S
\]

For path \(p\), discounted senior loss is:

\[
PVLoss_p^S=\sum_{t=1}^{T}\frac{\Delta PDL_{p,t}^S}{(1+r/12)^t}
\]

Guarantee draw is capped:

\[
Draw_p=\min(PVLoss_p^S,G_{cap})
\]

Expected guarantee draw is:

\[
E[Draw]=\frac{1}{P}\sum_{p=1}^{P}Draw_p
\]

The fair annual guarantee fee is:

\[
Fee_{fair}=\frac{E[Draw]}{G_{cap}\sum_{t=1}^{T}(1+r/12)^{-t}/12}
\]

The loaded fee is:

\[
Fee_{loaded}=Fee_{fair}+CapitalLoad+AdminMargin
\]

This is an expected-loss price, not a rating-agency capital model. It is the correct first model for a portfolio project because it ties directly to simulated losses and tranche protection.

## 12. WACC optimisation

The issuer WACC is:

\[
WACC=w_Sc_S+w_Mc_M+w_Er_E
\]

where:

- \(w_S,w_M,w_E\) are tranche thicknesses
- \(c_S,c_M\) are senior and mezzanine coupons
- \(r_E\) is the equity target IRR / cost of first-loss capital

The optimisation problem is:

\[
\min_{w_S,w_M,c_S,c_M} WACC
\]

subject to:

\[
w_S+w_M+w_E=1
\]

\[
w_E^{min}\leq w_E\leq w_E^{max}
\]

\[
c_S\geq h_S, \qquad c_M\geq h_M
\]

\[
1-w_S\geq Sub_{min}^{S}
\]

where \(h_S\) and \(h_M\) are institutional yield hurdles.

## 13. IRR, duration, and equity VaR

For each tranche, the internal rate of return solves:

\[
0=-N_0+\sum_{t=1}^{T}\frac{CF_t}{(1+r_m)^t}
\]

and annualises as:

\[
r_{ann}=(1+r_m)^{12}-1
\]

Macaulay duration is:

\[
D=\frac{\sum_{t=1}^{T}\frac{t}{12}\frac{CF_t}{(1+y/12)^t}}{\sum_{t=1}^{T}\frac{CF_t}{(1+y/12)^t}}
\]

Equity VaR is measured on path-level equity loss rate:

\[
VaR_{\alpha}^{E}=Q_{\alpha}\left(\frac{Loss_p^E}{L_0^E}\right)
\]

The equity tranche is where the model gets unforgiving. Tiny changes in default clustering, recoveries, or OC trigger timing can change equity IRR heavily. That is realistic.
