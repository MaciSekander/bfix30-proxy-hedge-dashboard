# BFIX 30-Minute Proxy Hedge Dashboard

An interactive Bloomberg BQuant dashboard for evaluating and sizing cross-currency proxy hedges with 30-minute Bloomberg FX Fixing (BFIX) data.

The dashboard reconstructs an aligned intraday BFIX panel, ranks potential hedge instruments, estimates hedge ratios using OLS, Newey-West/HAC, or feasible GLS, analyzes cointegration and spread mean reversion, and runs a causal walk-forward z-score backtest with a trade blotter and risk controls.

> **Environment requirement:** This project is designed for Bloomberg BQuant. It uses Bloomberg's `bql` package and requires an authenticated Bloomberg environment.

## Project Objective

The dashboard is designed to answer five practical questions:

1. Which available currency is the best statistical proxy for the target exposure?
2. How much hedge notional should be used for a given dollar target notional?
3. Is the hedge ratio stable through time?
4. Is the target-versus-hedge price spread unusually rich or cheap?
5. Would a z-score mean-reversion strategy have performed out of sample without look-ahead bias?

The project separates two related but different models:

- **Return regression:** estimates the beta used to size the hedge.
- **Log-price regression:** constructs the cointegration spread used for z-score and mean-reversion analysis.

## Main Features

- Reconstructs all 48 daily BFIX observations from `00:00` through `23:30`.
- Supports major, emerging-market, and Asian currencies plus XAU/USD BFIX.
- Aligns instruments to a common 30-minute timestamp grid.
- Removes future BFIX slots using a New York publication cutoff.
- Reports missing observations, coverage, first timestamp, and latest timestamp.
- Supports log returns and percentage returns.
- Optionally interpolates one isolated missing 30-minute price.
- Winsorizes extreme returns to reduce the influence of bad prints.
- Produces a return-correlation matrix and rolling pair correlation.
- Ranks proxy hedge candidates using correlation, spread z-score, and cointegration evidence.
- Estimates hedge beta using OLS, OLS with Newey-West/HAC inference, or feasible GLS with AR(1) residual covariance.
- Calculates beta standard errors, t-statistics, and GLS residual persistence.
- Converts target notional into a recommended hedge notional.
- Charts rolling beta stability over a user-selected calendar-day horizon.
- Estimates spread half-life in 30-minute bars, hours, and BFIX days.
- Displays rolling spread mean, standard deviation, bands, spread level, and z-score.
- Compares historical hedged and unhedged P&L.
- Runs a strictly out-of-sample walk-forward z-score backtest.
- Records entries, exits, prices, z-scores, holding periods, P&L, and exit reasons in a trade blotter.
- Includes a dollar stop-loss, maximum holding period, and signal rearming logic.

## Repository Structure

```text
.
|-- README.md
`-- outputs/
    |-- bfix30_dv01_cointegration_dashboard.py
    `-- bfix30_dv01_hedge_dashboard_explainer.md
```

The main application is:

```text
outputs/bfix30_dv01_cointegration_dashboard.py
```

## Requirements

The script imports:

```text
numpy
pandas
plotly
ipywidgets
IPython
statsmodels
bql
```

`statsmodels` is used for the Engle-Granger cointegration and ADF tests. If it is unavailable, the rest of the dashboard can run, but those p-values will be blank.

## Running the Dashboard

1. Open a Bloomberg BQuant notebook.
2. Upload `outputs/bfix30_dv01_cointegration_dashboard.py` to the project.
3. Run the script from a notebook cell or paste its contents into a cell.
4. Select the instruments and history window.
5. Click **Reload BFIX panel** after changing data-loading controls.
6. Select the target, hedge, estimation method, and risk parameters.

For an uploaded file, a BQuant notebook can generally run it with:

```python
%run outputs/bfix30_dv01_cointegration_dashboard.py
```

The script creates a BQL service automatically if the notebook does not already contain one:

```python
bq = bql.Service()
```

## BFIX Data Construction

Bloomberg stores each BFIX time slot as a separate security. The dashboard requests 48 tickers per selected instrument and reconstructs them into one time series.

Examples:

```text
EUR F000 Curncy  -> EUR BFIX at 00:00
EUR F003 Curncy  -> EUR BFIX at 00:30
EUR F010 Curncy  -> EUR BFIX at 01:00
XAU F233 Curncy  -> XAU/USD BFIX at 23:30
```

In this project:

```text
1 bar = 1 scheduled 30-minute BFIX observation
48 bars = 1 complete BFIX day
```

The timestamps are interpreted in `America/New_York`. A five-minute publication delay prevents a future scheduled slot from being included before it should be available.

## Return Construction

The default return method is the log return:

```math
r_t = \ln(P_t) - \ln(P_{t-1})
```

The alternative percentage return is:

```math
r_t = \frac{P_t}{P_{t-1}} - 1
```

Winsorization clips returns that are an extreme number of robust standard deviations from their center. This is a data-quality defense against isolated bad observations, not a replacement for reviewing suspicious Bloomberg data.

## Hedge-Ratio Model

For target return \(r_{T,t}\) and hedge return \(r_{H,t}\), the return regression is:

```math
r_{T,t} = \alpha_r + \beta_r r_{H,t} + \varepsilon_t
```

The slope \(\beta_r\) measures the estimated change in the target return associated with a one-unit change in the hedge return. The recommended hedge notional is:

```math
N_H = -N_T\beta_r
```

where:

- \(N_T\) is the user-entered target notional.
- \(N_H\) is the recommended signed hedge notional.
- A negative value means taking the opposite direction in the hedge instrument.

### Dollar-Notional Convention

The dashboard treats the target notional as a dollar-denominated exposure and reports the hedge in the same dollar-notional convention. It does **not** mean that one unit of EUR, GBP, or XAU equals one US dollar.

The current model sizes a proxy hedge from return sensitivity. It does not calculate instrument-level rates DV01 from cash flows, yields, duration, or curve risk. A true bond or swap DV01 workflow would require an additional conversion from the recommended dollar risk exposure into the native instrument quantity.

## Beta Estimation Methods

### OLS

Ordinary least squares estimates the intercept and return beta by minimizing squared residuals. Its classical standard error assumes well-behaved residuals.

### OLS with Newey-West/HAC

This method uses the same OLS beta but replaces the classical covariance estimate with a heteroskedasticity-and-autocorrelation-consistent standard error.

The **NW lags** setting determines how many lagged residual cross-products are included in the covariance estimate. It affects the beta standard error and t-statistic, not the OLS beta itself.

### Feasible GLS with AR(1) Omega

The GLS method first estimates OLS residuals and then estimates residual persistence:

```math
\hat{\rho} =
\frac{\sum_{t=2}^{T}\hat{\varepsilon}_t\hat{\varepsilon}_{t-1}}
{\sum_{t=2}^{T}\hat{\varepsilon}_{t-1}^{2}}
```

This \(\hat{\rho}\) defines an AR(1)-structured error covariance matrix. The model applies a Prais-Winsten transformation and solves the transformed regression using QR decomposition.

The displayed t-statistic is:

```math
t_{\beta} = \frac{\hat{\beta}}{SE(\hat{\beta})}
```

A large absolute t-statistic indicates that the estimated beta is large relative to its estimated uncertainty. It does not prove that the relationship will remain stable.

## Rolling Beta Stability

The rolling beta chart estimates return beta over a trailing calendar-day window:

```math
\beta_t = \frac{\operatorname{Cov}(r_T,r_H)_t}
{\operatorname{Var}(r_H)_t}
```

For example, a 30-day horizon uses the preceding 30 calendar days at each timestamp. Partial startup windows are suppressed, so a displayed 30-day beta represents a complete 30-day history.

A comparatively stable rolling beta supports more consistent hedge sizing. Large jumps, sign changes, or persistent drift suggest that the hedge relationship may be unstable.

## Cointegration Spread

The price model is separate from the return-beta model:

```math
\ln(P_{T,t}) = \alpha_p + \beta_p\ln(P_{H,t}) + s_t
```

The reconstructed spread is:

```math
s_t = \ln(P_{T,t}) - \alpha_p - \beta_p\ln(P_{H,t})
```

Here:

- \(\alpha_p\) is the fitted log-price intercept.
- \(\beta_p\) is the fitted log-price beta.
- \(s_t\) is the deviation not explained by the fitted long-run relationship.

The price beta constructs the spread. The return beta sizes the hedge. They answer different questions and should not be substituted for one another.

## Rolling Mean, Standard Deviation, and Z-Score

For a lookback window of \(W\) bars:

```math
\mu_t = \frac{1}{W}\sum_{i=t-W+1}^{t}s_i
```

```math
\sigma_t =
\sqrt{\frac{1}{W-1}\sum_{i=t-W+1}^{t}(s_i-\mu_t)^2}
```

```math
z_t = \frac{s_t-\mu_t}{\sigma_t}
```

Interpretation:

- A positive z-score means the target is rich relative to the fitted hedge relationship.
- A negative z-score means the target is cheap relative to the fitted hedge relationship.
- The absolute z-score measures how unusual the current spread is relative to its recent history.

A z-score is a standardized diagnostic, not a probability guarantee. Its usefulness depends on the spread remaining reasonably stable and mean reverting.

## Cointegration Tests

The dashboard reports:

- **Cointegration p-value:** Engle-Granger test of whether the two log-price series share a stationary linear combination.
- **ADF p-value:** Augmented Dickey-Fuller test applied to the fitted spread.

Smaller p-values provide stronger evidence against the relevant unit-root null hypothesis. They are evidence, not proof, and can be unstable across samples.

## Spread Half-Life

The dashboard estimates:

```math
\Delta s_t = c + \lambda s_{t-1} + u_t
```

When \(\lambda < 0\), the estimated half-life is:

```math
H = \frac{-\ln(2)}{\lambda}
```

The half-life estimates how many bars are required for half of a spread deviation to disappear. It is not the time required for the spread to reach exactly zero.

With 30-minute bars, a seven-bar half-life equals approximately 3.5 hours.

## Historical Hedge Diagnostic

The selected-pair hedge diagnostic compares:

```math
P\&L^{unhedged}_t = N_T r_{T,t}
```

```math
P\&L^{hedged}_t = N_T r_{T,t} + N_H r_{H,t}
```

It reports per-bar volatility, volatility reduction, cumulative P&L, worst bar, and hedged maximum drawdown.

Maximum drawdown is the worst decline from a previous cumulative-P&L peak:

```math
MDD = \min_t\left(C_t - \max_{\tau\leq t}C_\tau\right)
```

This diagnostic uses one beta over the selected sample. It is useful for hedge comparison, but it should not be interpreted as a causal trading backtest.

## Walk-Forward Z-Score Backtest

The walk-forward test is designed to avoid look-ahead bias.

At timestamp \(t\):

1. The model uses only observations strictly before \(t\).
2. Return beta and price-spread parameters are re-estimated from the trailing training window.
3. The spread mean and standard deviation use the selected trailing z-score window.
4. A signal is generated at \(t\).
5. The position begins at \(t+1\).
6. The entry hedge beta remains fixed for that trade.

The observations before the first complete training window are calibration history, not out-of-sample test observations.

### Entry Rules

For entry threshold \(z_{entry}>0\):

```text
z >= +z_entry  -> short target / long hedge
z <= -z_entry  -> long target / short hedge
```

### Mean-Reversion Exit

For exit band \(z_{exit}\):

```text
Long-spread trade exits when z >= -z_exit
Short-spread trade exits when z <= +z_exit
```

The default entry threshold is `2.0`; the default exit band is `0.25`.

### Dollar Stop-Loss

The strategy tracks cumulative P&L independently for each open trade:

```math
P\&L_{trade,t} =
\sum_{\tau=entry}^{t}
\left(N_{T,\tau}r_{T,\tau}+N_{H,\tau}r_{H,\tau}\right)
```

If the cumulative trade P&L reaches the negative user-entered loss limit, the exit reason is recorded as `DOLLAR_STOP_LOSS`.

The stop is evaluated after a completed 30-minute bar, and the causal exit is applied on the next bar. A fast move can therefore produce a realized loss larger than the selected threshold.

### Maximum Holding Period

The maximum holding period is a duration stop. If the spread does not mean revert within the selected number of active bars, the trade exits with `MAX_HOLDING_PERIOD`.

For example:

```text
48 bars x 30 minutes = 24 hours
```

This treats a slow or failed convergence as a potentially stale setup.

### Signal Rearming

After a dollar stop or maximum-holding exit, the strategy is disarmed. It stays flat until the spread returns inside the normal exit band:

```math
|z_t| \leq z_{exit}
```

Only after normalization can a new move through the entry threshold create another trade. This prevents the backtest from repeatedly entering the same still-extreme, failed spread cycle.

### Trade Blotter

Each trade includes:

- Signal and execution timestamps
- Direction
- Entry and exit z-scores
- Spread levels
- Target and hedge BFIX levels
- Bars held
- Trade P&L
- Open or closed status
- Exit reason: `MEAN_REVERSION`, `DOLLAR_STOP_LOSS`, or `MAX_HOLDING_PERIOD`

## Dashboard Controls

| Control | Purpose |
|---|---|
| Instruments | BFIX currencies or XAU to request and compare |
| History | Bloomberg history requested for the panel |
| Returns | Log-return or percentage-return construction |
| Interpolate gap | Fills one isolated internal missing price |
| Winsor z | Clips extreme returns beyond the selected robust threshold |
| Corr window | Trailing bars used for latest correlation |
| Spread z | Trailing spread observations used for z-score normalization |
| Walk-forward train | Prior observations used to refit each out-of-sample model |
| Beta horizon days | Calendar-day window used in the rolling beta chart |
| Beta method | OLS, OLS with HAC inference, or feasible GLS |
| NW lags | Autocorrelation lags used by the HAC covariance estimate |
| Target notional | Dollar-denominated target exposure |
| Signal \|z\| | Absolute z-score required to enter |
| Exit band | Normalization region used to exit and rearm |
| Max loss ($) | Maximum permitted cumulative loss per trade |
| Max holding bars | Maximum active 30-minute bars per trade |

## Important Limitations

- BFIX is a fixing dataset, not an executable streaming market price.
- The backtest does not currently model bid-ask spreads, transaction costs, slippage, market impact, financing, or liquidity constraints.
- A 30-minute stop can gap beyond its configured dollar threshold.
- Statistical relationships can break during regime changes.
- Correlation does not imply cointegration, and cointegration does not guarantee profitable convergence.
- P-values and half-life estimates are sample dependent.
- Ranked candidate scores are a screening heuristic, not an optimization objective or trading recommendation.
- The code assumes the selected BFIX levels and notional conventions are appropriate for the intended exposure.
- This project is for research and educational use, not investment advice.

## Suggested Next Steps

- Add transaction costs and bid-ask assumptions to the walk-forward P&L.
- Add volatility-scaled or risk-budgeted position sizing.
- Add regime filters for unstable beta, failed cointegration, or unusually long half-life.
- Export the walk-forward path and blotter to CSV.
- Add parameter-sensitivity tables evaluated only on separate validation samples.
- Add unit tests for ticker construction, timestamp alignment, signal transitions, and trade accounting.

## License

No license is currently included. Add a license before distributing or accepting external contributions.

