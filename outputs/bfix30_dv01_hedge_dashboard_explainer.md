# BFIX 30-Minute Proxy Hedge Dashboard Explainer

## 1. Product Goal

The goal of this dashboard is to help a trader answer a practical hedge question:

```text
I have a notional exposure in one currency.
Can I hedge it with another currency that behaves similarly?
If yes, how much hedge notional should I use?
And has that hedge historically reduced risk?
```

The product is not just a correlation dashboard. It is a proxy hedge decision tool.

It combines:

- 30-minute BFIX data
- return correlation
- beta hedge sizing
- cointegration spread diagnostics
- rich/cheap z-score interpretation
- backtested hedge performance
- z-score signal backtesting

The intended workflow is:

```text
1. Load high-frequency BFIX data.
2. Select a target currency.
3. Input the trader's target notional.
4. Rank possible proxy hedges.
5. Use beta to calculate recommended hedge notional.
6. Check whether the hedge historically reduced P&L volatility.
7. Use spread z-score and cointegration tests to judge rich/cheap timing.
```

---

## 2. Why BFIX 30-Minute Data Matters

The original dashboard used BFIX fixing data, but the important enhancement is that we reconstruct a full 30-minute time series.

Bloomberg BFIX does not arrive as one simple intraday time series. Instead, each fixing time has its own Bloomberg ticker.

For example:

```text
EUR F000 Curncy  -> EUR BFIX at 00:00
EUR F003 Curncy  -> EUR BFIX at 00:30
EUR F010 Curncy  -> EUR BFIX at 01:00
EUR F013 Curncy  -> EUR BFIX at 01:30
...
EUR F233 Curncy  -> EUR BFIX at 23:30
```

The dashboard requests all 48 daily BFIX fixing tickers for each selected currency.

There are 48 half-hour slots in a full day:

```text
00:00
00:30
01:00
01:30
...
23:00
23:30
```

So for one currency and one day, the target panel is:

```text
2026-07-28 00:00
2026-07-28 00:30
2026-07-28 01:00
...
2026-07-28 23:30
```

For 10 days, the theoretical maximum number of price bars is:

```text
10 days x 48 bars per day = 480 price bars
```

For 20 days:

```text
20 days x 48 bars per day = 960 price bars
```

This gives many more observations than daily data, which is useful for rolling hedge analytics.

---

## 3. What A Bar Means

In this dashboard:

```text
1 bar = one 30-minute BFIX observation
```

Price bars look like this:

```text
2026-07-28 00:00  EUR price
2026-07-28 00:30  EUR price
2026-07-28 01:00  EUR price
2026-07-28 01:30  EUR price
...
```

Return bars are calculated between adjacent price timestamps.

Example:

```text
00:30 return = move from 00:00 to 00:30
01:00 return = move from 00:30 to 01:00
01:30 return = move from 01:00 to 01:30
```

So if there are 480 price bars, there are usually 479 return bars.

Most analytics in the dashboard use 30-minute return bars.

---

## 4. Data Panel Controls

The left control panel determines what data is loaded from Bloomberg.

### Currencies

This is the multi-select list:

```python
bfix30_matrix_currencies
```

If the trader selects:

```text
EUR, GBP, CHF, AUD, CAD
```

the dashboard requests:

```text
5 currencies x 48 BFIX fixing times = 240 Bloomberg securities
```

Each selected currency becomes one column in the reconstructed BFIX price panel.

### History

This controls how much historical BFIX data is requested.

Examples:

```text
5 days
10 days
20 days
1 month
3 months
6 months
1 year
```

Longer history gives more stable correlation and beta estimates, but it also increases request size and can make the dashboard slower.

### Return Method

The dashboard supports two return calculations.

Log returns:

```python
returns = np.log(prices).diff()
```

Percent returns:

```python
returns = prices.pct_change()
```

For statistical work, log returns are usually cleaner because they are additive over time.

### Interpolate Single Missing Gap

If checked, the dashboard fills one missing 30-minute gap inside the price series.

This is useful when Bloomberg returns a small hole in the BFIX panel.

It does not fill long missing stretches.

### Winsor Z

Winsorization clips extreme returns.

This means the dashboard caps unusually large 30-minute returns so one bad print does not dominate correlation, beta, or the backtest.

Example:

```text
Raw returns:
-0.0004, 0.0002, 0.0001, 0.0003, 0.0150

After clipping:
-0.0004, 0.0002, 0.0001, 0.0003, 0.0012
```

The extreme observation is not deleted. It is capped.

Interpretation:

```text
Winsor z = 0      no clipping
Winsor z = 8      only very extreme observations are clipped
Winsor z = 4      more aggressive clipping
```

For this product, `8.0` is a reasonable default because it protects the analytics from bad BFIX prints without over-smoothing normal market moves.

---

## 5. Proxy Hedge Controls

The right control panel controls the actual hedge analytics.

### Target Currency

This is the currency exposure the trader wants to hedge.

Example:

```text
Target = NZD
```

### Hedge Currency

This is the proxy currency being tested as the hedge.

Example:

```text
Hedge = AUD
```

### Correlation Window

This controls the rolling correlation lookback.

The window is measured in 30-minute bars.

Examples:

```text
48 bars  = 1 BFIX day
96 bars  = 2 BFIX days
252 bars = 5.25 BFIX days
480 bars = 10 BFIX days
```

If the rolling window is 96 bars, the dashboard asks:

```text
Over the most recent 96 half-hour BFIX returns,
how correlated were the target and hedge currencies?
```

### Spread Z Window

This controls the lookback used to calculate the cointegration spread z-score.

If the spread z window is 252 bars, the dashboard calculates the recent average and volatility of the spread over:

```text
252 x 30 minutes = about 5.25 BFIX days
```

Then it asks:

```text
How far is the current spread from its recent average?
```

### Target Notional

This is the main trader input.

Example:

```text
Target notional = 100,000,000
```

The dashboard then uses beta to calculate the recommended hedge notional.

---

## 6. Return Beta

The dashboard calculates beta using target and hedge 30-minute returns.

The formula is:

```text
beta = covariance(target returns, hedge returns) / variance(hedge returns)
```

This is the same slope as a simple one-factor OLS regression:

```text
target return = intercept + beta x hedge return + error
```

The code implements the formula directly instead of calling a formal regression package.

Interpretation:

```text
beta = 0.75
```

means:

```text
When the hedge currency moves 1 unit,
the target currency has historically moved about 0.75 units.
```

This beta is used for hedge sizing.

---

## 7. Beta Method: OLS, HAC/Newey-West, Or GLS

The dashboard now lets the trader choose how the return beta is estimated.

### OLS

OLS estimates:

```text
target return = intercept + beta x hedge return + error
```

The OLS beta is the slope that minimizes the sum of squared regression errors:

```text
minimize sum((target return - intercept - beta x hedge return)^2)
```

This treats each 30-minute return observation as independent for the usual beta uncertainty calculation.

### OLS plus HAC/Newey-West

HAC means heteroskedasticity-and-autocorrelation-consistent. Newey-West allows the regression errors to have:

```text
different variance across bars
serial correlation across nearby bars
```

The point estimate of beta remains the OLS beta. The change is the uncertainty estimate around it.

The dashboard reports:

```text
return beta
beta SE
beta t-stat
NW lags
```

The Newey-West lag parameter controls how many nearby return lags are included in the robust covariance estimate. With 4 lags on 30-minute data, the uncertainty estimate accounts for dependence across approximately the next two hours of bars.

Therefore:

```text
HAC beta = OLS beta
HAC SE   = serial-correlation-robust beta uncertainty
```

HAC is useful when the trader wants a more realistic confidence measure for the beta without changing the hedge ratio itself.

### GLS with AR(1) Omega

GLS changes the beta estimate by explicitly modeling correlated regression errors.

The regression is:

```text
y = X beta + error
```

The dashboard first estimates the residual autocorrelation:

```text
error_t = rho x error_(t-1) + innovation_t
```

It then constructs an AR(1)-style covariance structure:

```text
Omega_ij is proportional to rho^abs(i-j)
```

Instead of explicitly inverting the full Omega matrix, the code applies a Prais-Winsten whitening transformation and solves the transformed regression with QR decomposition.

The GLS output reports:

```text
GLS beta
GLS beta SE
GLS beta t-stat
GLS rho
```

Interpretation:

```text
GLS rho near 0       little residual serial correlation
large positive rho   nearby errors tend to persist
negative rho         nearby errors tend to reverse
```

GLS is the selected hedge ratio when the trader wants the beta estimate itself to reflect the estimated residual covariance structure.

### What To Use

The practical interpretation is:

```text
OLS                  baseline hedge ratio
OLS + HAC/Newey-West baseline hedge ratio with more robust uncertainty
GLS                  covariance-adjusted hedge ratio
```

The selected beta method flows into the recommended hedge notional, the static hedge backtest, and the z-score signal backtest. The price-level beta used to construct the cointegration spread remains a separate price regression, because the spread is defined from log spot levels rather than return hedge sizing.

The selected-pair beta table shows all three methods side by side. This lets the trader see whether the covariance adjustment materially changes the hedge ratio before choosing which method is selected for sizing.

---

## 8. Recommended Hedge Notional

The simplified trader workflow is:

```text
Input target notional.
Use beta to calculate hedge notional.
```

The formula is:

```text
recommended hedge notional = -target notional x beta
```

Example:

```text
Target = NZD
Hedge = AUD
Target notional = 100,000,000
Beta = 0.75
```

Then:

```text
Recommended hedge notional = -100,000,000 x 0.75
                            = -75,000,000
```

So the dashboard recommends:

```text
Hedge with -75mm AUD
```

The negative sign means the hedge is in the opposite direction.

If the trader is long target exposure, the hedge is short proxy exposure.

If beta is negative, the recommended hedge notional can be positive because the hedge currency historically moves opposite to the target.

---

## 9. Correlation Matrix

The correlation matrix shows pairwise correlations between selected currencies.

It uses the reconstructed 30-minute return series.

High positive correlation means two currencies usually move together.

Example:

```text
EUR vs GBP correlation = 0.80
```

This means EUR and GBP 30-minute BFIX returns have historically moved together over the selected history.

Correlation is useful for hedge screening, but it is not enough by itself.

A pair can be correlated but still have unstable beta, poor mean reversion, or bad hedge performance.

---

## 10. Rolling Correlation

The rolling correlation chart shows whether the relationship is stable through time.

Example:

```text
96-bar rolling correlation: NZD versus AUD
```

This means each point on the chart uses the most recent 96 half-hour returns.

Useful interpretation:

```text
Stable high correlation    proxy hedge may be reliable
Falling correlation        hedge relationship may be breaking down
Very unstable correlation  hedge ratio is fragile
```

This is important because a pair can have a good full-sample correlation but fail recently.

---

## 11. Cointegration Spread

The dashboard also constructs a beta-adjusted price spread.

The formula is approximately:

```text
spread = log(target price) - intercept - beta x log(hedge price)
```

For example:

```text
NZD minus beta x AUD
```

means:

```text
After scaling AUD by the historical relationship,
is NZD expensive or cheap versus AUD?
```

This spread is used for rich/cheap analysis.

---

## 12. Spread Z-Score

The spread z-score asks:

```text
How stretched is the current beta-adjusted spread versus its recent history?
```

Formula:

```text
z-score = (current spread - average spread) / spread volatility
```

Interpretation:

```text
z = 0      spread is near normal
z = +2     target is rich versus hedge
z = -2     target is cheap versus hedge
```

Example:

```text
NZD minus beta x AUD z-score = +2.1
```

This means NZD is about 2.1 standard deviations rich versus AUD relative to the recent spread history.

Why this matters:

```text
Correlation tells you whether two currencies move together.
Z-score tells you whether their relationship is currently stretched.
```

That helps with timing.

If a proxy hedge is correlated and the spread is stretched in your favor, the hedge may have a relative-value tailwind.

If the spread is stretched against you, the hedge may still reduce directional risk, but it may add basis risk.

---

## 12A. Rolling Spread Mean And Standard Deviation

The dashboard also displays the rolling inputs used to calculate the z-score:

```text
spread level
rolling mean
rolling standard deviation
upper mean + 1 standard deviation band
lower mean - 1 standard deviation band
current z-score
```

The rolling mean is the recent average of the spread over the selected z-score window. The rolling standard deviation measures how widely the spread has moved around that average.

The chart shows the spread level, its rolling mean, the one-standard-deviation bands, and rolling standard deviation on a secondary axis. This helps distinguish two situations:

```text
Large z-score because the spread moved far from its mean
Large z-score because recent spread volatility is very low
```

The latest-value table makes the exact mean, standard deviation, spread level, and z-score visible for the selected target/hedge pair.

---

## 13. Cointegration P-Value

The dashboard calculates a cointegration p-value when `statsmodels` is available.

Cointegration tests whether two price series have a stable long-run relationship.

Lower p-values are better.

Rough interpretation:

```text
p < 0.05    strong evidence of cointegration
p < 0.10    weaker but possibly useful evidence
p > 0.10    weak evidence of a stable long-run spread
```

Example:

```text
coint p = 0.5574
```

This is high.

It means the test does not find strong evidence that the spread is reliably mean-reverting.

The pair may still be correlated, but the cointegration signal is weak.

---

## 14. ADF P-Value

The ADF test is applied to the constructed spread.

It asks:

```text
Does the spread itself look stationary?
```

Stationary means the spread tends to return toward an average rather than drifting forever.

Lower p-values are better.

Rough interpretation:

```text
ADF p < 0.05    strong evidence the spread is stationary
ADF p < 0.10    weaker but possibly useful evidence
ADF p > 0.10    weak evidence of stationarity
```

Example:

```text
ADF p = 0.3117
```

This is not convincing.

It suggests the spread does not look strongly mean-reverting over the selected sample.

---

## 15. Score

The score is a ranking heuristic.

It is not a formal statistical test.

The formula is:

```text
score = abs(latest correlation)
      + 0.35 x abs(spread z-score)
      + 0.5 x (1 - cointegration p-value)
```

The components are:

```text
abs(latest correlation)       hedge similarity
abs(spread z-score)           rich/cheap dislocation
1 - cointegration p-value     long-run relationship confidence
```

The `0.35` weight on z-score is used because z-score can be larger than correlation.

Correlation is bounded between:

```text
-1 and +1
```

So absolute correlation is between:

```text
0 and 1
```

But z-score can be:

```text
0, 1, 2, 3, 4, ...
```

Without the `0.35` weight, a pair with a huge z-score but poor hedge quality could rank too highly.

The score should be used as a screen, not as the final trading decision.

The trader should still look at:

```text
latest correlation
beta
recommended hedge notional
cointegration p-value
ADF p-value
spread z-score
backtest volatility reduction
```

---

## 16. Ranked Proxy Hedge Candidates Table

This table ranks possible hedge currencies for the selected target.

Columns include:

```text
target
hedge
observations
target notional
latest corr
full corr
return beta
spread z
half-life bars
coint p
ADF p
recommended hedge notional
score
```

The most trader-relevant columns are:

```text
latest corr
return beta
recommended hedge notional
spread z
coint p
ADF p
score
```

Example:

```text
target notional = 100,000,000
return beta = 0.82
recommended hedge notional = -82,000,000
```

This means:

```text
For 100mm target exposure,
the beta-implied proxy hedge is -82mm.
```

---

## 17. Backtest

The backtest is designed to answer:

```text
Would this beta-implied hedge have reduced risk historically?
```

For the selected target and hedge, the dashboard simulates:

```text
Long target notional
Plus beta-implied hedge notional
```

The hedge notional is:

```text
hedge notional = -target notional x beta
```

For each 30-minute return bar:

```text
unhedged P&L = target notional x target return
hedge P&L = hedge notional x hedge return
hedged P&L = unhedged P&L + hedge P&L
```

Then it builds cumulative P&L paths:

```text
cumulative unhedged P&L
cumulative hedged P&L
```

The backtest table shows:

```text
target
hedge
observations
target notional
return beta
beta method
beta SE
beta t-stat
GLS rho
hedge notional
unhedged vol/bar
hedged vol/bar
vol reduction
unhedged cumulative P&L
hedged cumulative P&L
hedged max drawdown
hedged worst bar
```

---

## 18. Z-Score Signal Backtest

The static hedge backtest answers:

```text
Does a beta-implied hedge reduce risk?
```

The z-score signal backtest answers a different question:

```text
Can the rich/cheap spread signal be traded as a mean-reversion strategy?
```

The z-score signal backtest uses the same beta-adjusted spread as the spread chart.

The spread is:

```text
spread = log(target price) - intercept - spread beta x log(hedge price)
```

Then the dashboard calculates the spread z-score:

```text
z-score = (current spread - rolling spread average) / rolling spread volatility
```

The strategy uses one threshold input plus an exit band:

```text
Signal |z|
Exit band
```

`Signal |z|` is treated as an absolute threshold. If a trader types `2.0` or `-2.0`, the strategy uses `2.0` as the trigger level.

The default rules are:

```text
If z-score >= +Signal |z|:
    target is rich versus hedge
    short target / long hedge

If z-score <= -Signal |z|:
    target is cheap versus hedge
    long target / short hedge

If in a long-spread trade and z-score rises back to -exit band:
    close trade

If in a short-spread trade and z-score falls back to +exit band:
    close trade
```

Example:

```text
Signal |z| = 1.5
Exit band = 0.25
```

This means:

```text
Short the target when it is rich by at least +1.5 z-score.
Long the target when it is cheap by at least -1.5 z-score.
Exit after the spread mean-reverts close to zero.
```

The strategy uses the target notional as the trade size.

The strategy uses the selected return beta for the hedge notional. The separate spread beta is retained for the log-spot spread and z-score calculation.

When the signal is long spread:

```text
target notional = +target notional input
hedge notional = -target notional input x hedge beta
```

When the signal is short spread:

```text
target notional = -target notional input
hedge notional = +target notional input x hedge beta
```

The backtest shifts the position by one bar.

That means:

```text
A signal observed at time t is traded from time t+1 onward.
```

This avoids look-ahead bias from trading on the same return bar used to observe the signal.

The z-score signal backtest reports:

```text
target
hedge
observations
entry z
exit z
spread beta
hedge beta
beta method
target notional
active bars
active %
signal triggers
completed trades
open trades
strategy vol/bar
strategy cumulative P&L
strategy max drawdown
strategy worst bar
strategy best bar
```

The chart shows:

```text
cumulative strategy P&L
position through time
```

The selected return beta is used for the hedge leg of the signal strategy. The spread beta remains the price-level beta used to calculate the z-score.

### Signal And Exit Spot Levels

The z-score strategy observes a signal on one BFIX bar and executes on the next bar. The blotter keeps both timestamps visible:

```text
signal time    bar where the z-score crossed the threshold
entry time     next BFIX bar where the position becomes active
exit signal time  bar where the exit condition was observed
exit time      next BFIX bar where the position becomes flat
```

For each trade, the blotter shows:

```text
signal z and signal spread level
signal target spot and signal hedge spot
entry target spot and entry hedge spot
exit z and exit spread level
exit signal target spot and exit signal hedge spot
exit target spot and exit hedge spot
direction
bars held
trade P&L
status: CLOSED or OPEN
```

This makes it possible to see the actual spot levels at which the signal was observed and the next-bar spot levels at which the simulated trade was entered or exited.

### Trade Count

`signal triggers` is the number of times the strategy moved from flat into a long or short spread position.

`completed trades` counts trades with both an entry and an exit in the sample.

`open trades` identifies a position that was still active at the last available BFIX bar.

The dashboard's trade blotter has one row per trigger, so the trader can inspect where each trigger occurred for the selected target/hedge pair.

Position interpretation:

```text
+1    long target / short hedge
 0    flat
-1    short target / long hedge
```

This backtest should be interpreted separately from the static hedge backtest.

Good static hedge performance means:

```text
The proxy hedge reduced exposure risk.
```

Good z-score signal performance means:

```text
The spread mean-reversion signal was historically tradable.
```

Those are related, but they are not the same thing.

---

## 19. Volatility Reduction

Volatility reduction is one of the most important backtest fields.

Formula:

```text
vol reduction = 1 - hedged volatility / unhedged volatility
```

Example:

```text
unhedged vol/bar = 100,000
hedged vol/bar = 45,000
```

Then:

```text
vol reduction = 1 - 45,000 / 100,000
              = 55%
```

Interpretation:

```text
Positive vol reduction     hedge reduced risk
Near zero                  hedge did not help much
Negative vol reduction     hedge made risk worse
```

This is often more useful than correlation alone.

Correlation can be high, but if the beta hedge does not reduce residual volatility, the hedge is not very useful.

---

## 20. Max Drawdown And Worst Bar

The backtest also reports:

```text
hedged max drawdown
hedged worst bar
```

Hedged max drawdown is the worst peak-to-trough loss in cumulative hedged P&L.

Hedged worst bar is the worst single 30-minute hedged P&L observation.

These help answer:

```text
How bad did the hedge get during the sample?
Was the risk reduction smooth, or did it have ugly tail events?
```

---

## 21. How To Read A Good Hedge Candidate

A strong hedge candidate usually has:

```text
High latest correlation
Stable full-sample relationship
Reasonable beta
Positive backtest volatility reduction
Lower cointegration p-value
Lower ADF p-value
Manageable spread z-score
Good data coverage
```

Example of a strong candidate:

```text
latest corr = 0.82
beta = 0.76
vol reduction = 48%
coint p = 0.04
ADF p = 0.06
spread z = +1.2
```

Interpretation:

```text
The hedge has moved with the target,
the beta hedge historically reduced volatility,
and the spread relationship has some evidence of stability.
```

---

## 22. How To Read A Weak Hedge Candidate

A weak hedge candidate might have:

```text
Low or unstable correlation
Very unstable beta
Negative volatility reduction
High cointegration p-value
High ADF p-value
Poor data coverage
Extreme spread z-score without mean-reversion evidence
```

Example:

```text
latest corr = 0.35
beta = 0.41
vol reduction = -12%
coint p = 0.56
ADF p = 0.31
spread z = +2.8
```

Interpretation:

```text
The pair may look stretched,
but the hedge relationship is weak and the backtest says the hedge made risk worse.
```

That is not a good proxy hedge.

---

## 23. Important Caveats

The dashboard is a decision-support tool, not an automatic trading system.

Key caveats:

```text
1. Beta is historical and can change.
2. Correlation can break during stress.
3. Cointegration tests are sample-dependent.
4. BFIX data can have missing or stale prints.
5. Backtest P&L is approximate because it uses return x notional.
6. The recommended hedge notional assumes the trader is comfortable using return beta as the hedge ratio.
7. The z-score signal backtest is a mean-reversion strategy test, not a hedge effectiveness test.
```

The output should be read as:

```text
This is the historically implied proxy hedge.
Now decide whether the relationship is trustworthy enough to trade.
```

---

## 24. Practical Trader Workflow

A clean workflow would be:

```text
1. Select the currency universe.
2. Load 20 days or 1 month of BFIX data.
3. Check data quality.
4. Select the target currency.
5. Enter target notional.
6. Look at ranked hedge candidates.
7. Choose the best hedge candidate.
8. Check the recommended hedge notional.
9. Review the backtest volatility reduction.
10. Check the z-score signal backtest if you want to trade rich/cheap mean reversion.
11. Check rolling correlation stability.
12. Check spread z-score for rich/cheap timing.
13. Avoid pairs where the hedge backtest worsens risk.
```

The best next product enhancement would be a summary card that automatically says:

```text
Recommended hedge: AUD -82mm
Confidence: Medium
Why: 0.78 rolling correlation, 35% vol reduction, beta stable, but weak cointegration.
```
