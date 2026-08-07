# ============================================================================
# BFIX 30-minute DV01 hedge dashboard
#
# Builds on the original BFIX 30-minute correlation dashboard, but separates:
#   1. BFIX ticker construction and BQL retrieval
#   2. Panel cleaning, frequency alignment, and return construction
#   3. Pair correlation, beta, spread, and cointegration analytics
#   4. DV01 hedge sizing for correlated/cointegrated proxy hedges
#
# Intended use:
#   Paste this into a Bloomberg BQuant notebook cell. It does not overwrite the
#   original spot-FX or single-time BFIX dashboards.
# ============================================================================

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets
import bql

from IPython.display import display, clear_output

try:
    from statsmodels.tsa.stattools import adfuller, coint
except Exception:
    adfuller = None
    coint = None


# Create a BQL service only if one does not already exist in the notebook.
try:
    bq
except NameError:
    bq = bql.Service()


# ----------------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------------

BFIX30_CURRENCIES = {
    "EUR": "EUR",
    "GBP": "GBP",
    "CHF": "CHF",
    "AUD": "AUD",
    "CAD": "CAD",
    "NZD": "NZD",
    "NOK": "NOK",
    "SEK": "SEK",
}

BFIX30_SLOTS = [
    (f"{hour:02d}:{minute:02d}", hour, minute)
    for hour in range(24)
    for minute in (0, 30)
]

BFIX30_BATCH_SIZE = 96


@dataclass(frozen=True)
class BFIXPanel:
    prices: pd.DataFrame
    returns: pd.DataFrame
    quality: pd.DataFrame
    raw: pd.DataFrame


@dataclass(frozen=True)
class PairStats:
    target: str
    hedge: str
    observations: int
    target_notional: float
    latest_corr: float
    full_corr: float
    beta: float
    intercept: float
    beta_method: str
    beta_se: float
    beta_tstat: float
    gls_rho: float
    hac_lags: int
    spread_last: float
    spread_z: float
    spread_mean: float
    spread_std: float
    half_life_bars: float
    coint_pvalue: float
    adf_pvalue: float
    recommended_hedge_notional: float
    score: float


# ----------------------------------------------------------------------------
# 2. BFIX ticker construction and BQL utilities
# ----------------------------------------------------------------------------

def bfix30_fixing_code(hour, minute):
    """
    Return the BFIX ticker time code.

    00:00 -> "000"
    00:30 -> "003"
    01:00 -> "010"
    01:30 -> "013"
    23:30 -> "233"
    """
    if minute not in (0, 30):
        raise ValueError("BFIX 30-minute slots must use minute=0 or minute=30.")

    minute_code = 0 if minute == 0 else 3
    return f"{hour:02d}{minute_code}"


def bfix30_ticker(currency, hour, minute):
    code = bfix30_fixing_code(hour, minute)
    return f"{currency} F{code} Curncy"


def bfix30_normalize_id(value):
    return " ".join(str(value).upper().split())


def bfix30_find_column(frame, expected):
    expected = expected.upper()
    for column in frame.columns:
        if str(column).upper() == expected:
            return column
    raise KeyError(f"Could not find column '{expected}' in {list(frame.columns)}")


def bfix30_chunked(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def bfix30_remove_duplicate_columns(frame):
    return frame.loc[:, ~frame.columns.duplicated(keep="last")]


def bfix30_expected_index(start_time, end_time):
    if pd.isna(start_time) or pd.isna(end_time):
        return pd.DatetimeIndex([])
    return pd.date_range(
        pd.Timestamp(start_time).floor("30min"),
        pd.Timestamp(end_time).floor("30min"),
        freq="30min",
    )


def bfix30_get_raw_prices(currency_labels, history_start):
    ticker_metadata = {}
    requested_tickers = []

    for currency in currency_labels:
        for _, hour, minute in BFIX30_SLOTS:
            ticker = bfix30_ticker(currency, hour, minute)
            requested_tickers.append(ticker)
            ticker_metadata[bfix30_normalize_id(ticker)] = {
                "currency": currency,
                "hour": hour,
                "minute": minute,
            }

    print(
        f"Requesting {len(requested_tickers)} BFIX securities "
        f"for {len(currency_labels)} currencies."
    )

    frames = []
    for batch_number, ticker_batch in enumerate(
        bfix30_chunked(requested_tickers, BFIX30_BATCH_SIZE),
        start=1,
    ):
        print(
            f"Loading BQL batch {batch_number}: "
            f"{len(ticker_batch)} BFIX securities..."
        )

        request = bql.Request(
            ticker_batch,
            {
                "px": bq.data.px_last(
                    dates=bq.func.range(history_start, "0D"),
                    per="D",
                )
            },
        )

        response = bq.execute(request)
        for response_item in response:
            response_frame = response_item.df().reset_index()
            if not response_frame.empty:
                frames.append(response_frame)

    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    print(f"Total BQL rows returned: {len(raw):,}")
    print("BQL columns:", list(raw.columns))

    id_column = bfix30_find_column(raw, "ID")
    date_column = bfix30_find_column(raw, "DATE")
    try:
        price_column = bfix30_find_column(raw, "PX")
    except KeyError:
        price_column = bfix30_find_column(raw, "PX_LAST")

    raw["bfix30_metadata"] = (
        raw[id_column].map(bfix30_normalize_id).map(ticker_metadata)
    )

    unmapped_ids = (
        raw.loc[raw["bfix30_metadata"].isna(), id_column]
        .dropna()
        .unique()
        .tolist()
    )
    if unmapped_ids:
        print("Unmapped Bloomberg IDs:")
        for identifier in unmapped_ids[:20]:
            print(f"  {identifier}")

    raw = raw.dropna(subset=["bfix30_metadata", price_column]).copy()
    if raw.empty:
        return raw

    raw["currency"] = raw["bfix30_metadata"].map(lambda metadata: metadata["currency"])
    raw["hour"] = raw["bfix30_metadata"].map(lambda metadata: metadata["hour"])
    raw["minute"] = raw["bfix30_metadata"].map(lambda metadata: metadata["minute"])
    raw["timestamp"] = (
        pd.to_datetime(raw[date_column]).dt.normalize()
        + pd.to_timedelta(raw["hour"], unit="h")
        + pd.to_timedelta(raw["minute"], unit="m")
    )
    raw["price"] = pd.to_numeric(raw[price_column], errors="coerce")

    return raw.dropna(subset=["timestamp", "currency", "price"]).copy()


# ----------------------------------------------------------------------------
# 3. Panel construction, cleaning, quality, and returns
# ----------------------------------------------------------------------------

def bfix30_quality_report(prices, selected):
    if prices.empty:
        return pd.DataFrame(
            columns=["currency", "observed", "missing", "coverage", "first", "last"]
        )

    expected = bfix30_expected_index(prices.index.min(), prices.index.max())
    aligned = prices.reindex(expected).reindex(columns=selected)

    rows = []
    for currency in selected:
        series = aligned[currency]
        observed = int(series.notna().sum())
        missing = int(series.isna().sum())
        coverage = observed / max(observed + missing, 1)
        valid_index = series.dropna().index
        rows.append(
            {
                "currency": currency,
                "observed": observed,
                "missing": missing,
                "coverage": coverage,
                "first": valid_index.min() if len(valid_index) else pd.NaT,
                "last": valid_index.max() if len(valid_index) else pd.NaT,
            }
        )

    return pd.DataFrame(rows).sort_values("coverage", ascending=False)


def bfix30_winsorize_returns(returns, z_limit):
    if not z_limit or z_limit <= 0:
        return returns

    clipped = returns.copy()
    for column in clipped.columns:
        series = clipped[column].dropna()
        if len(series) < 5:
            continue
        median = series.median()
        mad = (series - median).abs().median()
        if not np.isfinite(mad) or mad == 0:
            continue
        robust_sigma = 1.4826 * mad
        lower = median - z_limit * robust_sigma
        upper = median + z_limit * robust_sigma
        clipped[column] = clipped[column].clip(lower, upper)

    return clipped


def bfix30_construct_panel(
    currency_labels,
    history_start,
    return_method="log",
    align_frequency=True,
    fill_single_gap=False,
    winsor_z=8.0,
):
    raw = bfix30_get_raw_prices(currency_labels, history_start)
    if raw.empty:
        empty = pd.DataFrame(columns=currency_labels)
        return BFIXPanel(prices=empty, returns=empty, quality=empty, raw=raw)

    raw = (
        raw.drop_duplicates(subset=["timestamp", "currency"], keep="last")
        .sort_values(["timestamp", "currency"])
    )

    prices = raw.pivot(index="timestamp", columns="currency", values="price")
    prices.index = pd.to_datetime(prices.index)
    prices = bfix30_remove_duplicate_columns(prices)
    prices = prices.reindex(columns=currency_labels).sort_index()

    if align_frequency:
        expected = bfix30_expected_index(prices.index.min(), prices.index.max())
        prices = prices.reindex(expected)

    if fill_single_gap:
        prices = prices.interpolate(method="time", limit=1, limit_area="inside")

    if return_method == "log":
        returns = np.log(prices).diff()
    elif return_method == "pct":
        returns = prices.pct_change(fill_method=None)
    else:
        raise ValueError("return_method must be 'log' or 'pct'.")

    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    returns = bfix30_winsorize_returns(returns, winsor_z)

    quality = bfix30_quality_report(prices, currency_labels)
    return BFIXPanel(prices=prices, returns=returns, quality=quality, raw=raw)


# ----------------------------------------------------------------------------
# 4. Pair analytics, cointegration, and hedge sizing
# ----------------------------------------------------------------------------

def bfix30_ols_beta(y, x):
    paired = pd.concat([y, x], axis=1).dropna()
    if len(paired) < 3:
        return np.nan, np.nan

    y_values = paired.iloc[:, 0].astype(float)
    x_values = paired.iloc[:, 1].astype(float)
    design = np.column_stack([np.ones(len(x_values)), x_values.to_numpy()])
    if np.linalg.matrix_rank(design) < 2:
        return np.nan, np.nan

    try:
        intercept, beta = np.linalg.lstsq(
            design, y_values.to_numpy(), rcond=None
        )[0]
    except (np.linalg.LinAlgError, ValueError):
        return np.nan, np.nan

    return float(beta), float(intercept)


def bfix30_ols_beta_se(y, x):
    """Classical OLS standard error for the slope coefficient."""
    paired = pd.concat([y, x], axis=1).dropna()
    if len(paired) < 3:
        return np.nan

    y_values = paired.iloc[:, 0].astype(float).to_numpy()
    x_values = paired.iloc[:, 1].astype(float).to_numpy()
    design = np.column_stack([np.ones(len(x_values)), x_values])
    if np.linalg.matrix_rank(design) < 2:
        return np.nan

    try:
        coefficients = np.linalg.lstsq(design, y_values, rcond=None)[0]
        residuals = y_values - design @ coefficients
        degrees_of_freedom = len(y_values) - design.shape[1]
        if degrees_of_freedom <= 0:
            return np.nan
        sigma_squared = float(residuals @ residuals) / degrees_of_freedom
        covariance = sigma_squared * np.linalg.pinv(design.T @ design)
        variance = float(covariance[1, 1])
        return math.sqrt(variance) if np.isfinite(variance) and variance >= 0 else np.nan
    except (np.linalg.LinAlgError, ValueError):
        return np.nan


def bfix30_hac_beta_se(y, x, max_lag=4):
    """Newey-West HAC standard error for the OLS slope coefficient."""
    paired = pd.concat([y, x], axis=1).dropna()
    if len(paired) < 3:
        return np.nan

    y_values = paired.iloc[:, 0].astype(float).to_numpy()
    x_values = paired.iloc[:, 1].astype(float).to_numpy()
    design = np.column_stack([np.ones(len(x_values)), x_values])
    if np.linalg.matrix_rank(design) < 2:
        return np.nan

    try:
        coefficients = np.linalg.lstsq(design, y_values, rcond=None)[0]
        residuals = y_values - design @ coefficients
        scores = design * residuals[:, None]
        bread = np.linalg.pinv(design.T @ design)
        meat = scores.T @ scores
        lag_count = max(0, min(int(max_lag), len(scores) - 1))

        for lag in range(1, lag_count + 1):
            weight = 1.0 - lag / (lag_count + 1.0)
            autocovariance = scores[lag:].T @ scores[:-lag]
            meat += weight * (autocovariance + autocovariance.T)

        finite_sample_scale = len(scores) / max(len(scores) - design.shape[1], 1)
        covariance = finite_sample_scale * bread @ meat @ bread
        variance = float(covariance[1, 1])
        return math.sqrt(variance) if np.isfinite(variance) and variance >= 0 else np.nan
    except (np.linalg.LinAlgError, ValueError):
        return np.nan


def bfix30_gls_beta(y, x):
    """
    Estimate a slope using feasible GLS with AR(1) residual covariance.

    The AR(1) residual estimate defines Omega. Prais-Winsten whitening is then
    solved with QR, which is numerically more stable than explicitly inverting
    the full covariance matrix.
    """
    paired = pd.concat([y, x], axis=1).dropna()
    if len(paired) < 5:
        return np.nan, np.nan, np.nan, np.nan

    y_values = paired.iloc[:, 0].astype(float).to_numpy()
    x_values = paired.iloc[:, 1].astype(float).to_numpy()
    design = np.column_stack([np.ones(len(x_values)), x_values])
    if np.linalg.matrix_rank(design) < 2:
        return np.nan, np.nan, np.nan, np.nan

    try:
        ols_coefficients = np.linalg.lstsq(design, y_values, rcond=None)[0]
        ols_residuals = y_values - design @ ols_coefficients
        denominator = float(ols_residuals[:-1] @ ols_residuals[:-1])
        if denominator <= 0 or not np.isfinite(denominator):
            rho = 0.0
        else:
            rho = float((ols_residuals[1:] @ ols_residuals[:-1]) / denominator)
            rho = float(np.clip(rho, -0.95, 0.95))

        transformed_design = design.copy()
        transformed_y = y_values.copy()
        first_scale = math.sqrt(max(1.0 - rho * rho, 1e-8))
        transformed_design[0] = first_scale * design[0]
        transformed_y[0] = first_scale * y_values[0]
        transformed_design[1:] = design[1:] - rho * design[:-1]
        transformed_y[1:] = y_values[1:] - rho * y_values[:-1]

        q_matrix, r_matrix = np.linalg.qr(transformed_design, mode="reduced")
        if np.linalg.matrix_rank(r_matrix) < 2:
            return np.nan, np.nan, np.nan, rho

        coefficients = np.linalg.solve(
            r_matrix, q_matrix.T @ transformed_y
        )
        transformed_residuals = transformed_y - transformed_design @ coefficients
        degrees_of_freedom = len(y_values) - transformed_design.shape[1]
        sigma_squared = float(transformed_residuals @ transformed_residuals) / max(
            degrees_of_freedom, 1
        )
        covariance = sigma_squared * np.linalg.pinv(
            transformed_design.T @ transformed_design
        )
        variance = float(covariance[1, 1])
        beta_se = math.sqrt(variance) if np.isfinite(variance) and variance >= 0 else np.nan
        return float(coefficients[1]), float(coefficients[0]), beta_se, rho
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return np.nan, np.nan, np.nan, np.nan


def bfix30_estimate_beta(y, x, method="ols", hac_lags=4):
    """Return beta, intercept, standard error, t-statistic, and GLS rho."""
    method = str(method).lower()
    ols_beta, ols_intercept = bfix30_ols_beta(y, x)
    if not np.isfinite(ols_beta):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    gls_rho = np.nan
    if method == "gls":
        beta, intercept, beta_se, gls_rho = bfix30_gls_beta(y, x)
        if not np.isfinite(beta):
            beta, intercept = ols_beta, ols_intercept
            beta_se = bfix30_ols_beta_se(y, x)
    else:
        beta, intercept = ols_beta, ols_intercept
        if method == "hac":
            beta_se = bfix30_hac_beta_se(y, x, max_lag=hac_lags)
        else:
            beta_se = bfix30_ols_beta_se(y, x)

    beta_tstat = (
        beta / beta_se
        if np.isfinite(beta) and np.isfinite(beta_se) and beta_se > 0
        else np.nan
    )
    return beta, intercept, beta_se, beta_tstat, gls_rho


def bfix30_spread_half_life(spread):
    series = spread.dropna().astype(float)
    if len(series) < 20:
        return np.nan

    lagged = series.shift(1).dropna()
    delta = series.diff().dropna()
    paired = pd.concat([delta, lagged], axis=1).dropna()

    if len(paired) < 20:
        return np.nan

    slope, _ = bfix30_ols_beta(paired.iloc[:, 0], paired.iloc[:, 1])
    if not np.isfinite(slope) or slope >= 0:
        return np.nan

    half_life = -math.log(2) / slope
    return float(half_life) if np.isfinite(half_life) else np.nan


def bfix30_safe_coint_pvalue(series_a, series_b):
    if coint is None:
        return np.nan
    paired = pd.concat([series_a, series_b], axis=1).dropna()
    if len(paired) < 30:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            _, pvalue, _ = coint(paired.iloc[:, 0], paired.iloc[:, 1])
            return float(pvalue)
        except Exception:
            return np.nan


def bfix30_safe_adf_pvalue(spread):
    if adfuller is None:
        return np.nan
    series = spread.dropna().astype(float)
    if len(series) < 30:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(adfuller(series, autolag="AIC")[1])
        except Exception:
            return np.nan


def bfix30_pair_stats(
    prices,
    returns,
    target,
    hedge,
    rolling_window,
    z_window,
    target_notional,
    beta_method="ols",
    hac_lags=4,
):
    if target == hedge:
        return None

    required = [target, hedge]
    missing_returns = [column for column in required if column not in returns.columns]
    missing_prices = [column for column in required if column not in prices.columns]

    if missing_returns or missing_prices:
        return None

    paired_returns = returns.reindex(columns=required).dropna()
    observations = len(paired_returns)

    if observations < max(10, min(rolling_window, 30)):
        return None

    target_ret = paired_returns[target]
    hedge_ret = paired_returns[hedge]

    full_corr = target_ret.corr(hedge_ret)
    latest_corr = target_ret.rolling(rolling_window).corr(hedge_ret).dropna()
    latest_corr_value = latest_corr.iloc[-1] if len(latest_corr) else full_corr

    beta, ret_intercept, beta_se, beta_tstat, gls_rho = bfix30_estimate_beta(
        target_ret,
        hedge_ret,
        method=beta_method,
        hac_lags=hac_lags,
    )

    paired_prices = prices.reindex(columns=required).dropna()
    if len(paired_prices) < 30:
        return None

    log_target = np.log(paired_prices[target].astype(float))
    log_hedge = np.log(paired_prices[hedge].astype(float))
    price_beta, price_intercept = bfix30_ols_beta(log_target, log_hedge)

    if np.isfinite(price_beta):
        spread = log_target - price_intercept - price_beta * log_hedge
    else:
        spread = log_target - log_hedge

    spread_window = spread.dropna().tail(z_window)
    spread_mean = float(spread_window.mean()) if len(spread_window) else np.nan
    spread_std = float(spread_window.std()) if len(spread_window) > 1 else np.nan
    spread_last = float(spread.dropna().iloc[-1]) if len(spread.dropna()) else np.nan
    spread_z = (
        (spread_last - spread_mean) / spread_std
        if np.isfinite(spread_std) and spread_std > 0
        else np.nan
    )

    coint_pvalue = bfix30_safe_coint_pvalue(log_target, log_hedge)
    adf_pvalue = bfix30_safe_adf_pvalue(spread)
    half_life = bfix30_spread_half_life(spread)

    target_notional = float(target_notional)
    recommended_hedge_notional = np.nan

    if np.isfinite(beta):
        recommended_hedge_notional = -target_notional * float(beta)

    corr_component = abs(latest_corr_value) if np.isfinite(latest_corr_value) else 0
    z_component = abs(spread_z) if np.isfinite(spread_z) else 0
    coint_component = 1 - coint_pvalue if np.isfinite(coint_pvalue) else 0
    score = corr_component + 0.35 * z_component + 0.5 * coint_component

    return PairStats(
        target=target,
        hedge=hedge,
        observations=observations,
        target_notional=target_notional,
        latest_corr=float(latest_corr_value),
        full_corr=float(full_corr),
        beta=float(beta),
        intercept=float(ret_intercept),
        beta_method=str(beta_method).lower(),
        beta_se=float(beta_se) if np.isfinite(beta_se) else np.nan,
        beta_tstat=float(beta_tstat) if np.isfinite(beta_tstat) else np.nan,
        gls_rho=float(gls_rho) if np.isfinite(gls_rho) else np.nan,
        hac_lags=int(hac_lags),
        spread_last=spread_last,
        spread_z=float(spread_z) if np.isfinite(spread_z) else np.nan,
        spread_mean=spread_mean,
        spread_std=spread_std,
        half_life_bars=float(half_life) if np.isfinite(half_life) else np.nan,
        coint_pvalue=float(coint_pvalue) if np.isfinite(coint_pvalue) else np.nan,
        adf_pvalue=float(adf_pvalue) if np.isfinite(adf_pvalue) else np.nan,
        recommended_hedge_notional=(
            float(recommended_hedge_notional)
            if np.isfinite(recommended_hedge_notional)
            else np.nan
        ),
        score=float(score),
    )


def bfix30_pair_table(
    prices,
    returns,
    selected,
    target,
    rolling_window,
    z_window,
    target_notional,
    beta_method="ols",
    hac_lags=4,
):
    rows = []
    available = [
        currency
        for currency in selected
        if currency in prices.columns and currency in returns.columns
    ]

    if target not in available:
        return pd.DataFrame()

    for hedge in available:
        if hedge == target:
            continue
        stats = bfix30_pair_stats(
            prices=prices,
            returns=returns,
            target=target,
            hedge=hedge,
            rolling_window=rolling_window,
            z_window=z_window,
            target_notional=target_notional,
            beta_method=beta_method,
            hac_lags=hac_lags,
        )
        if stats is not None:
            rows.append(stats.__dict__)

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)
    return table.sort_values(["score", "observations"], ascending=[False, False])


def bfix30_max_drawdown(cumulative_pnl):
    series = cumulative_pnl.dropna().astype(float)
    if series.empty:
        return np.nan
    running_peak = series.cummax()
    drawdown = series - running_peak
    return float(drawdown.min())


def bfix30_backtest_pair(returns, target, hedge, target_notional, beta):
    required = [target, hedge]
    missing = [column for column in required if column not in returns.columns]
    if missing or target == hedge or not np.isfinite(beta):
        return pd.DataFrame(), pd.DataFrame()

    paired_returns = returns.reindex(columns=required).dropna()
    if paired_returns.empty:
        return pd.DataFrame(), pd.DataFrame()

    target_notional = float(target_notional)
    hedge_notional = -target_notional * float(beta)

    path = pd.DataFrame(index=paired_returns.index)
    path["target_return"] = paired_returns[target]
    path["hedge_return"] = paired_returns[hedge]
    path["unhedged_pnl"] = target_notional * path["target_return"]
    path["hedge_pnl"] = hedge_notional * path["hedge_return"]
    path["hedged_pnl"] = path["unhedged_pnl"] + path["hedge_pnl"]
    path["cumulative_unhedged_pnl"] = path["unhedged_pnl"].cumsum()
    path["cumulative_hedged_pnl"] = path["hedged_pnl"].cumsum()

    unhedged_vol = path["unhedged_pnl"].std()
    hedged_vol = path["hedged_pnl"].std()
    vol_reduction = (
        1 - hedged_vol / unhedged_vol
        if np.isfinite(unhedged_vol) and unhedged_vol > 0
        else np.nan
    )

    summary = pd.DataFrame(
        [
            {
                "target": target,
                "hedge": hedge,
                "observations": len(path),
                "target_notional": target_notional,
                "beta": float(beta),
                "hedge_notional": hedge_notional,
                "unhedged_vol_per_bar": unhedged_vol,
                "hedged_vol_per_bar": hedged_vol,
                "vol_reduction": vol_reduction,
                "unhedged_cumulative_pnl": path["cumulative_unhedged_pnl"].iloc[-1],
                "hedged_cumulative_pnl": path["cumulative_hedged_pnl"].iloc[-1],
                "hedged_max_drawdown": bfix30_max_drawdown(
                    path["cumulative_hedged_pnl"]
                ),
                "hedged_worst_bar": path["hedged_pnl"].min(),
            }
        ]
    )

    return summary, path


def bfix30_pair_spread_levels(prices, target, hedge, z_window):
    required = [target, hedge]
    missing = [column for column in required if column not in prices.columns]
    if missing or target == hedge:
        return pd.DataFrame(), np.nan, np.nan

    paired_prices = prices.reindex(columns=required).dropna()
    if len(paired_prices) < max(30, z_window // 4):
        return pd.DataFrame(), np.nan, np.nan

    log_target = np.log(paired_prices[target].astype(float))
    log_hedge = np.log(paired_prices[hedge].astype(float))
    price_beta, price_intercept = bfix30_ols_beta(log_target, log_hedge)

    if np.isfinite(price_beta):
        spread = log_target - price_intercept - price_beta * log_hedge
    else:
        spread = log_target - log_hedge

    min_periods = max(30, z_window // 4)
    rolling_mean = spread.rolling(z_window, min_periods=min_periods).mean()
    rolling_std = spread.rolling(z_window, min_periods=min_periods).std()
    spread_z = (spread - rolling_mean) / rolling_std

    levels = pd.DataFrame(
        {
            "spread_level": spread,
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
            "spread_z": spread_z,
        },
        index=paired_prices.index,
    )
    return levels.replace([np.inf, -np.inf], np.nan), price_beta, price_intercept


def bfix30_pair_spread_zscore(prices, target, hedge, z_window):
    levels, price_beta, price_intercept = bfix30_pair_spread_levels(
        prices=prices,
        target=target,
        hedge=hedge,
        z_window=z_window,
    )
    if levels.empty:
        return pd.Series(dtype=float), price_beta, price_intercept
    return levels["spread_z"].dropna(), price_beta, price_intercept


def bfix30_zscore_signal_positions(
    spread_z,
    signal_z,
    exit_z,
):
    position = []
    current_position = 0
    signal_threshold = abs(float(signal_z))
    exit_band = abs(float(exit_z))

    for value in spread_z:
        if pd.isna(value):
            position.append(current_position)
            continue

        if current_position == 0:
            if value >= signal_threshold:
                current_position = -1
            elif value <= -signal_threshold:
                current_position = 1
        elif current_position == 1 and value >= -exit_band:
            current_position = 0
        elif current_position == -1 and value <= exit_band:
            current_position = 0

        position.append(current_position)

    return pd.Series(position, index=spread_z.index, dtype=float)


def bfix30_zscore_signal_backtest(
    prices,
    returns,
    target,
    hedge,
    target_notional,
    z_window,
    signal_z,
    exit_z,
    hedge_beta=np.nan,
    beta_method="ols",
):
    required = [target, hedge]
    missing_returns = [column for column in required if column not in returns.columns]
    missing_prices = [column for column in required if column not in prices.columns]

    if missing_returns or missing_prices or target == hedge:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    spread_levels, spread_beta, _ = bfix30_pair_spread_levels(
        prices=prices,
        target=target,
        hedge=hedge,
        z_window=z_window,
    )
    if spread_levels.empty or not np.isfinite(spread_beta):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    paired_returns = returns.reindex(columns=required).dropna()
    paired_prices = prices.reindex(columns=required)
    paired_prices = paired_prices.rename(
        columns={target: "target_spot", hedge: "hedge_spot"}
    )
    path = paired_returns.join(spread_levels, how="inner")
    path = path.join(paired_prices, how="inner")
    path = path.dropna(subset=["spread_z"])

    if path.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    signal_position = bfix30_zscore_signal_positions(
        spread_z=path["spread_z"],
        signal_z=float(signal_z),
        exit_z=float(exit_z),
    )

    # Shift by one bar so a signal observed at time t is traded from t+1 onward.
    path["signal_position"] = signal_position
    path["position"] = signal_position.shift(1).fillna(0.0)
    selected_hedge_beta = (
        float(hedge_beta) if np.isfinite(hedge_beta) else float(spread_beta)
    )
    path["target_notional"] = path["position"] * float(target_notional)
    path["hedge_notional"] = (
        -path["position"] * float(target_notional) * selected_hedge_beta
    )
    path["target_pnl"] = path["target_notional"] * path[target]
    path["hedge_pnl"] = path["hedge_notional"] * path[hedge]
    path["strategy_pnl"] = path["target_pnl"] + path["hedge_pnl"]
    path["cumulative_strategy_pnl"] = path["strategy_pnl"].cumsum()

    active_bars = int((path["position"] != 0).sum())
    blotter = bfix30_zscore_trade_blotter(path, target=target, hedge=hedge)
    entries = int(len(blotter))
    completed_trades = int((blotter["status"] == "CLOSED").sum()) if not blotter.empty else 0
    open_trades = int((blotter["status"] == "OPEN").sum()) if not blotter.empty else 0
    pnl_vol = path["strategy_pnl"].std()

    summary = pd.DataFrame(
        [
            {
                "target": target,
                "hedge": hedge,
                "observations": len(path),
                "signal_z": abs(float(signal_z)),
                "exit_z": float(exit_z),
                "spread_beta": float(spread_beta),
                "hedge_beta": selected_hedge_beta,
                "beta_method": str(beta_method).lower(),
                "target_notional": float(target_notional),
                "active_bars": active_bars,
                "active_pct": active_bars / len(path) if len(path) else np.nan,
                "signal_triggers": entries,
                "completed_trades": completed_trades,
                "open_trades": open_trades,
                "strategy_vol_per_bar": pnl_vol,
                "strategy_cumulative_pnl": path["cumulative_strategy_pnl"].iloc[-1],
                "strategy_max_drawdown": bfix30_max_drawdown(
                    path["cumulative_strategy_pnl"]
                ),
                "strategy_worst_bar": path["strategy_pnl"].min(),
                "strategy_best_bar": path["strategy_pnl"].max(),
            }
        ]
    )

    return summary, path, blotter


def bfix30_zscore_trade_blotter(path, target, hedge):
    """Create one row per z-score trade with trigger and spot levels."""
    if path.empty or "position" not in path.columns:
        return pd.DataFrame()

    positions = path["position"].astype(float)
    rows = []
    open_trade = None
    entry_index = None

    for index_number, timestamp in enumerate(path.index):
        current_position = float(positions.iloc[index_number])
        previous_position = (
            float(positions.iloc[index_number - 1]) if index_number else 0.0
        )
        if current_position == previous_position:
            continue

        signal_index = max(index_number - 1, 0)
        signal_timestamp = path.index[signal_index]
        signal_row = path.iloc[signal_index]
        execution_row = path.iloc[index_number]

        if previous_position != 0 and current_position == 0 and open_trade is not None:
            open_trade.update(
                {
                    "exit_signal_time": signal_timestamp,
                    "exit_time": timestamp,
                    "exit_spread_z": float(signal_row["spread_z"]),
                    "exit_spread_level": float(signal_row["spread_level"]),
                    "exit_signal_target_spot": float(signal_row["target_spot"]),
                    "exit_signal_hedge_spot": float(signal_row["hedge_spot"]),
                    "exit_target_spot": float(execution_row["target_spot"]),
                    "exit_hedge_spot": float(execution_row["hedge_spot"]),
                    "bars_held": index_number - entry_index,
                    "trade_pnl": float(
                        path["strategy_pnl"].iloc[entry_index:index_number].sum()
                    ),
                    "status": "CLOSED",
                }
            )
            rows.append(open_trade)
            open_trade = None
            entry_index = None
            continue

        if previous_position == 0 and current_position != 0:
            direction = (
                "Long target / short hedge"
                if current_position > 0
                else "Short target / long hedge"
            )
            open_trade = {
                "trade_number": len(rows) + 1,
                "target": target,
                "hedge": hedge,
                "direction": direction,
                "signal_time": signal_timestamp,
                "entry_time": timestamp,
                "entry_spread_z": float(signal_row["spread_z"]),
                "entry_spread_level": float(signal_row["spread_level"]),
                "signal_target_spot": float(signal_row["target_spot"]),
                "signal_hedge_spot": float(signal_row["hedge_spot"]),
                "entry_target_spot": float(execution_row["target_spot"]),
                "entry_hedge_spot": float(execution_row["hedge_spot"]),
                "exit_signal_time": pd.NaT,
                "exit_time": pd.NaT,
                "exit_spread_z": np.nan,
                "exit_spread_level": np.nan,
                "exit_signal_target_spot": np.nan,
                "exit_signal_hedge_spot": np.nan,
                "exit_target_spot": np.nan,
                "exit_hedge_spot": np.nan,
                "bars_held": np.nan,
                "trade_pnl": np.nan,
                "status": "OPEN",
            }
            entry_index = index_number

    if open_trade is not None and entry_index is not None:
        open_trade.update(
            {
                "bars_held": len(path) - entry_index,
                "trade_pnl": float(path["strategy_pnl"].iloc[entry_index:].sum()),
                "status": "OPEN",
            }
        )
        rows.append(open_trade)

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 5. Dashboard state and widgets
# ----------------------------------------------------------------------------

bfix30_panel = BFIXPanel(
    prices=pd.DataFrame(),
    returns=pd.DataFrame(),
    quality=pd.DataFrame(),
    raw=pd.DataFrame(),
)

bfix30_matrix_currencies = widgets.SelectMultiple(
    options=list(BFIX30_CURRENCIES.keys()),
    value=("EUR", "GBP", "CHF", "AUD", "CAD"),
    description="Currencies:",
    layout=widgets.Layout(width="360px", height="190px"),
)

bfix30_initial_selection = list(bfix30_matrix_currencies.value)

bfix30_target_currency = widgets.Dropdown(
    options=bfix30_initial_selection,
    value="EUR",
    description="Target:",
    layout=widgets.Layout(width="250px"),
)

bfix30_hedge_currency = widgets.Dropdown(
    options=bfix30_initial_selection,
    value="GBP",
    description="Hedge:",
    layout=widgets.Layout(width="250px"),
)

bfix30_history_dropdown = widgets.Dropdown(
    options=[
        ("5 days", "-5D"),
        ("10 days", "-10D"),
        ("20 days", "-20D"),
        ("1 month", "-1M"),
        ("3 months", "-3M"),
        ("6 months", "-6M"),
        ("1 year", "-1Y"),
    ],
    value="-20D",
    description="History:",
    layout=widgets.Layout(width="250px"),
)

bfix30_return_method = widgets.ToggleButtons(
    options=[("Log", "log"), ("Pct", "pct")],
    value="log",
    description="Returns:",
)

bfix30_fill_single_gap = widgets.Checkbox(
    value=False,
    description="Interpolate single missing 30m gap",
    indent=False,
)

bfix30_winsor_slider = widgets.FloatSlider(
    value=8.0,
    min=0.0,
    max=15.0,
    step=0.5,
    description="Winsor z:",
    readout_format=".1f",
    layout=widgets.Layout(width="300px"),
)

bfix30_window_dropdown = widgets.Dropdown(
    options=[
        ("48 bars - 1 BFIX day", 48),
        ("96 bars - 2 BFIX days", 96),
        ("180 bars - 3.75 BFIX days", 180),
        ("252 bars - 5.25 BFIX days", 252),
        ("480 bars - 10 BFIX days", 480),
    ],
    value=96,
    description="Corr window:",
    layout=widgets.Layout(width="280px"),
)

bfix30_z_window_dropdown = widgets.Dropdown(
    options=[
        ("96 bars - 2 BFIX days", 96),
        ("252 bars - 5.25 BFIX days", 252),
        ("480 bars - 10 BFIX days", 480),
        ("960 bars - 20 BFIX days", 960),
    ],
    value=252,
    description="Spread z:",
    layout=widgets.Layout(width="280px"),
)

bfix30_beta_method = widgets.Dropdown(
    options=[
        ("OLS", "ols"),
        ("OLS + HAC / Newey-West", "hac"),
        ("GLS / AR(1) Omega", "gls"),
    ],
    value="gls",
    description="Beta method:",
    layout=widgets.Layout(width="320px"),
)

bfix30_hac_lags = widgets.Dropdown(
    options=[
        ("1 lag", 1),
        ("2 lags", 2),
        ("4 lags", 4),
        ("8 lags", 8),
        ("12 lags", 12),
    ],
    value=4,
    description="NW lags:",
    layout=widgets.Layout(width="240px"),
)

bfix30_target_notional = widgets.FloatText(
    value=100_000_000.0,
    description="Target notional:",
    layout=widgets.Layout(width="260px"),
)

bfix30_z_signal = widgets.FloatText(
    value=2.0,
    description="Signal |z|:",
    layout=widgets.Layout(width="260px"),
)

bfix30_z_exit = widgets.FloatText(
    value=0.25,
    description="Exit band:",
    layout=widgets.Layout(width="260px"),
)

bfix30_reload_button = widgets.Button(
    description="Reload BFIX panel",
    icon="refresh",
    button_style="primary",
)

bfix30_status = widgets.HTML()
bfix30_quality_output = widgets.Output()
bfix30_matrix_output = widgets.Output()
bfix30_pair_table_output = widgets.Output()
bfix30_beta_comparison_output = widgets.Output()
bfix30_pair_chart_output = widgets.Output()
bfix30_spread_chart_output = widgets.Output()
bfix30_spread_stats_output = widgets.Output()
bfix30_backtest_output = widgets.Output()
bfix30_zscore_backtest_output = widgets.Output()
bfix30_trade_blotter_output = widgets.Output()


# ----------------------------------------------------------------------------
# 6. Rendering functions
# ----------------------------------------------------------------------------

def bfix30_update_quality():
    with bfix30_quality_output:
        clear_output(wait=True)
        if bfix30_panel.quality.empty:
            print("No quality report yet.")
            return

        quality = bfix30_panel.quality.copy()
        quality["coverage"] = quality["coverage"].map(lambda value: f"{value:.1%}")
        display(quality)


def bfix30_update_matrix(*_):
    with bfix30_matrix_output:
        clear_output(wait=True)
        selected = list(bfix30_matrix_currencies.value)

        if bfix30_panel.returns.empty or len(selected) < 2:
            print("Load at least two currencies to show the BFIX matrix.")
            return

        selected_returns = bfix30_panel.returns.reindex(columns=selected).dropna(how="all")
        if selected_returns.empty:
            print("No usable BFIX 30-minute returns are available.")
            return

        correlation = selected_returns.corr()
        start_time = selected_returns.index.min()
        end_time = selected_returns.index.max()

        figure = go.Figure(
            data=go.Heatmap(
                z=correlation.values,
                x=correlation.columns,
                y=correlation.index,
                zmin=-1,
                zmax=1,
                colorscale="RdBu",
                reversescale=True,
                text=np.round(correlation.values, 2),
                texttemplate="%{text:.2f}",
                hovertemplate=(
                    "%{y} versus %{x}<br>"
                    "Correlation: %{z:.3f}<extra></extra>"
                ),
                colorbar=dict(title="Correlation"),
            )
        )
        figure.update_layout(
            title=(
                "BFIX 30-minute return correlation matrix "
                f"({start_time:%Y-%m-%d %H:%M} to {end_time:%Y-%m-%d %H:%M})"
            ),
            height=520,
            margin=dict(l=70, r=30, t=70, b=70),
        )
        figure.show()


def bfix30_format_hedge_table(table):
    formatted = table.copy()

    money_columns = [
        "target_notional",
        "recommended_hedge_notional",
    ]
    decimal_columns = [
        "latest_corr",
        "full_corr",
        "beta",
        "beta_se",
        "beta_tstat",
        "gls_rho",
        "spread_z",
        "half_life_bars",
        "coint_pvalue",
        "adf_pvalue",
        "score",
    ]

    for column in money_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{value:,.0f}"
            )

    for column in decimal_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.4f}"
            )

    return formatted.rename(
        columns={
            "target": "target",
            "hedge": "hedge",
            "observations": "obs",
            "target_notional": "target notional",
            "latest_corr": "latest corr",
            "full_corr": "full corr",
            "beta": "return beta",
            "beta_method": "beta method",
            "beta_se": "beta SE",
            "beta_tstat": "beta t-stat",
            "gls_rho": "GLS rho",
            "hac_lags": "NW lags",
            "spread_z": "spread z",
            "half_life_bars": "half-life bars",
            "coint_pvalue": "coint p",
            "adf_pvalue": "ADF p",
            "recommended_hedge_notional": "recommended hedge notional",
            "score": "score",
        }
    )


def bfix30_format_backtest_summary(summary):
    formatted = summary.copy()

    money_columns = [
        "target_notional",
        "hedge_notional",
        "unhedged_vol_per_bar",
        "hedged_vol_per_bar",
        "unhedged_cumulative_pnl",
        "hedged_cumulative_pnl",
        "hedged_max_drawdown",
        "hedged_worst_bar",
    ]
    decimal_columns = ["beta", "beta_se", "beta_tstat", "gls_rho"]
    percent_columns = ["vol_reduction"]

    for column in money_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{value:,.0f}"
            )

    for column in decimal_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.4f}"
            )

    for column in percent_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.1%}"
            )

    return formatted.rename(
        columns={
            "target": "target",
            "hedge": "hedge",
            "observations": "obs",
            "target_notional": "target notional",
            "beta": "return beta",
            "beta_method": "beta method",
            "beta_se": "beta SE",
            "beta_tstat": "beta t-stat",
            "gls_rho": "GLS rho",
            "hedge_notional": "hedge notional",
            "unhedged_vol_per_bar": "unhedged vol/bar",
            "hedged_vol_per_bar": "hedged vol/bar",
            "vol_reduction": "vol reduction",
            "unhedged_cumulative_pnl": "unhedged cumulative P&L",
            "hedged_cumulative_pnl": "hedged cumulative P&L",
            "hedged_max_drawdown": "hedged max drawdown",
            "hedged_worst_bar": "hedged worst bar",
        }
    )


def bfix30_format_zscore_backtest_summary(summary):
    formatted = summary.copy()

    money_columns = [
        "target_notional",
        "strategy_vol_per_bar",
        "strategy_cumulative_pnl",
        "strategy_max_drawdown",
        "strategy_worst_bar",
        "strategy_best_bar",
    ]
    decimal_columns = [
        "signal_z",
        "exit_z",
        "spread_beta",
        "hedge_beta",
    ]
    percent_columns = ["active_pct"]

    for column in money_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{value:,.0f}"
            )

    for column in decimal_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.4f}"
            )

    for column in percent_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.1%}"
            )

    return formatted.rename(
        columns={
            "target": "target",
            "hedge": "hedge",
            "observations": "obs",
            "signal_z": "signal z",
            "exit_z": "exit band",
            "spread_beta": "spread beta",
            "hedge_beta": "hedge beta",
            "beta_method": "beta method",
            "target_notional": "target notional",
            "active_bars": "active bars",
            "active_pct": "active %",
            "signal_triggers": "signal triggers",
            "completed_trades": "completed trades",
            "open_trades": "open trades",
            "strategy_vol_per_bar": "strategy vol/bar",
            "strategy_cumulative_pnl": "strategy cumulative P&L",
            "strategy_max_drawdown": "strategy max drawdown",
            "strategy_worst_bar": "strategy worst bar",
            "strategy_best_bar": "strategy best bar",
        }
    )


def bfix30_format_trade_blotter(blotter):
    formatted = blotter.copy()
    datetime_columns = [
        "signal_time",
        "entry_time",
        "exit_signal_time",
        "exit_time",
    ]
    spot_columns = [
        "signal_target_spot",
        "signal_hedge_spot",
        "entry_target_spot",
        "entry_hedge_spot",
        "exit_signal_target_spot",
        "exit_signal_hedge_spot",
        "exit_target_spot",
        "exit_hedge_spot",
        "entry_spread_level",
        "exit_spread_level",
        "entry_spread_z",
        "exit_spread_z",
    ]

    for column in datetime_columns:
        if column in formatted.columns:
            formatted[column] = pd.to_datetime(formatted[column]).map(
                lambda value: "" if pd.isna(value) else value.strftime("%Y-%m-%d %H:%M")
            )

    for column in spot_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.6f}"
            )

    if "bars_held" in formatted.columns:
        formatted["bars_held"] = formatted["bars_held"].map(
            lambda value: "" if pd.isna(value) else f"{int(value)}"
        )
    if "trade_pnl" in formatted.columns:
        formatted["trade_pnl"] = formatted["trade_pnl"].map(
            lambda value: "" if pd.isna(value) else f"{value:,.0f}"
        )

    return formatted.rename(
        columns={
            "trade_number": "trade",
            "target": "target",
            "hedge": "hedge",
            "direction": "direction",
            "signal_time": "signal time",
            "entry_time": "entry time",
            "entry_spread_z": "signal z",
            "entry_spread_level": "signal spread level",
            "signal_target_spot": "signal target spot",
            "signal_hedge_spot": "signal hedge spot",
            "entry_target_spot": "entry target spot",
            "entry_hedge_spot": "entry hedge spot",
            "exit_signal_time": "exit signal time",
            "exit_time": "exit time",
            "exit_spread_z": "exit z",
            "exit_spread_level": "exit spread level",
            "exit_signal_target_spot": "exit signal target spot",
            "exit_signal_hedge_spot": "exit signal hedge spot",
            "exit_target_spot": "exit target spot",
            "exit_hedge_spot": "exit hedge spot",
            "bars_held": "bars held",
            "trade_pnl": "trade P&L",
            "status": "status",
        }
    )


def bfix30_current_pair_stats():
    if bfix30_panel.prices.empty or bfix30_panel.returns.empty:
        return None

    target = bfix30_target_currency.value
    hedge = bfix30_hedge_currency.value
    if (
        target not in bfix30_panel.prices.columns
        or hedge not in bfix30_panel.prices.columns
        or target not in bfix30_panel.returns.columns
        or hedge not in bfix30_panel.returns.columns
    ):
        return None

    return bfix30_pair_stats(
        prices=bfix30_panel.prices,
        returns=bfix30_panel.returns,
        target=target,
        hedge=hedge,
        rolling_window=bfix30_window_dropdown.value,
        z_window=bfix30_z_window_dropdown.value,
        target_notional=bfix30_target_notional.value,
        beta_method=bfix30_beta_method.value,
        hac_lags=bfix30_hac_lags.value,
    )


def bfix30_update_pair_table(*_):
    with bfix30_pair_table_output:
        clear_output(wait=True)
        selected = list(bfix30_matrix_currencies.value)

        if bfix30_panel.returns.empty:
            print("Reload BFIX data to rank proxy hedges.")
            return

        table = bfix30_pair_table(
            prices=bfix30_panel.prices,
            returns=bfix30_panel.returns,
            selected=selected,
            target=bfix30_target_currency.value,
            rolling_window=bfix30_window_dropdown.value,
            z_window=bfix30_z_window_dropdown.value,
            target_notional=bfix30_target_notional.value,
            beta_method=bfix30_beta_method.value,
            hac_lags=bfix30_hac_lags.value,
        )

        if table.empty:
            print("No usable proxy hedge pairs for the current selection.")
            return

        columns = [
            "target",
            "hedge",
            "observations",
            "target_notional",
            "latest_corr",
            "full_corr",
            "beta",
            "beta_method",
            "beta_se",
            "beta_tstat",
            "gls_rho",
            "hac_lags",
            "spread_z",
            "half_life_bars",
            "coint_pvalue",
            "adf_pvalue",
            "recommended_hedge_notional",
            "score",
        ]
        display(bfix30_format_hedge_table(table.loc[:, columns]))


def bfix30_update_beta_comparison(*_):
    with bfix30_beta_comparison_output:
        clear_output(wait=True)

        if bfix30_panel.returns.empty:
            print("Reload BFIX data to compare beta methods.")
            return

        target = bfix30_target_currency.value
        hedge = bfix30_hedge_currency.value
        required = [target, hedge]
        if target == hedge or any(
            currency not in bfix30_panel.returns.columns for currency in required
        ):
            print("Load data and choose a valid target/hedge pair.")
            return

        paired_returns = bfix30_panel.returns.reindex(columns=required).dropna()
        rows = []
        labels = {
            "ols": "OLS",
            "hac": "OLS + HAC / Newey-West",
            "gls": "GLS / AR(1) Omega",
        }
        for method in ("ols", "hac", "gls"):
            beta, intercept, beta_se, beta_tstat, gls_rho = bfix30_estimate_beta(
                paired_returns[target],
                paired_returns[hedge],
                method=method,
                hac_lags=bfix30_hac_lags.value,
            )
            rows.append(
                {
                    "method": labels[method],
                    "selected": "Yes" if method == bfix30_beta_method.value else "",
                    "beta": beta,
                    "beta_se": beta_se,
                    "beta_tstat": beta_tstat,
                    "gls_rho": gls_rho,
                    "hedge_notional": (
                        -float(bfix30_target_notional.value) * beta
                        if np.isfinite(beta)
                        else np.nan
                    ),
                }
            )

        comparison = pd.DataFrame(rows)
        for column in ["beta", "beta_se", "beta_tstat", "gls_rho"]:
            comparison[column] = comparison[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.5f}"
            )
        comparison["hedge_notional"] = comparison["hedge_notional"].map(
            lambda value: "" if pd.isna(value) else f"{value:,.0f}"
        )
        display(
            comparison.rename(
                columns={
                    "method": "method",
                    "selected": "selected for hedge",
                    "beta": "beta",
                    "beta_se": "beta SE",
                    "beta_tstat": "beta t-stat",
                    "gls_rho": "GLS rho",
                    "hedge_notional": "hedge notional",
                }
            )
        )


def bfix30_update_pair_chart(*_):
    with bfix30_pair_chart_output:
        clear_output(wait=True)

        target = bfix30_target_currency.value
        hedge = bfix30_hedge_currency.value
        window = bfix30_window_dropdown.value

        if bfix30_panel.returns.empty:
            print("Reload BFIX data to display rolling correlation.")
            return

        if target == hedge:
            print("Choose two different currencies.")
            return

        required = [target, hedge]
        missing = [
            currency
            for currency in required
            if currency not in bfix30_panel.returns.columns
        ]
        if missing:
            print(
                "The loaded BFIX return panel is missing: "
                f"{', '.join(missing)}. Reload with those currencies selected."
            )
            return

        paired_returns = bfix30_panel.returns.reindex(columns=required).dropna()
        if len(paired_returns) < window:
            print(
                f"Not enough overlapping 30-minute BFIX bars for a "
                f"{window}-bar window. Available: {len(paired_returns)} bars."
            )
            return

        rolling_correlation = (
            paired_returns[target]
            .rolling(window=window, min_periods=window)
            .corr(paired_returns[hedge])
        )

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=rolling_correlation.index,
                y=rolling_correlation,
                name=f"{target} versus {hedge}",
                line=dict(color="#0f62fe", width=3),
            )
        )
        figure.add_hline(y=0, line_dash="dash", line_color="gray")
        figure.update_yaxes(range=[-1, 1], title="Correlation")
        figure.update_xaxes(title="Timestamp")
        figure.update_layout(
            title=(
                f"{window}-bar rolling correlation: {target} versus {hedge} "
                f"({len(paired_returns)} overlapping returns)"
            ),
            height=420,
            margin=dict(l=60, r=30, t=70, b=55),
            hovermode="x unified",
        )
        figure.show()


def bfix30_update_spread_chart(*_):
    with bfix30_spread_chart_output:
        clear_output(wait=True)

        target = bfix30_target_currency.value
        hedge = bfix30_hedge_currency.value
        stats = bfix30_current_pair_stats()

        if stats is None:
            print("Load data and choose a valid target/hedge pair.")
            return

        required = [target, hedge]
        missing = [
            currency
            for currency in required
            if currency not in bfix30_panel.prices.columns
        ]
        if missing:
            print(
                "The loaded BFIX price panel is missing: "
                f"{', '.join(missing)}. Reload with those currencies selected."
            )
            return

        spread_z, price_beta, _ = bfix30_pair_spread_zscore(
            prices=bfix30_panel.prices,
            target=target,
            hedge=hedge,
            z_window=bfix30_z_window_dropdown.value,
        )

        if spread_z.empty:
            print("Not enough price history to calculate the spread z-score.")
            return

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=spread_z.index,
                y=spread_z,
                name="Spread z-score",
                line=dict(color="#7c3aed", width=2.5),
            )
        )
        figure.add_hline(y=0, line_dash="dash", line_color="gray")
        figure.add_hline(y=2, line_dash="dot", line_color="#b91c1c")
        figure.add_hline(y=-2, line_dash="dot", line_color="#15803d")
        figure.update_yaxes(title="Z-score")
        figure.update_xaxes(title="Timestamp")
        figure.update_layout(
            title=(
                f"Cointegration spread z-score: {target} minus beta x {hedge} "
                f"| beta={price_beta:.3f}, coint p={stats.coint_pvalue:.4f}, "
                f"ADF p={stats.adf_pvalue:.4f}"
            ),
            height=420,
            margin=dict(l=60, r=30, t=70, b=55),
            hovermode="x unified",
        )
        figure.show()


def bfix30_update_spread_stats(*_):
    with bfix30_spread_stats_output:
        clear_output(wait=True)

        target = bfix30_target_currency.value
        hedge = bfix30_hedge_currency.value
        z_window = bfix30_z_window_dropdown.value

        if bfix30_panel.prices.empty:
            print("Reload BFIX data to display rolling spread statistics.")
            return

        levels, price_beta, _ = bfix30_pair_spread_levels(
            prices=bfix30_panel.prices,
            target=target,
            hedge=hedge,
            z_window=z_window,
        )

        if levels.empty:
            print("Not enough price history to calculate rolling spread statistics.")
            return

        usable = levels.dropna(
            subset=["spread_level", "rolling_mean", "rolling_std"]
        )
        if usable.empty:
            print(
                "The selected z-score window does not yet have enough observations "
                "for a rolling mean and standard deviation."
            )
            return

        latest = usable.iloc[-1]
        latest_table = pd.DataFrame(
            [
                {
                    "timestamp": usable.index[-1],
                    "spread level": latest["spread_level"],
                    "rolling mean": latest["rolling_mean"],
                    "rolling std dev": latest["rolling_std"],
                    "upper 1 std band": latest["rolling_mean"] + latest["rolling_std"],
                    "lower 1 std band": latest["rolling_mean"] - latest["rolling_std"],
                    "z-score": latest["spread_z"],
                    "window bars": z_window,
                }
            ]
        )

        numeric_columns = [
            "spread level",
            "rolling mean",
            "rolling std dev",
            "upper 1 std band",
            "lower 1 std band",
            "z-score",
        ]
        for column in numeric_columns:
            latest_table[column] = latest_table[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.8f}"
            )
        latest_table["timestamp"] = latest_table["timestamp"].map(
            lambda value: value.strftime("%Y-%m-%d %H:%M")
        )
        display(latest_table)

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=usable.index,
                y=usable["spread_level"],
                name="Spread level",
                line=dict(color="#7c3aed", width=2.5),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=usable.index,
                y=usable["rolling_mean"],
                name="Rolling mean",
                line=dict(color="#0f62fe", width=2),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=usable.index,
                y=usable["rolling_mean"] + usable["rolling_std"],
                name="Mean + 1 std",
                line=dict(color="#94a3b8", width=1, dash="dot"),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=usable.index,
                y=usable["rolling_mean"] - usable["rolling_std"],
                name="Mean - 1 std",
                line=dict(color="#94a3b8", width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(148, 163, 184, 0.15)",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=usable.index,
                y=usable["rolling_std"],
                name="Rolling std dev",
                yaxis="y2",
                line=dict(color="#f97316", width=2),
            )
        )
        figure.update_layout(
            title=(
                f"Rolling spread mean and standard deviation: {target} versus {hedge} "
                f"({z_window} bars, price beta={price_beta:.4f})"
            ),
            height=460,
            margin=dict(l=60, r=65, t=70, b=55),
            hovermode="x unified",
            yaxis=dict(title="Spread level"),
            yaxis2=dict(
                title="Rolling standard deviation",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
        )
        figure.update_xaxes(title="Timestamp")
        figure.show()


def bfix30_update_backtest(*_):
    with bfix30_backtest_output:
        clear_output(wait=True)

        target = bfix30_target_currency.value
        hedge = bfix30_hedge_currency.value
        stats = bfix30_current_pair_stats()

        if stats is None:
            print("Load data and choose a valid target/hedge pair.")
            return

        summary, path = bfix30_backtest_pair(
            returns=bfix30_panel.returns,
            target=target,
            hedge=hedge,
            target_notional=bfix30_target_notional.value,
            beta=stats.beta,
        )

        if summary.empty or path.empty:
            print("No usable overlapping return history for this backtest.")
            return

        summary["beta_method"] = stats.beta_method
        summary["beta_se"] = stats.beta_se
        summary["beta_tstat"] = stats.beta_tstat
        summary["gls_rho"] = stats.gls_rho
        display(bfix30_format_backtest_summary(summary))

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=path.index,
                y=path["cumulative_unhedged_pnl"],
                name=f"Unhedged {target}",
                line=dict(color="#64748b", width=2),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=path.index,
                y=path["cumulative_hedged_pnl"],
                name=f"Hedged with {hedge}",
                line=dict(color="#0f62fe", width=3),
            )
        )
        figure.add_hline(y=0, line_dash="dash", line_color="gray")
        figure.update_yaxes(title="Cumulative P&L")
        figure.update_xaxes(title="Timestamp")
        figure.update_layout(
            title=(
                f"Backtest: {target} {bfix30_target_notional.value:,.0f} "
                f"hedged with {hedge} {stats.recommended_hedge_notional:,.0f} "
                f"(beta={stats.beta:.3f}, method={stats.beta_method})"
            ),
            height=420,
            margin=dict(l=60, r=30, t=70, b=55),
            hovermode="x unified",
        )
        figure.show()


def bfix30_update_zscore_backtest(*_):
    with bfix30_trade_blotter_output:
        clear_output(wait=True)

    with bfix30_zscore_backtest_output:
        clear_output(wait=True)

        target = bfix30_target_currency.value
        hedge = bfix30_hedge_currency.value

        if bfix30_panel.prices.empty or bfix30_panel.returns.empty:
            print("Reload BFIX data to run the z-score signal backtest.")
            with bfix30_trade_blotter_output:
                print("No trade blotter until the signal backtest has usable data.")
            return

        exit_band = abs(float(bfix30_z_exit.value))
        signal_z = abs(float(bfix30_z_signal.value))

        if signal_z <= exit_band:
            print("Signal |z| should be outside the exit band.")
            with bfix30_trade_blotter_output:
                print("Signal |z| should be outside the exit band.")
            return

        stats = bfix30_current_pair_stats()
        if stats is None:
            print("Load data and choose a valid target/hedge pair.")
            with bfix30_trade_blotter_output:
                print("Load data and choose a valid target/hedge pair.")
            return

        summary, path, blotter = bfix30_zscore_signal_backtest(
            prices=bfix30_panel.prices,
            returns=bfix30_panel.returns,
            target=target,
            hedge=hedge,
            target_notional=bfix30_target_notional.value,
            z_window=bfix30_z_window_dropdown.value,
            signal_z=signal_z,
            exit_z=exit_band,
            hedge_beta=stats.beta,
            beta_method=stats.beta_method,
        )

        if summary.empty or path.empty:
            print("No usable history for the z-score signal backtest.")
            with bfix30_trade_blotter_output:
                print("No usable history for the z-score trade blotter.")
            return

        display(bfix30_format_zscore_backtest_summary(summary))

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=path.index,
                y=path["cumulative_strategy_pnl"],
                name="Z-score signal P&L",
                line=dict(color="#16a34a", width=3),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=path.index,
                y=path["position"],
                name="Position",
                yaxis="y2",
                line=dict(color="#f97316", width=1.5, dash="dot"),
            )
        )
        figure.add_hline(y=0, line_dash="dash", line_color="gray")
        figure.update_layout(
            title=(
                f"Z-score signal backtest: {target} versus {hedge} "
                f"(signal |z|={signal_z:.2f}, exit band={exit_band:.2f})"
            ),
            height=440,
            margin=dict(l=60, r=55, t=70, b=55),
            hovermode="x unified",
            yaxis=dict(title="Cumulative P&L"),
            yaxis2=dict(
                title="Position",
                overlaying="y",
                side="right",
                range=[-1.2, 1.2],
                showgrid=False,
            ),
        )
        figure.update_xaxes(title="Timestamp")
        figure.show()

    with bfix30_trade_blotter_output:
        if blotter.empty:
            print(
                f"No {target}/{hedge} signals crossed the selected "
                f"|z|={signal_z:.2f} threshold."
            )
        else:
            display(bfix30_format_trade_blotter(blotter))


def bfix30_update_all_charts(*_):
    bfix30_update_quality()
    bfix30_update_matrix()
    bfix30_update_pair_table()
    bfix30_update_beta_comparison()
    bfix30_update_pair_chart()
    bfix30_update_spread_chart()
    bfix30_update_spread_stats()
    bfix30_update_backtest()
    bfix30_update_zscore_backtest()


# ----------------------------------------------------------------------------
# 7. Reload and callback functions
# ----------------------------------------------------------------------------

def bfix30_reload_history(*_):
    global bfix30_panel

    selected = list(bfix30_matrix_currencies.value)
    if len(selected) < 2:
        bfix30_status.value = (
            "<b style='color:#b91c1c'>Select at least two BFIX currencies.</b>"
        )
        return

    try:
        security_count = len(selected) * len(BFIX30_SLOTS)
        bfix30_status.value = (
            f"<span style='color:#1d4ed8'>Loading {security_count} BFIX "
            f"securities for {bfix30_history_dropdown.label}...</span>"
        )

        bfix30_panel = bfix30_construct_panel(
            currency_labels=selected,
            history_start=bfix30_history_dropdown.value,
            return_method=bfix30_return_method.value,
            align_frequency=True,
            fill_single_gap=bfix30_fill_single_gap.value,
            winsor_z=bfix30_winsor_slider.value,
        )

        if bfix30_panel.prices.empty:
            raise ValueError(
                "No BFIX rows were returned. Test one known ticker, such as "
                "EUR F000 Curncy, in a separate BQL request first."
            )

        if bfix30_panel.returns.empty:
            raise ValueError(
                "BFIX prices were returned, but no usable 30-minute returns exist."
            )

        start_time = bfix30_panel.prices.index.min()
        end_time = bfix30_panel.prices.index.max()

        bfix30_update_all_charts()

        statsmodels_note = ""
        if coint is None or adfuller is None:
            statsmodels_note = " Cointegration p-values require statsmodels."

        bfix30_status.value = (
            f"<span style='color:#15803d'>Loaded "
            f"{len(bfix30_panel.prices):,} aligned BFIX timestamps from "
            f"{start_time:%Y-%m-%d %H:%M} through {end_time:%Y-%m-%d %H:%M}. "
            f"Requested {security_count} BFIX securities.{statsmodels_note}</span>"
        )

    except Exception as exc:
        bfix30_status.value = (
            "<b style='color:#b91c1c'>BFIX 30-minute historical-data request "
            "failed.</b><br>"
            f"<code>{type(exc).__name__}: {exc}</code>"
        )


def bfix30_on_currency_selection_change(change):
    selected = list(change["new"])
    bfix30_target_currency.options = selected
    bfix30_hedge_currency.options = selected

    if len(selected) < 2:
        return

    if bfix30_target_currency.value not in selected:
        bfix30_target_currency.value = selected[0]
    if bfix30_hedge_currency.value not in selected:
        bfix30_hedge_currency.value = selected[1]

    bfix30_reload_history()


bfix30_reload_button.on_click(bfix30_reload_history)
bfix30_matrix_currencies.observe(bfix30_on_currency_selection_change, names="value")
bfix30_history_dropdown.observe(bfix30_reload_history, names="value")
bfix30_return_method.observe(bfix30_reload_history, names="value")
bfix30_fill_single_gap.observe(bfix30_reload_history, names="value")
bfix30_winsor_slider.observe(bfix30_reload_history, names="value")

for widget in [
    bfix30_target_currency,
    bfix30_hedge_currency,
    bfix30_window_dropdown,
    bfix30_z_window_dropdown,
    bfix30_beta_method,
    bfix30_hac_lags,
    bfix30_target_notional,
    bfix30_z_signal,
    bfix30_z_exit,
]:
    widget.observe(bfix30_update_all_charts, names="value")


# ----------------------------------------------------------------------------
# 8. Layout
# ----------------------------------------------------------------------------

bfix30_data_controls = widgets.VBox(
    [
        widgets.HTML("<h3>1. BFIX 30-minute data panel</h3>"),
        widgets.HTML(
            "Loads all 48 BFIX fixing times per selected currency and aligns "
            "them to a 30-minute timestamp grid."
        ),
        bfix30_matrix_currencies,
        bfix30_history_dropdown,
        bfix30_return_method,
        bfix30_fill_single_gap,
        bfix30_winsor_slider,
        bfix30_reload_button,
    ],
    layout=widgets.Layout(width="440px"),
)

bfix30_pair_controls = widgets.VBox(
    [
        widgets.HTML("<h3>2. Proxy hedge analytics</h3>"),
        bfix30_target_currency,
        bfix30_hedge_currency,
        bfix30_window_dropdown,
        bfix30_z_window_dropdown,
        bfix30_beta_method,
        bfix30_hac_lags,
        bfix30_target_notional,
        bfix30_z_signal,
        bfix30_z_exit,
        widgets.HTML(
            "<span style='font-size:12px;color:#64748b'>"
            "Recommended hedge notional = -target notional x return beta."
            "</span>"
        ),
    ],
    layout=widgets.Layout(width="440px"),
)

bfix30_dashboard = widgets.VBox(
    [
        widgets.HTML("<h1>BFIX 30-Minute DV01 Hedge Dashboard</h1>"),
        widgets.HTML(
            "<i>Correlation shows hedge similarity; cointegration/spread z-score "
            "helps identify rich/cheap proxy timing; target notional and beta "
            "translate into recommended hedge notional.</i>"
        ),
        widgets.HBox([bfix30_data_controls, bfix30_pair_controls]),
        bfix30_status,
        widgets.HTML("<hr><h3>Data quality</h3>"),
        bfix30_quality_output,
        widgets.HTML("<hr><h3>Correlation matrix</h3>"),
        bfix30_matrix_output,
        widgets.HTML("<hr><h3>Ranked proxy hedge candidates</h3>"),
        bfix30_pair_table_output,
        widgets.HTML("<hr><h3>Selected pair beta methods</h3>"),
        bfix30_beta_comparison_output,
        widgets.HTML("<hr><h3>Selected pair hedge backtest</h3>"),
        bfix30_backtest_output,
        widgets.HTML("<hr><h3>Z-score signal backtest</h3>"),
        bfix30_zscore_backtest_output,
        widgets.HTML("<hr><h3>Z-score trade blotter</h3>"),
        bfix30_trade_blotter_output,
        widgets.HTML("<hr><h3>Rolling correlation</h3>"),
        bfix30_pair_chart_output,
        widgets.HTML("<hr><h3>Cointegration spread z-score</h3>"),
        bfix30_spread_chart_output,
        widgets.HTML("<hr><h3>Rolling spread mean and standard deviation</h3>"),
        bfix30_spread_stats_output,
    ]
)

display(bfix30_dashboard)


# Initial request. Start with 20 days; increase after confirming request speed.
bfix30_reload_history()
