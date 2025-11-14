import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None


# ---------------------------------------------------------
# Fast rolling standard deviation using cumulative sums
# ---------------------------------------------------------
def fast_rolling_std(a, window):
    cumsum = np.cumsum(np.insert(a, 0, 0))
    cumsum2 = np.cumsum(np.insert(a**2, 0, 0))

    mean = (cumsum[window:] - cumsum[:-window]) / window
    mean2 = (cumsum2[window:] - cumsum2[:-window]) / window
    std = np.sqrt(np.maximum(mean2 - mean**2, 0))

    pad = np.full(window - 1, std[0])
    return np.concatenate([pad, std])


# ---------------------------------------------------------
# Adaptive Exposure Model
# ---------------------------------------------------------
def compute_exposure(df):
    close = df["QQQ_close"].values
    vix = df["VIX"].values
    lt = df["LT_Score"].values
    st = df["ST_Score"].values

    # Daily returns
    ret = np.zeros_like(close, dtype=float)
    ret[1:] = (close[1:] / close[:-1]) - 1
    df["ret"] = ret

    # Realized volatility
    vol20 = fast_rolling_std(ret, 20) * np.sqrt(252)
    df["realized_vol_20"] = np.where(np.isnan(vol20), np.nanmedian(vol20), vol20)

    # Normalized features
    LT_norm = lt / np.nanmax(lt)
    ST_norm = st / np.nanmax(st)

    rolling_median_vol = (
        pd.Series(df["realized_vol_20"])
        .rolling(250, min_periods=20)
        .median()
        .fillna(method="bfill")
    )
    vol_norm = df["realized_vol_20"] / rolling_median_vol
    vol_norm = vol_norm.fillna(1.0).values

    # -----------------------------------------------------
    # Exposure logic
    # -----------------------------------------------------
    vol_weight = 1 / np.sqrt(np.maximum(vol_norm, 0.5))
    momentum = 1.8 * LT_norm + 1.3 * ST_norm
    risk_adj = momentum * vol_weight

    # Smooth nonlinear activation
    exp = 2.5 * np.tanh(risk_adj - 0.5)

    # VIX-based risk throttle
    exp *= np.where(vix < 15, 1.2, 1.0)
    exp *= np.where(vix > 25, 0.6, 1.0)

    # Mean-reversion dampener
    roll_cum = pd.Series(ret).rolling(10, min_periods=3).sum().fillna(0)
    exp *= np.where(roll_cum > 0.04, 0.8, 1.0)

    # Final smoothing
    kernel = np.ones(7) / 7
    exp = np.convolve(exp, kernel, mode="same")

    # Clip
    df["exposure"] = np.clip(exp, -1.0, 2.0)

    return df


# ---------------------------------------------------------
# Backtest helper
# ---------------------------------------------------------
def run_backtest(df):
    df["strategy_ret"] = df["exposure"].shift(1) * df["QQQ_today_close_to_tmrw_close_return"]
    df["cum_equity"] = (1 + df["strategy_ret"]).cumprod()
    df["bh_equity"] = (1 + df["QQQ_today_close_to_tmrw_close_return"]).cumprod()
    return df
