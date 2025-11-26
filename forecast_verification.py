"""
General-purpose forecast evaluation utilities for continuous variables.

Author: Prasad Deshpande
Source: https://github.com/hydroinfo-prasad/Kansas_State_Mesonet_Tools/

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from scipy import stats


# ---------------------------------------------------------------------
# Core Metrics
# ---------------------------------------------------------------------

def evaluate_forecast(obs, forecast):
    """
    Computes performance metrics for two continuous time series.

    Parameters
    ----------
    obs : array-like
        Observed values.
    forecast : array-like
        Forecasted values.

    Returns
    -------
    metrics : dict
        Bias, MAE, RMSE, R, R², NSE, MBE, MAPE.
    """

    obs = np.array(obs, dtype=float)
    forecast = np.array(forecast, dtype=float)

    mask = ~np.isnan(obs) & ~np.isnan(forecast)
    obs = obs[mask]
    forecast = forecast[mask]

    if len(obs) == 0:
        raise ValueError("No valid data after removing NaNs.")

    bias = np.mean(forecast - obs)
    mae = mean_absolute_error(obs, forecast)
    rmse = np.sqrt(mean_squared_error(obs, forecast))
    r = np.corrcoef(obs, forecast)[0, 1]
    r2 = r2_score(obs, forecast)

    # Nash–Sutcliffe efficiency
    nse = 1 - np.sum((forecast - obs) ** 2) / np.sum((obs - np.mean(obs)) ** 2)

    # Mean Bias Error
    mbe = np.mean(forecast - obs)

    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((obs - forecast) / obs)) * 100

    metrics = {
        "Bias": bias,
        "MAE": mae,
        "RMSE": rmse,
        "R": r,
        "R2": r2,
        "NSE": nse,
        "MBE": mbe,
        "MAPE (%)": mape,
        "N_Samples": len(obs),
    }

    return metrics


# ---------------------------------------------------------------------
# Plotting Utilities
# ---------------------------------------------------------------------

def plot_scatter(obs, forecast, var_name="Variable"):
    plt.figure(figsize=(6, 6))
    plt.scatter(obs, forecast, alpha=0.5)
    mn, mx = min(obs.min(), forecast.min()), max(obs.max(), forecast.max())
    plt.plot([mn, mx], [mn, mx], '--r', label="1:1 Line")
    plt.xlabel(f"Observed {var_name}")
    plt.ylabel(f"Forecast {var_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_time_series(time, obs, forecast, var_name="Variable"):
    plt.figure(figsize=(10, 5))
    plt.plot(time, obs, label="Observed", marker="o", alpha=0.6)
    plt.plot(time, forecast, label="Forecast", marker="x", alpha=0.6)
    plt.xlabel("Time")
    plt.ylabel(var_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_histogram(obs, forecast, var_name="Variable"):
    plt.figure(figsize=(6, 4))
    plt.hist(obs, bins=30, alpha=0.5, label="Observed")
    plt.hist(forecast, bins=30, alpha=0.5, label="Forecast")
    plt.xlabel(var_name)
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_residuals(obs, forecast):
    residuals = obs - forecast

    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30, density=True, edgecolor="black", alpha=0.7)
    plt.xlabel("Residuals (Obs - Forecast)")
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_qq(obs, forecast):
    residuals = obs - forecast
    plt.figure(figsize=(6, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Master Function (evaluation + optional plots)
# ---------------------------------------------------------------------

def run_full_evaluation(
    obs,
    forecast,
    time=None,
    var_name="Variable",
    scatter=True,
    time_series=True,
    histogram=True,
    residuals=True,
    qq=True
):
    """
    Full evaluation workflow for any continuous forecast variable.

    Parameters
    ----------
    obs : array-like
        Observed values.
    forecast : array-like
        Forecasted values.
    time : array-like or None
        Optional timestamps for time-series plotting.
    var_name : str
        Name to display on plots.
    scatter, time_series, histogram, residuals, qq : bool
        Plot toggles.

    Returns
    -------
    metrics : dict
        Forecast evaluation metrics.
    """

    metrics = evaluate_forecast(obs, forecast)

    if scatter:
        plot_scatter(obs, forecast, var_name)

    if time_series and time is not None:
        plot_time_series(time, obs, forecast, var_name)

    if histogram:
        plot_histogram(obs, forecast, var_name)

    if residuals:
        plot_residuals(obs, forecast)

    if qq:
        plot_qq(obs, forecast)

    return metrics
