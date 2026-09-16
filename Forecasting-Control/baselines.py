"""Naive, classical and linear baselines.

All predictors are fitted on the training split only and evaluated on the test
split. The seasonal ARIMA path uses one-step filtered predictions at h=1 (no
look-ahead) and rolling-origin dynamic forecasts at h=24.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def climatology(y_train, n_test):
    return np.full(n_test, float(np.mean(y_train)))


def persistence(series_test_lag):
    """Forecast equals the most recent observation."""
    return np.asarray(series_test_lag, float)


def seasonal_naive(series, test_idx, period: int = 24):
    v = np.asarray(series, float)
    return v[np.asarray(test_idx) - period]


def ridge_forecast(X_tr, y_tr, X_te, alpha: float = 1.0):
    scaler = StandardScaler().fit(X_tr)
    model = Ridge(alpha=alpha).fit(scaler.transform(X_tr), y_tr)
    return model.predict(scaler.transform(X_te)), model


def sarima_forecast(train, test, horizon: int = 1, order=(1, 0, 1),
                    seasonal_order=(1, 0, 1, 24), exog_train=None, exog_test=None):
    """Seasonal ARIMA / SARIMAX predictions on the test split.

    h=1  : one-step filtered predictions from the fitted state-space model.
    h=24 : rolling-origin 24-step dynamic forecasts at every test origin.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    train = np.asarray(train, float)
    test = np.asarray(test, float)
    mu = train.mean()

    fit = SARIMAX(
        train - mu,
        exog=None if exog_train is None else np.asarray(exog_train, float),
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    full = np.concatenate([train, test]) - mu
    exog_full = None
    if exog_train is not None:
        exog_full = np.concatenate(
            [np.asarray(exog_train, float), np.asarray(exog_test, float)]
        )
    applied = fit.apply(full, exog=exog_full, refit=False)

    n_tr = len(train)
    if horizon == 1:
        pred = applied.get_prediction(start=n_tr, dynamic=False).predicted_mean
        return np.asarray(pred) + mu

    out = []
    for origin in range(n_tr, n_tr + len(test) - horizon + 1):
        f = applied.get_prediction(
            start=origin, end=origin + horizon - 1, dynamic=True
        ).predicted_mean
        out.append(float(np.asarray(f)[-1]))
    return np.asarray(out) + mu
