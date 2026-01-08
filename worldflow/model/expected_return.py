from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


@dataclass
class ExpectedReturnConfig:
    train_window: int = 252
    ridge_alpha: float = 5.0
    neighbor_lambda: float = 1.0
    regime_mixture: bool = True


def _safe_weights(w: np.ndarray) -> Optional[np.ndarray]:
    if w is None:
        return None
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 1e-12:
        return None
    return w

def _fit_ridge(X_train: np.ndarray, y: np.ndarray, alpha: float, sample_weight: Optional[np.ndarray] = None):
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X_train)
    model = Ridge(alpha=alpha, fit_intercept=True)
    # sklearn Ridge supports sample_weight in recent versions
    try:
        model.fit(Xs, y, sample_weight=sample_weight)
    except TypeError:
        model.fit(Xs, y)
    return model, scaler


def fit_predict_day(
    change: pd.DataFrame,
    factors: pd.DataFrame,
    W: pd.DataFrame,
    t_idx: int,
    cfg: ExpectedReturnConfig,
    risk_off_prob: Optional[pd.Series] = None
) -> Tuple[pd.Series, Dict[str, Dict[str, float]]]:
    """Predict change for all instruments on date index t_idx."""
    date = change.index[t_idx]
    start = max(0, t_idx - cfg.train_window)
    train_idx = change.index[start:t_idx]  # up to t-1

    insts = [c for c in change.columns if c in W.index]
    if len(insts) < 3:
        return pd.Series(index=change.columns, dtype=float), {}

    rt = change.loc[date, insts]
    neighbor_t = W.loc[insts, insts].dot(rt).rename("neighbor") * cfg.neighbor_lambda

    neigh_train = []
    for d in train_idx:
        rday = change.loc[d, insts]
        neigh_train.append(W.loc[insts, insts].dot(rday).rename(d))
    neighbor_hist = pd.DataFrame(neigh_train, index=train_idx) * cfg.neighbor_lambda

    Xf_train = factors.reindex(train_idx)
    Xf_t = factors.reindex([date])

    preds = pd.Series(index=change.columns, dtype=float)
    explain: Dict[str, Dict[str, float]] = {}

    # regime weights for training window
    p_off_t = 0.0
    w_off = None
    w_on = None
    if risk_off_prob is not None:
        p_off = risk_off_prob.reindex(train_idx).fillna(method="ffill").fillna(0.0).clip(0.0, 1.0)
        w_off = p_off.values.astype(float)
        w_on = (1.0 - p_off.values).astype(float)
        p_off_t = float(risk_off_prob.reindex([date]).fillna(method="ffill").fillna(0.0).clip(0.0, 1.0).iloc[0])

    for i in insts:
        y_train = change.loc[train_idx, i]
        df_train = pd.concat([y_train, Xf_train, neighbor_hist[i].rename("neighbor")], axis=1).dropna()
        if len(df_train) < 80:
            continue

        X_train = df_train.drop(columns=[i]).values
        y = df_train[i].values

        feature_names = list(Xf_train.columns) + ["neighbor"]

        Xt_df = pd.concat([Xf_t, pd.DataFrame({"neighbor":[float(neighbor_t[i])]}, index=[date])], axis=1)
        Xt_df.columns = feature_names
        Xt = Xt_df.values

        if cfg.regime_mixture and (w_on is not None) and (w_off is not None):
            # align sample weights to df_train index
            idx_train = df_train.index
            wo = pd.Series(w_off, index=train_idx).reindex(idx_train).fillna(0.0).values
            wn = pd.Series(w_on, index=train_idx).reindex(idx_train).fillna(0.0).values

            wn2 = _safe_weights(wn)
            wo2 = _safe_weights(wo)

            if wn2 is None and wo2 is None:
                model_on, sc_on = _fit_ridge(X_train, y, cfg.ridge_alpha, sample_weight=None)
                model_off, sc_off = model_on, sc_on
            elif wn2 is None:
                model_off, sc_off = _fit_ridge(X_train, y, cfg.ridge_alpha, sample_weight=wo2)
                model_on, sc_on = model_off, sc_off
            elif wo2 is None:
                model_on, sc_on = _fit_ridge(X_train, y, cfg.ridge_alpha, sample_weight=wn2)
                model_off, sc_off = model_on, sc_on
            else:
                model_on, sc_on = _fit_ridge(X_train, y, cfg.ridge_alpha, sample_weight=wn2)
                model_off, sc_off = _fit_ridge(X_train, y, cfg.ridge_alpha, sample_weight=wo2)

            
            Xt_on = sc_on.transform(Xt)
            Xt_off = sc_off.transform(Xt)

            yhat_on = float(model_on.predict(Xt_on)[0])
            yhat_off = float(model_off.predict(Xt_off)[0])
            yhat = (1.0 - p_off_t) * yhat_on + p_off_t * yhat_off

            preds[i] = yhat

            # explain: blended contributions in standardized space
            contrib_on = {feature_names[k]: float(model_on.coef_[k] * Xt_on[0, k]) for k in range(len(feature_names))}
            contrib_off = {feature_names[k]: float(model_off.coef_[k] * Xt_off[0, k]) for k in range(len(feature_names))}
            contrib = {k: (1.0 - p_off_t) * contrib_on.get(k, 0.0) + p_off_t * contrib_off.get(k, 0.0) for k in feature_names}
            contrib["intercept"] = float((1.0 - p_off_t) * model_on.intercept_ + p_off_t * model_off.intercept_)
            contrib["pred"] = float(yhat)
            explain[i] = contrib
        else:
            model, sc = _fit_ridge(X_train, y, cfg.ridge_alpha, sample_weight=None)
            Xt_s = sc.transform(Xt)
            yhat = float(model.predict(Xt_s)[0])
            preds[i] = yhat
            contrib = {feature_names[k]: float(model.coef_[k] * Xt_s[0, k]) for k in range(len(feature_names))}
            contrib["intercept"] = float(model.intercept_)
            contrib["pred"] = float(yhat)
            explain[i] = contrib

    return preds, explain