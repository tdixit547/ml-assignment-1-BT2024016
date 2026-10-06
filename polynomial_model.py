"""Fit and serialize polynomial regressors without pickled estimator objects."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import ARDRegression, LassoLars, LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def design_matrix(x, powers, basis="monomial"):
    """Products of univariate polynomials, with total degree bounded by powers."""
    x = np.asarray(x, dtype=float)
    powers = np.asarray(powers, dtype=int)
    if not len(powers):
        return np.empty((len(x), 0))
    degree = int(powers.max())
    out = np.ones((len(x), len(powers)))
    for j in range(x.shape[1]):
        if basis == "legendre":
            vander = np.polynomial.legendre.legvander(x[:, j], degree)
        elif basis == "monomial":
            vander = np.polynomial.polynomial.polyvander(x[:, j], degree)
        else:
            raise ValueError(f"Unknown polynomial basis: {basis}")
        out *= vander[:, powers[:, j]]
    return out


def predict_model(model, x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(model["features"]):
        raise ValueError("Input feature count does not match the fitted model.")
    if "components" in model:
        predictions = [predict_model(component, x) for component in model["components"]]
        return np.average(predictions, axis=0, weights=model["weights"])
    shifted = (x - np.asarray(model["input_mean"])) / np.asarray(model["input_scale"])
    z = design_matrix(shifted, model["powers"], model["basis"])
    result = z @ np.asarray(model["coefficients"]) + model["intercept"]
    if not np.isfinite(result).all():
        raise ValueError("Non-finite predictions.")
    return result


def fit_model(x, y, features, spec):
    """Fit preprocessing and regression using only the provided training rows."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if "components" in spec:
        models = [fit_model(x, y, features, part) for part in spec["components"]]
        return {"schema_version": 1, "features": list(features), "degree": max(m["degree"] for m in models),
                "spec": spec, "components": models, "weights": spec.get("weights", [1.] * len(models)),
                "fit_rows": len(x)}
    input_mean = np.zeros(x.shape[1]); input_scale = np.ones(x.shape[1])
    if spec.get("input_standardize", False):
        ss = StandardScaler().fit(x)
        input_mean, input_scale = ss.mean_, ss.scale_
    shifted = (x-input_mean) / input_scale
    powers = PolynomialFeatures(spec["degree"], include_bias=False).fit(shifted).powers_
    basis = spec.get("basis", "monomial")
    z = design_matrix(shifted, powers, basis)
    scaler = StandardScaler().fit(z)
    standardized = scaler.transform(z)
    weights = powers.sum(axis=1).astype(float) ** (-spec.get("penalty", 0.))
    method = spec.get("method", "ridge")
    alpha = spec.get("alpha", 0.)
    if method == "ols":
        estimator = LinearRegression()
    elif method == "ridge":
        estimator = Ridge(alpha=alpha, solver="svd")
    elif method == "lasso":
        estimator = LassoLars(alpha=alpha, max_iter=2000)
    elif method == "ard":
        estimator = ARDRegression(max_iter=500, tol=1e-4)
    else:
        raise ValueError(f"Unknown fitting method: {method}")
    estimator.fit(standardized*weights, y)
    coef = estimator.coef_ * weights
    intercept = float(estimator.intercept_)
    relax = spec.get("relax", 0.)
    if relax:
        selected = np.abs(coef) > 1e-9
        if np.any(selected):
            refit = Ridge(alpha=.01, solver="svd").fit(standardized[:, selected], y)
            adjusted = np.zeros_like(coef)
            adjusted[selected] = refit.coef_
            coef = (1-relax)*coef + relax*adjusted
            intercept = (1-relax)*intercept + relax*float(refit.intercept_)
    raw_coef = coef / scaler.scale_
    raw_intercept = intercept - float(raw_coef @ scaler.mean_)
    active = raw_coef != 0
    model = {"schema_version": 1, "features": list(features), "degree": int(spec["degree"]),
             "basis": basis, "spec": dict(spec), "input_mean": input_mean.tolist(),
             "input_scale": input_scale.tolist(), "powers": powers[active].tolist(),
             "coefficients": raw_coef[active].tolist(), "intercept": raw_intercept,
             "fit_rows": len(x), "candidate_terms": len(powers), "active_terms": int(active.sum())}
    expected = standardized@coef+intercept
    np.testing.assert_allclose(predict_model(model,x), expected, rtol=1e-8, atol=1e-8)
    return model


def save_model(model, path):
    Path(path).write_text(json.dumps(model,indent=2,allow_nan=False),encoding="utf-8")


def load_model(path):
    model=json.loads(Path(path).read_text(encoding="utf-8"))
    if model.get("schema_version") != 1:
        raise ValueError("Unsupported model file version.")
    return model
