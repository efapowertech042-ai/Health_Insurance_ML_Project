"""
Loads the model artifacts exported from Section 9 of the training notebook
(gb_model.pkl, gb_lower.pkl, gb_upper.pkl, metadata.pkl) and exposes a single
`predict_charges(...)` function used by the Django view.

Models are loaded once per process (module-level cache) rather than on every
request.
"""

import os
import joblib
import numpy as np
import pandas as pd
from django.conf import settings

_model = None
_model_lower = None
_model_upper = None
_metadata = None


def _load_artifacts():
    global _model, _model_lower, _model_upper, _metadata

    if _model is not None:
        return  # already loaded

    artifacts_dir = settings.MODEL_ARTIFACTS_DIR

    _model = joblib.load(os.path.join(artifacts_dir, "gb_model.pkl"))
    _model_lower = joblib.load(os.path.join(artifacts_dir, "gb_lower.pkl"))
    _model_upper = joblib.load(os.path.join(artifacts_dir, "gb_upper.pkl"))
    _metadata = joblib.load(os.path.join(artifacts_dir, "metadata.pkl"))


def _build_input_row(age, sex, bmi, children, smoker, region):
    """Recreate the exact preprocessing used in training (Section 6 of the notebook)."""
    sex_map = _metadata["sex_map"]
    smoker_map = _metadata["smoker_map"]
    region_categories = _metadata["region_categories"]  # first entry = dropped base category
    feature_columns = _metadata["feature_columns"]

    row = {
        "age": age,
        "sex": sex_map[sex],
        "bmi": bmi,
        "children": children,
        "smoker": smoker_map[smoker],
    }
    for cat in region_categories[1:]:
        row[f"region_{cat}"] = 1 if region == cat else 0

    df_row = pd.DataFrame([row])
    df_row = df_row.reindex(columns=feature_columns, fill_value=0)
    return df_row


def _extrapolation_warning(age, bmi):
    """Flag inputs outside the training data's observed range separately from
    the interval-based confidence flag -- these are two different risk signals."""
    warnings = []
    if age < 18 or age > 64:
        warnings.append("Age is outside the training data range (18-64) — treat as extrapolation.")
    if bmi < 15 or bmi > 53:
        warnings.append("BMI is outside the training data range (~15-53) — treat as extrapolation.")
    return warnings


def predict_charges(age, sex, bmi, children, smoker, region):
    """
    Returns a dict with the point prediction, 90% interval bounds, and
    confidence flagging -- mirrors the logic built in Section 9 of the notebook.
    """
    _load_artifacts()

    X_input = _build_input_row(age, sex, bmi, children, smoker, region)

    pred = float(_model.predict(X_input)[0])
    pred_lower = float(_model_lower.predict(X_input)[0])
    pred_upper = float(_model_upper.predict(X_input)[0])

    # The models were trained on log1p(charges), not raw charges (see Section
    # 7.2 / 9.1 of the notebook) -- every .predict() call above returns a
    # log-scale value. Invert all three individually, before any of the math
    # below runs, so interval width and flagging operate on real dollars.
    # Branches on the metadata flag rather than assuming log1p, so this still
    # works unchanged if the model is ever retrained on the raw target.
    if _metadata.get("target_transform") == "log1p":
        pred = np.expm1(pred)
        pred_lower = np.expm1(pred_lower)
        pred_upper = np.expm1(pred_upper)

    # gb_lower and gb_upper are independently trained quantile models with no
    # constraint tying them to each other or to gb's point prediction, so
    # "quantile crossing" (lower > upper, or the point estimate landing
    # outside its own interval) can happen on individual inputs even when
    # the models are well-calibrated overall. Enforce both explicitly here,
    # mirroring the same safeguard added in Section 9.1 of the notebook.
    pred_lower, pred_upper = min(pred_lower, pred_upper), max(pred_lower, pred_upper)
    pred_lower = min(pred_lower, pred)
    pred_upper = max(pred_upper, pred)

    interval_width = pred_upper - pred_lower

    flag_threshold = _metadata["flag_threshold"]
    is_low_confidence = interval_width > flag_threshold

    return {
        "prediction": pred,
        "interval_low": max(pred_lower, 0),
        "interval_high": pred_upper,
        "interval_width": interval_width,
        "flag_threshold": flag_threshold,
        "is_low_confidence": is_low_confidence,
        "extrapolation_warnings": _extrapolation_warning(age, bmi),
        "model_r2": _metadata["test_r2"],
        "model_rmse": _metadata["test_rmse"],
        "model_mae": _metadata["test_mae"],
    }
