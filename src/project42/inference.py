"""Reusable inference engine for Project 42.

Uses the exact saved StandardScaler -> GaussianNB artifact from the supplied
Project42_Code_and_Dataset.zip. The model is UNCALIBRATED. Cost units and
sensor ranges are research/demo assumptions, not industrial operating limits.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import math

import joblib
import numpy as np

CLASSES = ("Normal", "Warning", "Critical")
FEATURES = (
    "vibration_amplitude_g",
    "acoustic_decibels",
    "bearing_temperature_c",
    "motor_amperage",
)
FEATURE_LABELS = (
    "Vibration amplitude",
    "Acoustic level",
    "Bearing temperature",
    "Motor amperage",
)
UNITS = ("g", "dB", "°C", "A")

# Exact notebook matrix: rows = actual class, columns = chosen/predicted class.
# These are illustrative cost units, not measured currency or plant losses.
COST_MATRIX = np.array(
    [[0.0, 1.0, 2.0],
     [10.0, 0.0, 3.0],
     [100.0, 30.0, 0.0]],
    dtype=float,
)

# Observed min/max in the supplied 6,000-row synthetic dataset.
# These are data bounds, NOT validated safe operating limits.
OBSERVED_RANGES = {
    FEATURES[0]: (0.05, 5.302),
    FEATURES[1]: (57.54, 88.49),
    FEATURES[2]: (41.25, 103.62),
    FEATURES[3]: (15.42, 33.42),
}


@dataclass(frozen=True)
class SensorReadings:
    vibration_amplitude_g: float
    acoustic_decibels: float
    bearing_temperature_c: float
    motor_amperage: float

    def to_array(self) -> np.ndarray:
        values = np.array([
            self.vibration_amplitude_g,
            self.acoustic_decibels,
            self.bearing_temperature_c,
            self.motor_amperage,
        ], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("All four sensor readings must be finite numbers.")
        if (values < 0).any():
            raise ValueError("Sensor readings cannot be negative.")
        return values.reshape(1, -1)


class SensorDiagnosisEngine:
    """Loads once, validates inputs, predicts, and explains a single machine."""

    def __init__(self, model_path: str | Path | None = None):
        if model_path is None:
            model_path = Path(__file__).resolve().parents[2] / "model" / "sensor_fault_gnb_pipeline.joblib"
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")

        self.model = joblib.load(self.model_path)
        if not hasattr(self.model, "named_steps") or not {"scaler", "gnb"}.issubset(self.model.named_steps):
            raise TypeError("Expected the saved sklearn Pipeline with 'scaler' and 'gnb' steps.")
        gnb = self.model.named_steps["gnb"]
        if tuple(int(c) for c in gnb.classes_) != (0, 1, 2):
            raise ValueError(f"Unexpected model classes: {gnb.classes_!r}; expected [0, 1, 2].")
        if int(self.model.n_features_in_) != len(FEATURES):
            raise ValueError(f"Model expects {self.model.n_features_in_} features; expected {len(FEATURES)}.")

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model": "Gaussian Naive Bayes",
            "pipeline": ["StandardScaler", "GaussianNB"],
            "classes": list(CLASSES),
            "features_in_exact_order": list(FEATURES),
            "probabilities_calibrated": False,
            "dataset": "6,000-row physics-inspired synthetic telemetry dataset",
            "cost_matrix_note": "Illustrative notebook cost units; not real industrial costs.",
        }

    def diagnose(self, readings: SensorReadings | dict[str, float]) -> dict[str, Any]:
        if isinstance(readings, dict):
            missing = [f for f in FEATURES if f not in readings]
            extra = [k for k in readings if k not in FEATURES]
            if missing or extra:
                raise ValueError(f"Input keys must be exactly {list(FEATURES)}. Missing={missing}; extra={extra}.")
            readings = SensorReadings(**{f: readings[f] for f in FEATURES})
        if not isinstance(readings, SensorReadings):
            raise TypeError("readings must be SensorReadings or a dict with the four exact feature names.")

        x = readings.to_array()
        probabilities = self.model.predict_proba(x)[0]
        probabilities = np.asarray(probabilities, dtype=float)
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ArithmeticError("Model returned invalid probabilities.")
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-8):
            raise ArithmeticError(f"Model probabilities do not sum to 1 (sum={probabilities.sum()}).")

        argmax_idx = int(np.argmax(probabilities))
        expected_risks = probabilities @ COST_MATRIX
        risk_idx = int(np.argmin(expected_risks))

        # Explain Critical vs Normal evidence using the exact transformed features
        # and GaussianNB parameters inside the saved pipeline.
        scaler = self.model.named_steps["scaler"]
        gnb = self.model.named_steps["gnb"]
        xt = scaler.transform(x)[0]
        means = np.asarray(gnb.theta_, dtype=float)
        variances = np.asarray(gnb.var_, dtype=float)
        priors = np.asarray(gnb.class_prior_, dtype=float)
        log_likelihoods = -0.5 * np.log(2.0 * np.pi * variances) - 0.5 * ((xt[None, :] - means) ** 2) / variances
        # index 2 = Critical, index 0 = Normal, matching the notebook's class order.
        evidence = log_likelihoods[2] - log_likelihoods[0]
        prior_log_odds = float(np.log(priors[2] / priors[0]))
        evidence_items = [
            {"sensor": FEATURE_LABELS[i], "feature": FEATURES[i], "unit": UNITS[i],
             "critical_vs_normal_log_likelihood_ratio": float(evidence[i]),
             "direction": "supports Critical" if evidence[i] > 0 else ("supports Normal" if evidence[i] < 0 else "neutral")}
            for i in range(len(FEATURES))
        ]

        input_values = x[0]
        range_warnings = []
        for i, feature in enumerate(FEATURES):
            lo, hi = OBSERVED_RANGES[feature]
            if input_values[i] < lo or input_values[i] > hi:
                range_warnings.append(
                    f"{FEATURE_LABELS[i]} ({input_values[i]:g} {UNITS[i]}) is outside the observed synthetic dataset range [{lo:g}, {hi:g}]. Prediction is extrapolative."
                )

        return {
            "predicted_state": CLASSES[argmax_idx],
            "predicted_class_code": argmax_idx,
            "probabilities": {CLASSES[i]: float(probabilities[i]) for i in range(3)},
            "standard_decision": CLASSES[argmax_idx],
            "cost_sensitive_decision": CLASSES[risk_idx],
            "expected_risk_illustrative_units": {CLASSES[i]: float(expected_risks[i]) for i in range(3)},
            "cost_matrix": COST_MATRIX.tolist(),
            "cost_matrix_orientation": "rows=actual class; columns=chosen action",
            "probabilities_calibrated": False,
            "calibration_warning": "These are raw GaussianNB probabilities from the saved, uncalibrated pipeline; do not interpret them as calibrated real-world confidence.",
            "critical_vs_normal_evidence": {
                "prior_log_odds_critical_vs_normal": prior_log_odds,
                "sensor_contributions": evidence_items,
                "sum_log_odds": float(prior_log_odds + evidence.sum()),
                "interpretation": "Positive sensor contribution supports Critical relative to Normal; negative supports Normal. This is model evidence, not causal proof.",
            },
            "readings": {FEATURES[i]: float(input_values[i]) for i in range(4)},
            "range_warnings": range_warnings,
            "limitations": [
                "The model was trained on synthetic data, not measurements from a deployed industrial machine.",
                "Observed dataset ranges are not engineering safety limits.",
                "The cost matrix is illustrative and must be replaced with plant-approved costs before operational use.",
                "This research prototype is not a certified safety or shutdown controller.",
            ],
        }


def diagnose_machine(vibration_amplitude_g: float, acoustic_decibels: float,
                     bearing_temperature_c: float, motor_amperage: float,
                     model_path: str | Path | None = None) -> dict[str, Any]:
    """Convenience wrapper retaining the notebook's four-argument calling style."""
    engine = SensorDiagnosisEngine(model_path=model_path)
    return engine.diagnose(SensorReadings(
        vibration_amplitude_g, acoustic_decibels,
        bearing_temperature_c, motor_amperage,
    ))
