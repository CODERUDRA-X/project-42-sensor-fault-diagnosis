# Prediction engine

The reusable inference layer lives in `src/project42/inference.py`. It is deliberately independent of Streamlit so that command-line scripts, tests, and a future web UI all call the same prediction implementation.

## Model contract

- Artifact: `model/sensor_fault_gnb_pipeline.joblib`
- Pipeline: `StandardScaler` followed by `GaussianNB`
- Exact feature order: `vibration_amplitude_g`, `acoustic_decibels`, `bearing_temperature_c`, `motor_amperage`
- Class codes: `0 = Normal`, `1 = Warning`, `2 = Critical`
- Probability status: **uncalibrated raw GaussianNB probabilities**

## Run

From the repository root:

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python -m project42
```

## Use from Python

```python
from project42 import SensorDiagnosisEngine, SensorReadings

engine = SensorDiagnosisEngine()  # loads the repository's model artifact
result = engine.diagnose(SensorReadings(
    vibration_amplitude_g=3.4,
    acoustic_decibels=83.0,
    bearing_temperature_c=88.0,
    motor_amperage=24.5,
))
print(result["predicted_state"])
print(result["probabilities"])
```

## Decision outputs

- `standard_decision`: maximum-probability class.
- `cost_sensitive_decision`: action minimizing expected risk under the notebook's illustrative cost matrix.
- `expected_risk_illustrative_units`: expected cost for each possible action.
- `critical_vs_normal_evidence`: per-feature log-likelihood-ratio contributions comparing Critical against Normal. These are model evidence contributions, not causal explanations.

## Important limitations

The model was trained on synthetic data. Its probabilities are not calibrated. The cost matrix contains illustrative units, not measured industrial losses. Observed data ranges are not engineering safety limits. This prototype must not be used as an industrial safety controller or as the sole basis for maintenance decisions.
