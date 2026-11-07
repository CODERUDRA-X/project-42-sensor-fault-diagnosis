"""Run a small command-line diagnosis example with: python -m project42"""
from .inference import SensorDiagnosisEngine, SensorReadings

engine = SensorDiagnosisEngine()
readings = SensorReadings(
    vibration_amplitude_g=3.4,
    acoustic_decibels=83.0,
    bearing_temperature_c=88.0,
    motor_amperage=24.5,
)
result = engine.diagnose(readings)
print("Project 42 — illustrative synthetic telemetry diagnosis")
print("Predicted state:", result["predicted_state"])
print("Raw probabilities:")
for name, probability in result["probabilities"].items():
    print(f"  {name:8s}: {probability:.6%}")
print("Standard decision:", result["standard_decision"])
print("Cost-sensitive decision:", result["cost_sensitive_decision"])
print("Expected illustrative risks:", result["expected_risk_illustrative_units"])
if result["range_warnings"]:
    print("Range warnings:")
    for warning in result["range_warnings"]:
        print(" -", warning)
print("\nNOTE:", result["calibration_warning"])
