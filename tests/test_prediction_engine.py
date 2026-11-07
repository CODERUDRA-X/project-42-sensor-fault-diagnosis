import unittest
from pathlib import Path
import numpy as np
from project42.inference import SensorDiagnosisEngine, SensorReadings, FEATURES

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / 'src'))

class TestSensorDiagnosisEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = SensorDiagnosisEngine(ROOT / 'model' / 'sensor_fault_gnb_pipeline.joblib')

    def test_probabilities_are_valid_and_sum_to_one(self):
        result = self.engine.diagnose(SensorReadings(3.4, 83.0, 88.0, 24.5))
        p = list(result['probabilities'].values())
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(all(0 <= v <= 1 for v in p))
        self.assertAlmostEqual(sum(p), 1.0, places=8)

    def test_decisions_and_risks_are_valid(self):
        result = self.engine.diagnose(SensorReadings(0.7, 66.0, 58.0, 20.0))
        self.assertIn(result['standard_decision'], ('Normal', 'Warning', 'Critical'))
        self.assertIn(result['cost_sensitive_decision'], ('Normal', 'Warning', 'Critical'))
        self.assertTrue(all(v >= 0 for v in result['expected_risk_illustrative_units'].values()))

    def test_exact_feature_keys_required(self):
        with self.assertRaises(ValueError):
            self.engine.diagnose({'vibration_amplitude_g': 1, 'acoustic_decibels': 2})

    def test_negative_and_nonfinite_values_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.diagnose(SensorReadings(-1, 60, 50, 20))
        with self.assertRaises(ValueError):
            self.engine.diagnose(SensorReadings(float('nan'), 60, 50, 20))

    def test_out_of_observed_range_warns_but_does_not_silently_clip(self):
        result = self.engine.diagnose(SensorReadings(99, 66, 58, 20))
        self.assertTrue(result['range_warnings'])
        self.assertEqual(result['readings'][FEATURES[0]], 99)

    def test_uncalibrated_status_is_explicit(self):
        result = self.engine.diagnose(SensorReadings(1, 66, 58, 20))
        self.assertFalse(result['probabilities_calibrated'])
        self.assertIn('uncalibrated', result['calibration_warning'].lower())

if __name__ == '__main__':
    unittest.main()
