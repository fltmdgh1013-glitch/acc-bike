import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from solution import predict, scores, select_model


class ForecastModelTests(unittest.TestCase):
    def test_persistence_and_teacher_error(self):
        pred = predict(None, pd.DataFrame({'current_count': [490]}))
        np.testing.assert_equal(pred, [490])
        self.assertEqual(scores([339], pred)['MAE'], 151)

    def test_validation_mae_selects_model(self):
        rows = [{'model': 'A', 'MAE': 10, 'RMSE': 99},
                {'model': 'B', 'MAE': 11, 'RMSE': 12}]
        self.assertEqual(select_model(rows), 'A')
