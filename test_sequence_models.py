import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from forecast_data import prepare
from sequence_models import window_indices, sequence_data, RNNForecast, TransformerForecast


class SequenceTests(unittest.TestCase):
    def test_window_ends_at_current_not_future(self):
        times = pd.date_range('2020-01-01', periods=30, freq='h')
        w = window_indices(times)
        np.testing.assert_equal(w[0], np.arange(24))
        self.assertTrue((w[:, -1] - w[:, 0] == 23).all())
        self.assertEqual(w[-1, -1], 29)

    def test_gap_not_bridged(self):
        times = pd.date_range('2020-01-01', periods=60, freq='h').delete(25).as_unit('ns')
        w = window_indices(times)
        self.assertTrue(np.all(np.diff(times.asi8[w], axis=1) == pd.Timedelta(hours=1).value))

    def test_same_validation_and_test_targets(self):
        sets = prepare()
        batches, mean, std = sequence_data(sets)
        for key in ['validation', 'test']:
            pd.testing.assert_series_equal(batches[key][2].target_time, sets[key].target_time.reset_index(drop=True))
        changed = {k: v.copy() for k, v in sets.items()}
        changed['test']['current_count'] = 99999
        altered, m2, s2 = sequence_data(changed)
        np.testing.assert_equal(batches['train'][0], altered['train'][0])
        np.testing.assert_equal(batches['validation'][0], altered['validation'][0])
        self.assertEqual((mean, std), (m2, s2))

    def test_rnn_shape_and_backward(self):
        model = RNNForecast(19)
        output = model(torch.randn(5,24,19))
        self.assertEqual(tuple(output.shape), (5,))
        output.sum().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_transformer_shape_position_and_backward(self):
        model = TransformerForecast(19)
        model.eval()
        x = torch.randn(5,24,19)
        output = model(x)
        self.assertEqual(tuple(output.shape), (5,))
        self.assertFalse(torch.equal(model.position[:, 0], model.position[:, 1]))
        self.assertFalse(torch.allclose(output, model(x.flip(1))))
        output.sum().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
