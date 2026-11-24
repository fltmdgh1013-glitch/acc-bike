"""Check time alignment and guard against leaks across missing hours."""
import sys
import unittest
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from forecast_data import RAW, build_examples, split_examples, FEATURES


class ForecastDataTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.read_csv(RAW, encoding='latin1')

    def test_teacher_example(self):
        examples, _ = build_examples(self.raw)
        row = examples.loc[examples.input_time == pd.Timestamp('2017-12-01 09:00')].iloc[0]
        self.assertEqual(row.current_count, 490)
        self.assertEqual(row.target_count, 339)
        self.assertEqual(row['Temperature(°C)'], -6.5)
        self.assertNotIn('target_count', FEATURES)
        self.assertTrue(((examples.target_time - examples.input_time) == pd.Timedelta(hours=1)).all())

    def test_missing_hour_is_not_next_hour(self):
        examples, audit = build_examples(self.raw.drop(index=10))
        self.assertFalse((examples.input_time == pd.Timestamp('2017-12-01 09:00')).any())
        self.assertEqual(audit['gap_pairs_excluded'], 1)

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            build_examples(pd.concat([self.raw, self.raw.iloc[:1]]))

    def test_future_weather_cannot_change_current_features(self):
        before, _ = build_examples(self.raw)
        changed = self.raw.copy()
        changed.loc[10, 'Temperature(°C)'] = 99
        after, _ = build_examples(changed)
        pd.testing.assert_series_equal(before.iloc[9][FEATURES], after.iloc[9][FEATURES])

    def test_split_order(self):
        examples, _ = build_examples(self.raw)
        sets = split_examples(examples)
        self.assertEqual(sum(map(len, sets.values())), len(examples))
        self.assertLess(sets['train'].target_time.max(), sets['validation'].target_time.min())
        self.assertLess(sets['validation'].target_time.max(), sets['test'].target_time.min())


if __name__ == '__main__':
    unittest.main()
