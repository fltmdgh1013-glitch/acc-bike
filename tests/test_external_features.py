import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from forecast_data import prepare, NUMERIC
from external_features import EXTERNAL_FILE, EXTERNAL_COLUMNS, attach_calendar, enrich_sets
from sequence_models import sequence_data


class ExternalFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sets = prepare()
        cls.calendar = pd.read_csv(EXTERNAL_FILE)

    def test_target_date_not_input_date_at_midnight(self):
        frame = self.sets['train']
        frame = frame.loc[frame.target_time.between('2017-12-24 23:00', '2017-12-25 01:00')]
        new = attach_calendar(frame, self.calendar)
        self.assertEqual(new['특별공휴일여부'].tolist(), [0,1,1])
        pd.testing.assert_frame_equal(new[frame.columns], frame)

    def test_missing_or_duplicate_date_rejected(self):
        with self.assertRaises(ValueError):
            attach_calendar(self.sets['train'], self.calendar.iloc[1:])
        with self.assertRaises(ValueError):
            attach_calendar(self.sets['train'], pd.concat([self.calendar,self.calendar.iloc[:1]]))

    def test_join_covers_every_split_preserving_targets(self):
        new = enrich_sets(self.sets, self.calendar.sample(frac=1, random_state=42))
        for key, original in self.sets.items():
            pd.testing.assert_frame_equal(new[key][original.columns], original)
            self.assertFalse(new[key][EXTERNAL_COLUMNS].isna().any().any())

    def test_future_test_features_do_not_change_training_transform(self):
        sets = enrich_sets(self.sets, self.calendar)
        batches, mean, std = sequence_data(sets, numeric=NUMERIC+EXTERNAL_COLUMNS)
        changed = {k:v.copy() for k,v in sets.items()}
        changed['test']['연속휴일길이_일'] = 999
        after, mean2, std2 = sequence_data(changed, numeric=NUMERIC+EXTERNAL_COLUMNS)
        np.testing.assert_equal(batches['train'][0], after['train'][0])
        np.testing.assert_equal(batches['validation'][0], after['validation'][0])
        self.assertEqual((mean,std), (mean2,std2))
