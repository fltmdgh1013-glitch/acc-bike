import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from forecast_data import prepare,NUMERIC
from solar_features import SOLAR_COLUMNS
from external_features import EXTERNAL_COLUMNS
from air_quality_features import AIR_COLUMNS,AIR_FILE
from run_solar_mix_experiment import build_mix_sets,VARIANTS
from sequence_models import sequence_data


class SolarMixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.original=prepare()

    def test_exact_combinations_preserve_targets_and_order(self):
        for key,spec in VARIANTS.items():
            sets,columns,audit=build_mix_sets(self.original,key)
            expected=SOLAR_COLUMNS+(EXTERNAL_COLUMNS if spec['calendar'] else [])+(AIR_COLUMNS if spec['air'] else [])
            self.assertEqual(columns,expected)
            for split,frame in self.original.items():
                pd.testing.assert_frame_equal(frame,sets[split][frame.columns])
                self.assertEqual(set(sets[split])-set(frame),set(expected))
                self.assertFalse(sets[split][expected].isna().any().any())
            self.assertEqual(bool(audit),spec['air'])

    def test_combined_air_still_uses_lagged_input_time(self):
        sets,_,_=build_mix_sets(self.original,'solar_calendar_air')
        hourly=pd.read_csv(AIR_FILE,parse_dates=['측정종료시각']).set_index('측정종료시각')
        frame=sets['test'].iloc[[0,1,-1]]
        expected=hourly.loc[frame.input_time-pd.Timedelta(hours=1),'미세먼지_PM10'].to_numpy()
        np.testing.assert_allclose(frame[AIR_COLUMNS[0]].to_numpy(),expected)

    def test_test_feature_changes_do_not_change_train_or_validation(self):
        sets,columns,_=build_mix_sets(self.original,'solar_calendar_air')
        before,mean,std=sequence_data(sets,numeric=NUMERIC+columns)
        changed={k:v.copy() for k,v in sets.items()}
        changed['test'][columns]=9999
        after,mean2,std2=sequence_data(changed,numeric=NUMERIC+columns)
        np.testing.assert_array_equal(before['train'][0],after['train'][0])
        np.testing.assert_array_equal(before['validation'][0],after['validation'][0])
        self.assertEqual((mean,std),(mean2,std2))
