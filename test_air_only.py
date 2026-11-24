import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from forecast_data import prepare,NUMERIC
from air_quality_features import AIR_FILE,AIR_COLUMNS,enrich_air_sets
from external_features import EXTERNAL_COLUMNS
from sequence_models import sequence_data


class AirOnlyTests(unittest.TestCase):
    def test_no_external_calendar_and_same_labels(self):
        original=prepare()
        air,metadata=enrich_air_sets(original,pd.read_csv(AIR_FILE))
        for split in original:
            pd.testing.assert_frame_equal(air[split][original[split].columns],original[split])
            self.assertEqual(set(air[split].columns)-set(original[split].columns),set(AIR_COLUMNS))
            self.assertTrue(set(EXTERNAL_COLUMNS).isdisjoint(air[split].columns))
        batches,mean,std=sequence_data(air,numeric=NUMERIC+AIR_COLUMNS)
        for split in ['validation','test']:
            pd.testing.assert_series_equal(batches[split][2].target_time,original[split].target_time.reset_index(drop=True))
        changed={k:v.copy() for k,v in air.items()}
        changed['test'][AIR_COLUMNS[0]]=99999
        after,mean2,std2=sequence_data(changed,numeric=NUMERIC+AIR_COLUMNS)
        np.testing.assert_equal(batches['train'][0],after['train'][0])
        np.testing.assert_equal(batches['validation'][0],after['validation'][0])
        self.assertEqual((mean,std),(mean2,std2))
