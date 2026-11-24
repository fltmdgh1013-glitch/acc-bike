import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from forecast_data import prepare,NUMERIC
from discomfort_features import DISCOMFORT_FILE,DISCOMFORT_COLUMNS,discomfort_index,attach_discomfort,enrich_discomfort_sets
from sequence_models import sequence_data


class DiscomfortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sets=prepare();cls.companion=pd.read_csv(DISCOMFORT_FILE)

    def test_percent_humidity_matches_hand_calculated_cases(self):
        np.testing.assert_allclose(discomfort_index([30,30,30],[70,100,0]),[81.38,86,70.6],rtol=0,atol=1e-12)

    def test_nonfinite_or_out_of_range_inputs_fail(self):
        for t,h in [([np.nan],[50]),([25],[np.inf]),([25],[-1]),([25],[101]),([25,26],[50])]:
            with self.assertRaises(ValueError):discomfort_index(t,h)

    def test_every_split_preserves_order_labels_and_only_one_extra_column(self):
        enriched=enrich_discomfort_sets(self.sets,self.companion.sample(frac=1,random_state=42))
        for split,frame in self.sets.items():
            pd.testing.assert_frame_equal(frame,enriched[split][frame.columns])
            self.assertEqual(set(enriched[split])-set(frame),set(DISCOMFORT_COLUMNS))
            self.assertFalse(enriched[split][DISCOMFORT_COLUMNS].isna().any().any())

    def test_duplicate_missing_stale_or_tampered_values_fail(self):
        frame=self.sets['train']
        bads=[self.companion.iloc[1:],pd.concat([self.companion,self.companion.iloc[:1]])]
        wrong=self.companion.copy();wrong.loc[0,'현재_불쾌지수']+=1;bads.append(wrong)
        stale=self.companion.copy();stale.loc[0,'기온_C']+=1
        stale.loc[0,'현재_불쾌지수']=discomfort_index(stale.loc[[0],'기온_C'],stale.loc[[0],'상대습도_pct'])[0]
        bads.append(stale)
        for bad in bads:
            with self.assertRaises(ValueError):attach_discomfort(frame,bad)

    def test_join_uses_current_weather_at_midnight_not_target_weather(self):
        frame=self.sets['train'];frame=frame[frame.input_time.isin(pd.to_datetime(['2017-12-01 23:00','2017-12-02 00:00']))]
        frame=frame.iloc[::-1]
        result=attach_discomfort(frame,self.companion)
        np.testing.assert_allclose(result['현재_불쾌지수'],discomfort_index(frame['Temperature(°C)'],frame['Humidity(%)']))
        pd.testing.assert_frame_equal(frame,result[frame.columns])

    def test_future_values_do_not_change_training_scaling(self):
        enriched=enrich_discomfort_sets(self.sets,self.companion)
        before,mean,std=sequence_data(enriched,numeric=NUMERIC+DISCOMFORT_COLUMNS)
        changed={k:v.copy() for k,v in enriched.items()};changed['test']['현재_불쾌지수']=99999
        after,mean2,std2=sequence_data(changed,numeric=NUMERIC+DISCOMFORT_COLUMNS)
        np.testing.assert_array_equal(before['train'][0],after['train'][0])
        np.testing.assert_array_equal(before['validation'][0],after['validation'][0])
        self.assertEqual((mean,std),(mean2,std2))
