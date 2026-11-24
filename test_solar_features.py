import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from forecast_data import prepare,NUMERIC
from solar_features import SOLAR_FILE,SOLAR_COLUMNS,attach_solar,enrich_solar_sets
from sequence_models import sequence_data


class SolarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sets=prepare();cls.solar=pd.read_csv(SOLAR_FILE)

    def test_midnight_uses_target_date_and_preserves_row_order(self):
        frame=pd.DataFrame({'input_time':pd.to_datetime(['2018-01-01 23:00','2018-01-01 22:00']),
                            'target_time':pd.to_datetime(['2018-01-02 00:00','2018-01-01 23:00'])},index=[7,3])
        solar=pd.DataFrame({'날짜':['2018-01-01','2018-01-02'],
                            '일출시각':['2018-01-01 07:00','2018-01-02 08:00'],
                            '일몰시각':['2018-01-01 17:00','2018-01-02 18:00']})
        new=attach_solar(frame,solar)
        pd.testing.assert_frame_equal(frame,new[frame.columns])
        self.assertEqual(new['일출_분'].tolist(),[480,420])
        self.assertEqual(new['주간여부'].tolist(),[0,0])

    def test_daylight_includes_sunrise_excludes_sunset(self):
        frame=pd.DataFrame({'target_time':pd.to_datetime(['2018-01-01 06:59','2018-01-01 07:00','2018-01-01 16:59','2018-01-01 17:00'])})
        solar=pd.DataFrame({'날짜':['2018-01-01'],'일출시각':['2018-01-01 07:00'],'일몰시각':['2018-01-01 17:00']})
        new=attach_solar(frame,solar)
        self.assertEqual(new['주간여부'].tolist(),[0,1,1,0])
        self.assertEqual(new[SOLAR_COLUMNS[-1]].tolist(),[1,0,1,0])

    def test_bad_or_missing_solar_records_are_rejected(self):
        for solar in [self.solar.iloc[1:],pd.concat([self.solar,self.solar.iloc[:1]])]:
            with self.assertRaises(ValueError):attach_solar(self.sets['train'],solar)
        solar=self.solar.copy();solar.loc[0,'일출시각']=solar.loc[0,'일몰시각']
        with self.assertRaises(ValueError):attach_solar(self.sets['train'],solar)

    def test_full_coverage_and_unseen_features_do_not_change_training(self):
        enriched=enrich_solar_sets(self.sets,self.solar.sample(frac=1,random_state=42))
        for key,frame in self.sets.items():
            pd.testing.assert_frame_equal(frame,enriched[key][frame.columns])
            self.assertFalse(enriched[key][SOLAR_COLUMNS].isna().any().any())
            self.assertEqual(set(enriched[key])-set(frame),set(SOLAR_COLUMNS))
        batches,mean,std=sequence_data(enriched,numeric=NUMERIC+SOLAR_COLUMNS)
        altered={k:v.copy() for k,v in enriched.items()};altered['test']['일출_분']=99999
        other,mean2,std2=sequence_data(altered,numeric=NUMERIC+SOLAR_COLUMNS)
        np.testing.assert_array_equal(batches['train'][0],other['train'][0])
        np.testing.assert_array_equal(batches['validation'][0],other['validation'][0])
        self.assertEqual((mean,std),(mean2,std2))
