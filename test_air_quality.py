import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from prepare_air_quality import observation_end, aggregate
from air_quality_features import attach_air, enrich_air_sets, VALUES


class AirQualityTests(unittest.TestCase):
    def test_24_is_next_midnight(self):
        self.assertEqual(str(observation_end(2017123124)), '2018-01-01 00:00:00')
        self.assertEqual(str(observation_end(2017120101)), '2017-12-01 01:00:00')
        with self.assertRaises(ValueError): observation_end(2017120100)

    def test_invalid_measurements_excluded_not_zero(self):
        stations=pd.DataFrame({'측정종료시각':['2020-01-01']*3,'측정소코드':['a','b','c'],
                               'PM10':[10,20,-999],'PM25':[None,40,-999]})
        result=aggregate(stations).iloc[0]
        self.assertEqual(result['미세먼지_PM10'],15)
        self.assertEqual(result['초미세먼지_PM25'],40)
        self.assertEqual(result['PM10_유효측정소수'],2)

    def test_use_past_measurement_not_current_or_future(self):
        frame=pd.DataFrame({'input_time':pd.to_datetime(['2020-01-01 09:00'])})
        hourly=pd.DataFrame({'측정종료시각':pd.to_datetime(['2020-01-01 08:00','2020-01-01 09:00','2020-01-01 10:00']),
                             '미세먼지_PM10':[10,90,999],'초미세먼지_PM25':[5,45,999]})
        result=attach_air(frame,hourly)
        self.assertEqual(result[VALUES[0]].iloc[0],10)
        self.assertEqual(result[VALUES[1]].iloc[0],5)
        with self.assertRaises(ValueError): attach_air(frame,pd.concat([hourly,hourly.iloc[:1]]))

    def test_imputation_uses_train_only_no_backfill(self):
        sets={k:pd.DataFrame({'input_time':pd.to_datetime([stamp])}) for k,stamp in
              [('train','2020-01-01 09:00'),('validation','2020-01-01 10:00'),('test','2020-01-01 11:00')]}
        hourly=pd.DataFrame({'측정종료시각':pd.to_datetime(['2020-01-01 08:00','2020-01-01 10:00']),
                             '미세먼지_PM10':[10,999],'초미세먼지_PM25':[5,999]})
        result,meta=enrich_air_sets(sets,hourly)
        self.assertEqual(result['validation'][VALUES[0]].iloc[0],10)
        self.assertEqual(result['validation']['PM10_결측여부'].iloc[0],1)
        self.assertEqual(meta['median_from_train_only'][VALUES[0]],10)
