import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rain_forecast_features import attach_rain_forecast as attach_city,aggregate_city_forecasts,RAIN_FORECAST_COLUMNS

# 시간 연결 검사용 가상 두 격자. 실제 서울 전체 격자 목록이 아니다.
TEST_GRIDS=pd.DataFrame({'격자X':[60,61],'격자Y':[127,127]})


def attach_rain_forecast(frame,forecasts,**kwargs):
    expanded=pd.concat([forecasts.assign(격자X=x,격자Y=y) for x,y in TEST_GRIDS.itertuples(index=False,name=None)],ignore_index=True)
    return attach_city(frame,expanded,expected_grids=TEST_GRIDS,**kwargs)


class RainForecastTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'input_time':pd.to_datetime(['2018-01-01 09:00','2018-01-01 10:00']),
                             'target_time':pd.to_datetime(['2018-01-01 10:00','2018-01-01 11:00']),
                             'target_count':[500,600]},index=[8,2])

    def forecasts(self):
        return pd.DataFrame({'발표시각_KST':['2018-01-01 08:30','2018-01-01 09:30','2018-01-01 09:30'],
                             '예보대상시각_KST':['2018-01-01 10:00','2018-01-01 10:00','2018-01-01 11:00'],
                             '강수형태코드':[1,0,3]})

    def test_future_issue_is_not_used_and_order_is_preserved(self):
        frame=self.frame().iloc[::-1]
        new=attach_rain_forecast(frame,self.forecasts().iloc[::-1])
        pd.testing.assert_frame_equal(new[frame.columns],frame)
        self.assertEqual(new[RAIN_FORECAST_COLUMNS[0]].tolist(),[1,1])
        self.assertEqual(new[RAIN_FORECAST_COLUMNS[1]].tolist(),[1,0])
        self.assertEqual(new.loc[8,'사용예보_발표시각'],pd.Timestamp('2018-01-01 08:30'))
        self.assertTrue((new['사용예보_가용시각']<=new.input_time).all())

    def test_unavailable_forecast_fails_instead_of_filling_zero(self):
        with self.assertRaises(ValueError):attach_rain_forecast(self.frame(),self.forecasts().iloc[1:])
        with self.assertRaises(ValueError):attach_rain_forecast(self.frame(),self.forecasts(),publication_lag_minutes=31)

    def test_utc_conversion_and_midnight_target(self):
        frame=pd.DataFrame({'input_time':pd.to_datetime(['2017-12-31 23:00']),
                            'target_time':pd.to_datetime(['2018-01-01 00:00'])})
        fc=pd.DataFrame({'발표시각_KST':['2017-12-31 13:30:00+00:00'],
                         '예보대상시각_KST':['2017-12-31 15:00:00+00:00'],'강수형태코드':[0]})
        new=attach_rain_forecast(frame,fc)
        self.assertEqual(new['사용예보_발표시각'].iloc[0],pd.Timestamp('2017-12-31 22:30'))
        self.assertEqual(new[RAIN_FORECAST_COLUMNS[0]].iloc[0],0)

    def test_bad_codes_conflicting_records_and_actuals_rejected(self):
        fc=self.forecasts()
        bad=fc.copy();bad.loc[0,'강수형태코드']=-999
        with self.assertRaises(ValueError):attach_rain_forecast(self.frame(),bad)
        bad=pd.concat([fc,fc.iloc[:1].assign(강수형태코드=0)])
        with self.assertRaises(ValueError):attach_rain_forecast(self.frame(),bad)
        bad=fc.copy();bad.loc[0,'발표시각_KST']=bad.loc[0,'예보대상시각_KST']
        with self.assertRaises(ValueError):attach_rain_forecast(self.frame(),bad)

    def test_later_revision_cannot_change_an_earlier_decision(self):
        frame=self.frame();fc=self.forecasts()
        before=attach_rain_forecast(frame,fc)
        fc.loc[1,'강수형태코드']=2
        after=attach_rain_forecast(frame,fc)
        pd.testing.assert_frame_equal(before,after)

    def spatial_rows(self):
        return pd.DataFrame({'발표시각_KST':['2018-01-01 08:30']*2,
                             '예보대상시각_KST':['2018-01-01 10:00']*2,
                             '격자X':[60,61],'격자Y':[127,127],'강수형태코드':[0,3]})

    def test_shared_admin_grid_is_not_double_weighted(self):
        rows=self.spatial_rows()
        rows=pd.concat([rows,rows.iloc[1:]],ignore_index=True)
        result=aggregate_city_forecasts(rows,TEST_GRIDS)
        self.assertEqual(result['서울_강수예보격자비율'].iloc[0],.5)
        self.assertEqual(result['서울_눈예보격자비율'].iloc[0],.5)
        self.assertEqual(result['예보_집계격자수'].iloc[0],2)

    def test_single_or_missing_grid_cannot_claim_city_coverage(self):
        rows=self.spatial_rows()
        with self.assertRaises(ValueError):aggregate_city_forecasts(rows.iloc[:1],TEST_GRIDS.iloc[:1])
        with self.assertRaises(ValueError):aggregate_city_forecasts(rows.iloc[:1],TEST_GRIDS)
        with self.assertRaises(ValueError):aggregate_city_forecasts(rows.assign(격자X=[60,99]),TEST_GRIDS)

    def test_incomplete_latest_snapshot_falls_back_to_complete_older_one(self):
        rows=self.spatial_rows()
        late=rows.iloc[:1].assign(발표시각_KST='2018-01-01 09:00',강수형태코드=1)
        data=pd.concat([rows,late],ignore_index=True)
        frame=pd.DataFrame({'input_time':pd.to_datetime(['2018-01-01 09:30']),
                            'target_time':pd.to_datetime(['2018-01-01 10:00'])})
        result=attach_city(frame,data,expected_grids=TEST_GRIDS)
        self.assertEqual(result['사용예보_발표시각'].iloc[0],pd.Timestamp('2018-01-01 08:30'))
        self.assertEqual(result['서울_강수예보격자비율'].iloc[0],.5)

    def test_conflicting_same_grid_cannot_be_silently_averaged(self):
        rows=self.spatial_rows()
        bad=pd.concat([rows,rows.iloc[:1].assign(강수형태코드=1)],ignore_index=True)
        with self.assertRaises(ValueError):aggregate_city_forecasts(bad,TEST_GRIDS)
