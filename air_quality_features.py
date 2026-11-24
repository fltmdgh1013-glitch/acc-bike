"""과거에 끝난 측정값만 연결하고 결측 대체 기준은 학습 자료로만 계산한다."""
import pandas as pd
from forecast_data import ROOT, RAW

AIR_FILE = ROOT / 'data/external/air_quality/서울_시간별_미세먼지.csv'
AIR_COLUMNS = ['미세먼지_PM10_1시간지연','초미세먼지_PM25_1시간지연',
               'PM10_결측여부','PM25_결측여부']
VALUES = AIR_COLUMNS[:2]


def attach_air(frame, hourly):
    hourly = hourly[['측정종료시각','미세먼지_PM10','초미세먼지_PM25']].copy()
    hourly['측정종료시각'] = pd.to_datetime(hourly['측정종료시각'])
    if hourly['측정종료시각'].duplicated().any():
        raise ValueError('미세먼지 시간별 자료에 중복이 있습니다.')
    for col in ['미세먼지_PM10','초미세먼지_PM25']:
        hourly[col] = pd.to_numeric(hourly[col], errors='coerce')
        hourly.loc[hourly[col] < 0,col] = float('nan')
    result = frame.copy()
    result['_air_time'] = result.input_time-pd.Timedelta(hours=1)
    result['_air_order'] = range(len(result))
    result = result.merge(hourly,left_on='_air_time',right_on='측정종료시각',how='left',validate='many_to_one',sort=False)
    result = result.sort_values('_air_order')
    result['PM10_결측여부'] = result['미세먼지_PM10'].isna().astype(int)
    result['PM25_결측여부'] = result['초미세먼지_PM25'].isna().astype(int)
    result = result.rename(columns={'미세먼지_PM10':VALUES[0],'초미세먼지_PM25':VALUES[1]})
    result = result.drop(columns=['_air_time','_air_order','측정종료시각'])
    result.index = frame.index
    return result


def enrich_air_sets(sets, hourly):
    joined = {k:attach_air(v,hourly) for k,v in sets.items()}
    medians = joined['train'][VALUES].median()
    if medians.isna().any():
        raise ValueError('학습 기간에 유효한 측정값이 없습니다.')
    metadata = {'median_from_train_only':{k:float(v) for k,v in medians.items()},
                'missing_rows_by_split':{k:{c:int(v[c].isna().sum()) for c in VALUES} for k,v in joined.items()},
                'lag_hours':1,'features':AIR_COLUMNS}
    for frame in joined.values():
        frame[VALUES] = frame[VALUES].fillna(medians)
    return joined,metadata


def export_hourly_companion(hourly):
    raw=pd.read_csv(RAW,encoding='latin1')
    times=pd.DataFrame({'input_time':pd.to_datetime(raw.Date,format='%d/%m/%Y')+pd.to_timedelta(raw.Hour,unit='h')})
    external=attach_air(times,hourly)
    external.insert(1,'측정자료종료시각',external.input_time-pd.Timedelta(hours=1))
    external.rename(columns={'input_time':'입력시각'}).to_csv(ROOT/'외부자료_미세먼지.csv',index=False,encoding='utf-8-sig')
    return external
