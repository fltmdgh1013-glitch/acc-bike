"""현재 시각의 원본 기온·습도로 계산한 불쾌지수만 연결한다."""
import numpy as np
import pandas as pd
from forecast_data import ROOT

DISCOMFORT_FILE=ROOT/'외부자료_불쾌지수.csv'
DISCOMFORT_COLUMNS=['현재_불쾌지수']
TEMPERATURE='Temperature(°C)'
HUMIDITY='Humidity(%)'


def discomfort_index(temperature,humidity_percent):
    t=np.asarray(temperature,dtype=float);h=np.asarray(humidity_percent,dtype=float)
    if t.shape!=h.shape:raise ValueError('기온과 습도의 배열 크기가 다릅니다.')
    if not (np.isfinite(t).all() and np.isfinite(h).all()):raise ValueError('기온·습도 결측 또는 비수치')
    if ((h<0)|(h>100)).any():raise ValueError('상대습도는 0~100%여야 합니다.')
    return 1.8*t-.55*(1-h/100)*(1.8*t-26)+32


def attach_discomfort(frame,companion):
    required=['입력시각','기온_C','상대습도_pct','현재_불쾌지수']
    if not set(required).issubset(companion.columns):raise ValueError('불쾌지수 자료 필수 열 누락')
    if DISCOMFORT_COLUMNS[0] in frame:raise ValueError('불쾌지수가 이미 연결되어 있습니다.')
    source=companion[required].copy()
    source['입력시각']=pd.to_datetime(source['입력시각'],errors='raise')
    if source['입력시각'].isna().any() or source['입력시각'].duplicated().any():raise ValueError('불쾌지수 시각 결측·중복')
    for col in required[1:]:source[col]=pd.to_numeric(source[col],errors='raise')
    expected=discomfort_index(source['기온_C'],source['상대습도_pct'])
    if not np.isfinite(source['현재_불쾌지수']).all() or not np.allclose(source['현재_불쾌지수'],expected,rtol=0,atol=1e-10):
        raise ValueError('저장된 불쾌지수가 기온·습도 계산식과 다릅니다.')
    result=frame.copy();result['_di_order']=range(len(result))
    result=result.merge(source,left_on='input_time',right_on='입력시각',how='left',validate='many_to_one',sort=False)
    if result[required[1:]].isna().any().any():raise ValueError('현재 입력 시각의 불쾌지수 자료가 없습니다.')
    if not (np.allclose(result[TEMPERATURE],result['기온_C'],rtol=0,atol=1e-10) and
            np.allclose(result[HUMIDITY],result['상대습도_pct'],rtol=0,atol=1e-10)):
        raise ValueError('연결한 기온·습도가 현재 시각 원본과 다릅니다.')
    result=result.sort_values('_di_order').drop(columns=['_di_order','입력시각','기온_C','상대습도_pct'])
    result.index=frame.index
    pd.testing.assert_frame_equal(result[frame.columns],frame)
    return result


def enrich_discomfort_sets(sets,companion):
    return {name:attach_discomfort(frame,companion) for name,frame in sets.items()}
