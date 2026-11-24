"""미리 계산 가능한 천문 시각만 예측 대상 KST 날짜·시각에 연결한다."""
import numpy as np
import pandas as pd
from forecast_data import ROOT

SOLAR_FILE = ROOT/'외부자료_일출일몰.csv'
SOLAR_COLUMNS = ['일출_분','일몰_분','주간여부','일출일몰_가까운경계까지_분']


def attach_solar(frame, solar):
    solar=solar[['날짜','일출시각','일몰시각']].copy()
    for c in solar:
        solar[c]=pd.to_datetime(solar[c],errors='raise')
    solar['날짜']=solar['날짜'].dt.normalize()
    if solar['날짜'].duplicated().any() or solar.isna().any().any():
        raise ValueError('일출·일몰 자료의 날짜 중복 또는 결측')
    if not ((solar['일출시각']<solar['일몰시각'])&
            (solar['일출시각'].dt.normalize()==solar['날짜'])&
            (solar['일몰시각'].dt.normalize()==solar['날짜'])).all():
        raise ValueError('일출·일몰 날짜 또는 시각 순서 오류')
    result=frame.copy();result['_solar_date']=result.target_time.dt.normalize();result['_solar_order']=range(len(result))
    result=result.merge(solar,left_on='_solar_date',right_on='날짜',how='left',validate='many_to_one',sort=False)
    if result[['일출시각','일몰시각']].isna().any().any():
        raise ValueError('예측 대상 날짜의 일출·일몰 자료 누락')
    result['일출_분']=(result['일출시각']-result['날짜']).dt.total_seconds()/60
    result['일몰_분']=(result['일몰시각']-result['날짜']).dt.total_seconds()/60
    result['주간여부']=((result.target_time>=result['일출시각'])&(result.target_time<result['일몰시각'])).astype(int)
    result['일출일몰_가까운경계까지_분']=np.minimum(
        (result.target_time-result['일출시각']).dt.total_seconds().abs(),
        (result.target_time-result['일몰시각']).dt.total_seconds().abs())/60
    result=result.sort_values('_solar_order').drop(columns=['_solar_date','_solar_order','날짜','일출시각','일몰시각'])
    result.index=frame.index
    return result


def enrich_solar_sets(sets,solar):
    return {name:attach_solar(frame,solar) for name,frame in sets.items()}
