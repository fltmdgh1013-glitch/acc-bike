"""보관한 Open-Meteo 일출·일몰 응답으로 날짜별 외부자료를 재생성한다."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from urllib.request import urlopen
import pandas as pd
from forecast_data import ROOT

SOURCE = ROOT/'data/external/solar/open_meteo_sunrise_sunset.json'
SOLAR_FILE = ROOT/'외부자료_일출일몰.csv'
URL = ('https://archive-api.open-meteo.com/v1/archive?latitude=37.5665&longitude=126.9780'
       '&start_date=2017-12-01&end_date=2018-11-30&daily=sunrise,sunset&timezone=Asia%2FSeoul')


def main(download=False):
    if download:
        SOURCE.parent.mkdir(parents=True,exist_ok=True)
        with urlopen(URL,timeout=45) as response:
            SOURCE.write_bytes(response.read())
    raw=json.loads(SOURCE.read_text(encoding='utf-8'))
    assert raw['timezone']=='Asia/Seoul' and raw['utc_offset_seconds']==32400
    daily=raw['daily']
    df=pd.DataFrame({'날짜':pd.to_datetime(daily['time']),
                     '일출시각':pd.to_datetime(daily['sunrise']), '일몰시각':pd.to_datetime(daily['sunset'])})
    expected=pd.date_range('2017-12-01','2018-11-30',freq='D')
    assert pd.DatetimeIndex(df['날짜']).equals(expected)
    assert df.notna().all().all() and (df['일출시각']<df['일몰시각']).all()
    assert (df['일출시각'].dt.normalize()==df['날짜']).all()
    assert (df['일몰시각'].dt.normalize()==df['날짜']).all()
    df['낮길이_분']=(df['일몰시각']-df['일출시각']).dt.total_seconds()/60
    df.to_csv(SOLAR_FILE,index=False,encoding='utf-8-sig',date_format='%Y-%m-%d %H:%M:%S')
    note={'내용':'서울 대표 좌표 일출·일몰 천문 계산값. 실제 일조시간·구름 관측값 아님.',
          '기간':{'시작':'2017-12-01','끝':'2018-11-30','날짜수':len(df),'원본시간수':8760},
          '요청좌표':{'위도':37.5665,'경도':126.9780},
          '응답격자좌표':{'위도':raw['latitude'],'경도':raw['longitude']},
          '시간대':'Asia/Seoul (UTC+9)','시각정밀도':'응답 ISO8601 분 단위',
          '출처':URL,'문서':'https://open-meteo.com/en/docs/historical-weather-api',
          '천문계산설명':'https://open-meteo.com/en/docs/gem-api',
          '보관응답':str(SOURCE.relative_to(ROOT)).replace('\\','/'),
          'sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'CSVsha256':hashlib.sha256(SOLAR_FILE.read_bytes()).hexdigest(),
          '재생성':'python src/prepare_solar.py (다운로드는 --download)',
          '특성연결':'예측 대상 target_time의 날짜, KST 정시 순간 기준. 일출 포함·일몰 제외.',
          '한계':'대표 한 지점의 천문 시각이며 지형·건물·구름·조명·실제 일조량을 반영하지 않음. API가 반환한 격자좌표도 보관.'}
    metadata=SOURCE.parent/'SOURCE.json'
    if metadata.exists():
        previous=json.loads(metadata.read_text(encoding='utf-8'))
        if previous.get('sha256')==note['sha256']:
            note['수집시각UTC']=previous['수집시각UTC']
    note.setdefault('수집시각UTC',datetime.now(timezone.utc).isoformat())
    metadata.write_text(json.dumps(note,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'{SOLAR_FILE}: {len(df)} days')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--download',action='store_true')
    main(parser.parse_args().download)
