"""에어코리아 공식 전국 원본에서 서울 도시대기 측정소 PM10/PM25를 추출한다."""
import csv
import argparse
import gzip
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
import openpyxl
import pandas as pd
from forecast_data import ROOT

DATA = ROOT / 'data/external/air_quality'
STATIONS = DATA / '서울_도시대기_측정소원문.csv.gz'
HOURLY = DATA / '서울_시간별_미세먼지.csv'


def observation_end(value):
    text = str(int(value))
    if len(text) != 10:
        raise ValueError(f'잘못된 관측 시간: {value}')
    hour = int(text[-2:])
    if not 1 <= hour <= 24:
        raise ValueError(f'에어코리아 관측시각 01~24 범위 오류: {value}')
    return datetime.strptime(text[:8],'%Y%m%d') + timedelta(hours=hour)


def aggregate(stations):
    stations = stations.copy()
    stations['측정종료시각'] = pd.to_datetime(stations['측정종료시각'])
    if stations.duplicated(['측정종료시각','측정소코드']).any():
        raise ValueError('측정소·시간 중복')
    for c in ['PM10','PM25']:
        stations[c] = pd.to_numeric(stations[c],errors='coerce')
        stations.loc[stations[c] < 0,c] = float('nan')
    group = stations.groupby('측정종료시각')
    means = group[['PM10','PM25']].mean().rename(columns={'PM10':'미세먼지_PM10','PM25':'초미세먼지_PM25'})
    counts = group[['PM10','PM25']].count().rename(columns={'PM10':'PM10_유효측정소수','PM25':'PM25_유효측정소수'})
    return means.join(counts).reset_index()


def main():
    DATA.mkdir(parents=True,exist_ok=True)
    sources=[]
    with gzip.open(STATIONS,'wt',encoding='utf-8',newline='') as output:
        writer=csv.writer(output)
        writer.writerow(['측정종료시각','원본측정일시','측정소코드','측정소명','지역','망','PM10','PM25'])
        for year in [2017,2018]:
            archive=DATA/f'airkorea_{year}.zip'
            if not archive.exists():
                raise FileNotFoundError(f'공식 전국 원본이 필요합니다: {archive}')
            selected=[]
            with zipfile.ZipFile(archive) as z:
                for item in z.infolist():
                    match=re.search(r'(201[78])년\s*(\d+)월\.xlsx$',item.filename)
                    if match and ((year==2017 and int(match[2]) in [11,12]) or (year==2018 and int(match[2])<=11)):
                        selected.append((int(match[2]),item.filename))
                for month,name in sorted(selected):
                    workbook=openpyxl.load_workbook(io.BytesIO(z.read(name)),read_only=True,data_only=True)
                    sheet=workbook.active
                    records=sheet.iter_rows(values_only=True)
                    header=next(records)
                    required=['지역','망','측정소코드','측정소명','측정일시','PM10','PM25']
                    indices={c:header.index(c) for c in required}
                    total,kept=0,0
                    for row in records:
                        total+=1
                        region=row[indices['지역']]
                        if not isinstance(region,str) or not region.startswith('서울 ') or row[indices['망']]!='도시대기':
                            continue
                        code=str(row[indices['측정소코드']])
                        stamp=row[indices['측정일시']]
                        writer.writerow([observation_end(stamp).isoformat(sep=' '),int(stamp),code,
                                         row[indices['측정소명']],region,'도시대기',
                                         row[indices['PM10']],row[indices['PM25']]])
                        kept+=1
                    workbook.close()
                    sources.append({'year':year,'month':month,'member':name,'national_rows':total,'seoul_urban_rows':kept})
                    output.flush()
                    print(f'{year}-{month:02d}: 서울 도시대기 {kept:,}개 / 전국 {total:,}개',flush=True)
    stations=pd.read_csv(STATIONS)
    hourly=aggregate(stations)
    hourly.to_csv(HOURLY,index=False,encoding='utf-8-sig')
    metadata={
        '자료':'한국환경공단 에어코리아 최종확정 측정자료',
        '확인일':'2026-11-20','공식페이지':'https://www.airkorea.or.kr/web/last_amb_hour_data',
        '원본':[{'year':y,'url':f'https://www.airkorea.or.kr/jfile/readDownloadFile.do?fileId=1657eebba9237&fileSeq={str(y)[-2:]}',
                 'file':f'airkorea_{y}.zip','bytes':(DATA/f'airkorea_{y}.zip').stat().st_size,
                 'sha256':hashlib.sha256((DATA/f'airkorea_{y}.zip').read_bytes()).hexdigest()} for y in [2017,2018]],
        '추출파일':STATIONS.name,'시간별파일':HOURLY.name,'월별추출내역':sources,
        '선택':'지역이 서울로 시작하며 망이 도시대기인 측정소. 도로변 측정소는 제외.',
        '집계':'매시각 유효한 서울 도시대기 측정소의 PM10과 PM25를 각각 산술평균. 측정소 수를 함께 기록.',
        '단위':'PM10과 PM25 모두 ㎍/㎥',
        '결측':'음수(-999 포함)·비수치·빈칸은 결측으로 간주하고 평균에서 제외. 결측을 0으로 바꾸지 않음.',
        '시간표기':'원본 YYYYMMDDHH의 01~24를 시간 종료 표기로 해석. 24는 다음날 00시. 시간대 KST.',
        '모델연결':'예측 입력 시각 t보다 한 시간 이전(t-1)에 종료한 측정값을 사용. 다음 시간 실측 오염도는 사용하지 않음.',
        '한계':'최종확정 자료는 당시 실시간 게시값과 다를 수 있고 게시 지연을 복원하지 못함. 한 시간 지연 가정의 과거자료 실험이며 실시간 운영 성능을 보장하지 않음.',
        '범위':{'시작':str(hourly['측정종료시각'].min()),'끝':str(hourly['측정종료시각'].max()),'시간수':len(hourly),'측정소수':stations['측정소코드'].nunique()},
        '추출본sha256':hashlib.sha256(STATIONS.read_bytes()).hexdigest()
    }
    (DATA/'미세먼지_출처와처리.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print('집계 완료:',len(hourly),'시간',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--from-extracted',action='store_true',help='커밋된 서울 측정소 추출본에서 시간별 평균만 재생성')
    args=parser.parse_args()
    if args.from_extracted:
        saved=json.loads((DATA/'미세먼지_출처와처리.json').read_text(encoding='utf-8'))
        assert hashlib.sha256(STATIONS.read_bytes()).hexdigest()==saved['추출본sha256']
        aggregate(pd.read_csv(STATIONS)).to_csv(HOURLY,index=False,encoding='utf-8-sig')
        print('서울 추출본에서 시간별 평균 재생성 완료')
    else:
        main()
