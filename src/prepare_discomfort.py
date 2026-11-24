"""보관된 서울 자전거 원본으로 시간별 불쾌지수와 계산 근거를 보관한다."""
import hashlib
import json
import pandas as pd
from forecast_data import ROOT,RAW
from discomfort_features import DISCOMFORT_FILE,discomfort_index,TEMPERATURE,HUMIDITY


def main():
    raw=pd.read_csv(RAW,encoding='latin1')
    times=pd.to_datetime(raw.Date,format='%d/%m/%Y')+pd.to_timedelta(raw.Hour,unit='h')
    assert len(raw)==8760 and not times.duplicated().any()
    assert times.is_monotonic_increasing and times.diff().dropna().eq(pd.Timedelta(hours=1)).all()
    frame=pd.DataFrame({'입력시각':times,'기온_C':raw[TEMPERATURE],'상대습도_pct':raw[HUMIDITY],
                        '현재_불쾌지수':discomfort_index(raw[TEMPERATURE],raw[HUMIDITY])})
    frame.to_csv(DISCOMFORT_FILE,index=False,encoding='utf-8-sig')
    folder=ROOT/'data/external/discomfort';folder.mkdir(parents=True,exist_ok=True)
    pdf=ROOT/'data/external/weather_candidates/kma_discomfort_index_20110714.pdf'
    previous=json.loads((ROOT/'data/external/weather_candidates/SOURCE.json').read_text(encoding='utf-8'))
    source=next(item for item in previous['자료'] if item['보관파일'].endswith('.pdf'))
    assert hashlib.sha256(pdf.read_bytes()).hexdigest()==source['SHA256']
    meta={'내용':'외부 관측 추가 없이 원본의 현재 기온·상대습도로 계산한 파생 입력',
          '기간':{'시작':str(times.min()),'종료':str(times.max()),'행수':len(frame),'시간대':'KST'},
          '산식':'DI = 1.8*T - 0.55*(1-H/100)*(1.8*T-26) + 32',
          '단위':{'T':'기온 °C','H':'상대습도 %','DI':'지수'},'원문습도단위':'0~1 소수이므로 % 습도는 100으로 나눔',
          '입력시각':'현재 input_time. 다음 시간 실제 기온·습도는 사용하지 않음.',
          '추가열':['현재_불쾌지수'],'원본sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),
          '계산파일':DISCOMFORT_FILE.name,'계산파일sha256':hashlib.sha256(DISCOMFORT_FILE.read_bytes()).hexdigest(),
          '산식근거':{**source,'쪽':2},'재현':'python src/prepare_discomfort.py',
          '검사':{'결측':int(frame.isna().sum().sum()),'불쾌지수최소':float(frame['현재_불쾌지수'].min()),
                  '불쾌지수최대':float(frame['현재_불쾌지수'].max())},
          '한계':'2011년 과거 설명자료의 여름철 무더위 산식. 공식 예보 지수의 복원이나 구별 기상자료의 공간 집계가 아님. 원본 제공 기온·습도를 그대로 사용. 겨울 값은 수치로 보존하되 여름 체감 단계로 해석하지 않음. 복사·바람 미반영.'}
    (folder/'SOURCE.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(DISCOMFORT_FILE, len(frame), 'rows', flush=True)


if __name__=='__main__':main()
