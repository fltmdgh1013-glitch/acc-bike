"""예측 대상 날짜의 미리 알 수 있는 달력 정보만 연결한다."""
import pandas as pd
from forecast_data import ROOT

EXTERNAL_FILE = ROOT / '외부자료.csv'
EXTERNAL_COLUMNS = ['주말여부', '특별공휴일여부', '휴일여부_주말포함',
                    '휴일시작전날여부', '휴일종료다음날여부', '연속휴일길이_일',
                    '연속휴일중순번', '평일근무일여부_달력기준']


def attach_calendar(frame, calendar):
    calendar = calendar[['날짜', *EXTERNAL_COLUMNS]].copy()
    calendar['날짜'] = pd.to_datetime(calendar['날짜'], format='%Y-%m-%d').dt.normalize()
    if calendar['날짜'].duplicated().any():
        raise ValueError('외부자료에 중복 날짜가 있습니다.')
    if calendar[EXTERNAL_COLUMNS].isna().any().any():
        raise ValueError('외부자료에 결측값이 있습니다.')
    result = frame.copy()
    result['_external_date'] = result.target_time.dt.normalize()
    result['_external_order'] = range(len(result))
    result = result.merge(calendar, left_on='_external_date', right_on='날짜',
                          how='left', validate='many_to_one', sort=False)
    if result[EXTERNAL_COLUMNS].isna().any().any():
        raise ValueError('예측 대상 날짜에 대응하는 외부자료가 없습니다.')
    result = result.sort_values('_external_order').drop(columns=['날짜','_external_date','_external_order'])
    result.index = frame.index
    return result


def enrich_sets(sets, calendar):
    return {key: attach_calendar(frame, calendar) for key, frame in sets.items()}
