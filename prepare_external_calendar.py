"""공식 월력요항을 확인해 전사한 날짜에서 과제 기간의 달력 특성을 만든다.
원본 기상/대여 기록이나 정답을 특성 생성에 사용하지 않는다.
"""
from pathlib import Path
from datetime import date, timedelta
import hashlib
import json
import pandas as pd
from forecast_data import RAW, ROOT

EXTERNAL = ROOT / 'data/external'
OUTPUT = ROOT / '외부자료.csv'
SOURCE_2018 = 'https://astro.kasi.re.kr/life/post/almanac?search_year=2018&bbs_uniq_id=almanac'
SOURCE_2017 = 'https://www.timeanddate.com/holidays/south-korea/2017'


def main():
    checked = json.loads((EXTERNAL / '공휴일_확인목록.json').read_text(encoding='utf-8-sig'))
    holidays = {date.fromisoformat(k): v for k, v in checked.items()}
    # Day-level official source cross-check: source table explicitly excludes Constitution Day.
    source = (EXTERNAL / 'kasi_2018.html').read_text(encoding='utf-8')
    for required in ['2018년 월력요항', '6월 13일', '5월 7일', '9월 26일', '관공서의 공휴일이 아님']:
        assert required in source, required
    assert len([d for d in holidays if d.year == 2018]) == 18
    # Scope is deliberately fixed: the 2017 list only covers December, not all of 2017.
    start, end = date(2017,12,1), date(2018,11,30)
    def is_rest(d):
        return d.weekday() >= 5 or d in holidays
    rows = []
    for ts in pd.date_range(start, end, freq='D'):
        day = ts.date()
        first, last = day, day
        if is_rest(day):
            while is_rest(first - timedelta(days=1)):
                first -= timedelta(days=1)
            while is_rest(last + timedelta(days=1)):
                last += timedelta(days=1)
            length, order = (last-first).days + 1, (day-first).days + 1
        else:
            length, order = 0, 0
        rows.append({
            '날짜': day.isoformat(), '요일': '월화수목금토일'[day.weekday()],
            '주말여부': int(day.weekday() >= 5),
            '특별공휴일여부': int(day in holidays),
            '공휴일명': holidays.get(day, ''),
            '휴일여부_주말포함': int(is_rest(day)),
            '휴일시작전날여부': int(not is_rest(day) and is_rest(day + timedelta(days=1))),
            '휴일종료다음날여부': int(not is_rest(day) and is_rest(day - timedelta(days=1))),
            '연속휴일길이_일': length,
            '연속휴일중순번': order,
            '평일근무일여부_달력기준': int(not is_rest(day)),
        })
    frame = pd.DataFrame(rows)
    check = frame.set_index('날짜')
    assert len(frame) == 365 and frame['날짜'].is_unique
    for d in ['2017-12-25','2018-05-07','2018-06-13','2018-09-26']:
        assert check.loc[d,'특별공휴일여부'] == 1
    assert check.loc['2018-07-17','특별공휴일여부'] == 0
    assert check.loc['2018-05-01','특별공휴일여부'] == 0  # Labor Day is not a universal public holiday.
    assert check.loc['2018-09-24','연속휴일길이_일'] == 5
    assert check.loc['2018-09-27','휴일종료다음날여부'] == 1
    # The source count data is read ONLY to verify join coverage, never to derive features.
    raw = pd.read_csv(RAW, encoding='latin1')
    days = pd.to_datetime(raw.Date, format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
    assert days.isin(frame['날짜']).all()
    frame.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
    metadata = {
        '파일명': OUTPUT.name, '확인일': '2026-11-20',
        '내용': '공휴일·주말·연휴 달력 정보. 미세먼지와 예보자료는 포함하지 않음.',
        '기간': {'시작': str(start), '끝': str(end), '일수': len(frame), '원본과연결가능한시간행수': len(raw)},
        '출처': [
            {'기관':'한국천문연구원', '자료':'2018년 공식 월력요항', '주소':SOURCE_2018,
             '보관원문':'data/external/kasi_2018.html',
             '추출방법':'그 외 공휴일·대체휴무일·국경일 표를 확인해 날짜를 수기 전사. 제헌절은 공휴일이 아니므로 제외.',
             'sha256':hashlib.sha256((EXTERNAL/'kasi_2018.html').read_bytes()).hexdigest()},
            {'기관':'Time and Date AS', '자료':'2017년 대한민국 공휴일 목록 중 12월 25일',
             '주소':SOURCE_2017, '확인내용':'2017-12-25 월요일 Christmas Day, Public Holiday',
             '보충':'한국천문연구원 구 월력요항 검색 결과의 12월25일 월요일 기독탄신일과 교차 확인. 현재 2017년 공식 첨부 다운로드는 서버 오류로 보관하지 못함.',
             '교차확인주소':'https://astro.kasi.re.kr/life/pageView/4'}],
        '출처구분':'공식 날짜를 직접 확인해 전사한 목록 + 2017년 12월의 별도 달력 자료 교차 확인. 원본 UCI Holiday 열을 복사하지 않음.',
        '추가정보계산방법': {
            '요일':'월~일', '주말여부':'토요일 또는 일요일이면 1',
            '특별공휴일여부':'별도 확인 목록에 있는 공휴일이면 1. 정기 일요일은 이 열에 자동 포함하지 않음.',
            '공휴일명':'특별공휴일의 명칭. 나머지는 빈칸.',
            '휴일여부_주말포함':'토·일 또는 특별공휴일이면 1',
            '휴일시작전날여부':'오늘은 비휴일이고 내일은 휴일이면 1',
            '휴일종료다음날여부':'오늘은 비휴일이고 어제는 휴일이면 1',
            '연속휴일길이_일':'현재 날짜가 속한 연속 휴일 구간의 전체 길이. 비휴일은 0.',
            '연속휴일중순번':'해당 구간의 몇 번째 날짜인지. 비휴일은 0.',
            '평일근무일여부_달력기준':'월~금 중 특별공휴일이 아닌 날. 개인별 실제 근무나 자전거 운영 여부를 뜻하지 않음.'},
        '단위':'날짜별 한 행, 0=아님/1=해당, 날짜는 한국시간 기준',
        '연결방법':'기존 예측의 target_time(예측 대상 시각)에서 날짜를 추출한 뒤 외부자료.csv의 날짜에 다대일 연결한다.',
        '예측시점정보':'요일·주말·공표된 공휴일 일정으로만 생성. 미래 대여량·실측 날씨·실제 자전거 운영 결과는 입력하지 않음.',
        '한계':['추가한 정보가 성능을 높이는지는 아직 실험하지 않음.',
                '일부 공휴일 여부는 기존 UCI 열과 겹치며 새 차이는 연휴 길이·순번·휴일 전후 정보임.',
                '특별공휴일은 관공서 달력 기준. 근로자의날을 모든 사람의 휴일로 간주하지 않음.',
                '확인목록은 2017년 12월 및 2018년용이며 다른 연도나 2017년 다른 달에 그대로 확장하면 안 됨.']
    }
    (EXTERNAL/'외부자료_출처와설명.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print('External calendar rows:', len(frame))
    print('Original hourly rows covered:', len(raw))
    print('Special holiday dates in period:', int(frame['특별공휴일여부'].sum()))


if __name__ == '__main__':
    main()
