"""보고서·그래프·열람용 CSV에 사용하는 공통 한국어 표기."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

MODEL_NAMES = {
 'Persistence': '직전 대여량 기준', 'LinearRegression': '선형회귀',
 'GradientBoosting': '그래디언트 부스팅', 'RNN': '순환 신경망(RNN)',
 'Transformer': '트랜스포머', 'AlwaysHigh': '항상 고수요로 예측',
}
COLUMNS = {
 '미세먼지_PM10_1시간지연':'이전 시간 미세먼지 PM10(㎍/㎥)',
 '초미세먼지_PM25_1시간지연':'이전 시간 초미세먼지 PM2.5(㎍/㎥)',
 'PM10_결측여부':'미세먼지 결측 여부', 'PM25_결측여부':'초미세먼지 결측 여부',
 'variant':'외부자료 조건', 'validation_before_MAE':'검증 오차: 추가 전(대)',
 'validation_after_MAE':'검증 오차: 추가 후(대)', 'validation_delta_MAE':'검증 오차 변화(후-전·대)',
 'test_before_MAE':'시험 오차: 추가 전(대)', 'test_after_MAE':'시험 오차: 추가 후(대)',
 'test_delta_MAE':'시험 오차 변화(후-전·대)',
 'before_prediction':'추가 전 예측(대)', 'after_prediction':'추가 후 예측(대)',
 'before_absolute_error':'추가 전 절대 오차(대)', 'after_absolute_error':'추가 후 절대 오차(대)',
 'error_change':'절대 오차 변화(후-전·대)',
 'model':'모델', 'MAE':'평균 절대 오차(대)', 'RMSE':'제곱평균제곱근 오차(대)', 'R2':'결정계수(R²)',
 'selected_by_validation':'검증으로 선택', 'split':'자료 구분', 'rows':'사례 수', 'n':'사례 수',
 'input_start':'입력 시작 시각', 'input_end':'입력 마지막 시각',
 'target_start':'정답 시작 시각', 'target_end':'정답 마지막 시각',
 'input_time':'입력 시각', 'target_time':'예측 대상 시각',
 'current_count':'현재 대여량(대)', 'target_count':'실제 다음 시간 대여량(대)',
 'prediction':'예측 대여량(대)', 'absolute_error':'절대 오차(대)',
 'current_functioning':'현재 운영 상태', 'target_actual_functioning':'실제 다음 시간 운영 상태',
 'Rainfall(mm)':'현재 강수량(mm)', 'target_actual_rainfall':'실제 다음 시간 강수량(mm)',
 'hour':'예측 대상 시각(시)', 'group':'시간대·분석 구분', 'value':'구분 값', 'period':'시간대',
 'target_weather':'실제 다음 시간 강수 여부', 'train_rows':'학습 사례 수',
 'best_epoch':'선택한 학습 반복 횟수', 'epochs_run':'실제 학습 반복 횟수', 'epoch':'학습 반복 횟수',
 'parameters':'학습 계수 수', 'train_scaled_MSE':'학습 오차(표준화 MSE)',
 'validation_MAE':'검증 평균 절대 오차(대)', 'positives':'실제 고수요 수', 'negatives':'실제 비고수요 수',
 'positive_rate':'실제 고수요 비율', 'precision':'정밀도', 'recall':'재현율', 'accuracy':'정확도',
 'F1':'정밀도·재현율 종합(F1)', 'specificity':'비고수요 재현율(특이도)',
 'balanced_accuracy':'고수요·비고수요 평균 재현율', 'TP':'고수요 적중(TP)', 'FP':'잘못 울린 경보(FP)',
 'FN':'놓친 고수요(FN)', 'TN':'비고수요 적중(TN)', 'error_type':'오판 종류',
 'Temperature(°C)':'현재 기온(℃)', 'Humidity(%)':'현재 습도(%)',
 'Wind speed (m/s)':'현재 풍속(m/s)', 'Visibility (10m)':'현재 가시거리(10m)',
 'Dew point temperature(°C)':'현재 이슬점 온도(℃)', 'Solar Radiation (MJ/m2)':'현재 일사량(MJ/m²)',
 'Snowfall (cm)':'현재 적설량(cm)', 'target_hour_sin':'다음 시간 시각의 사인값',
 'target_hour_cos':'다음 시간 시각의 코사인값', 'target_weekday':'다음 시간 요일(월=0)',
 'target_month':'다음 시간 월', 'current_holiday':'현재 휴일 여부', 'current_season':'현재 계절',
}
VALUES = {**MODEL_NAMES,
 'before':'외부자료 추가 전', 'after':'외부자료 추가 후',
 'calendar_rest_day':'달력상 휴일(주말 포함)', 'calendar_work_day':'달력상 비휴일',
 'train':'학습', 'validation':'검증', 'test':'최종 시험',
 'all':'전체 시간', 'morning_all_days':'출근 시간(모든 요일)', 'evening_all_days':'퇴근 시간(모든 요일)',
 'commute_all_days':'출퇴근 합계(모든 요일)', 'weekday_morning':'평일 출근 시간',
 'weekday_evening':'평일 퇴근 시간', 'weekday_commute':'평일 출퇴근 합계',
 'non_commute_all_days':'출퇴근 외 시간(모든 요일)',
 'weekday_commute_actual_operating':'평일 출퇴근 중 실제 운영 시간(사후 진단)',
 'commute (07-09,17-19)':'출퇴근(07~09시·17~19시)', 'other':'출퇴근 외 시간',
 'period':'시간대', 'target_weather':'실제 다음 시간 강수 여부',
 'target_actual_functioning':'실제 다음 시간 운영 상태',
 'rain':'비가 옴', 'no rain':'비가 오지 않음', 'Yes':'운영', 'No':'비운영',
 'Winter':'겨울', 'Spring':'봄', 'Summer':'여름', 'Autumn':'가을',
 'Holiday':'휴일', 'No Holiday':'휴일 아님',
 'false_alarm':'잘못 울린 경보', 'missed_high_demand':'놓친 고수요',
}


def model_name(name):
    return MODEL_NAMES.get(name, name)


def column_name(name):
    if name.endswith('_AE'):
        return model_name(name[:-3]) + ' 절대 오차(대)'
    return MODEL_NAMES.get(name, COLUMNS.get(name, name))


def translate_frame(frame):
    translated = frame.copy()
    for col in translated:
        if col == 'selected_by_validation':
            translated[col] = translated[col].map({True:'선택', False:'미선택'})
        elif pd.api.types.is_string_dtype(translated[col].dtype) or translated[col].dtype == object:
            translated[col] = translated[col].map(lambda v: VALUES.get(v, v) if isinstance(v, str) else v)
    translated.columns = [column_name(str(c)) for c in translated.columns]
    return translated


def configure_korean_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((name for name in ['Malgun Gothic','AppleGothic','NanumGothic','Noto Sans CJK KR'] if name in available), None)
    if chosen is None:
        raise RuntimeError('한글 그래프용 글꼴이 없습니다. 맑은 고딕 또는 나눔고딕을 설치하세요.')
    plt.rcParams['font.family'] = chosen
    plt.rcParams['axes.unicode_minus'] = False
    return chosen


def export_korean_csvs():
    from forecast_data import OUT
    count = 0
    for path in sorted(OUT.rglob('*.csv')):
        if path.stem.endswith('_한국어'):
            continue
        translated = translate_frame(pd.read_csv(path))
        translated.to_csv(path.with_name(path.stem + '_한국어.csv'), index=False, encoding='utf-8-sig')
        count += 1
    print(f'한국어 열람용 CSV {count}개 생성 완료')


if __name__ == '__main__':
    export_korean_csvs()
