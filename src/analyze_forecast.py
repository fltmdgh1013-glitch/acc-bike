"""Generate descriptive error analysis; never train or select models here."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from forecast_data import OUT, RAW
from korean_labels import translate_frame, model_name, configure_korean_font

configure_korean_font()

MODELS = ['Persistence', 'LinearRegression', 'GradientBoosting']


def markdown_table(frame):
    frame = translate_frame(frame)
    rows = ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
            '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    for record in frame.itertuples(index=False, name=None):
        rows.append('| ' + ' | '.join(f'{x:.2f}' if isinstance(x, float) else str(x) for x in record) + ' |')
    return '\n'.join(rows)


def main():
    p = pd.read_csv(OUT / 'test_predictions.csv', parse_dates=['input_time', 'target_time'])
    selected = json.loads((OUT / 'selection.json').read_text(encoding='utf-8'))['selected_model']
    metrics = pd.read_csv(OUT / 'test_metrics.csv')
    validation = pd.read_csv(OUT / 'validation_metrics.csv')
    splits = pd.read_csv(OUT / 'split_summary.csv')
    audit = json.loads((OUT / 'data_audit.json').read_text(encoding='utf-8'))
    raw = pd.read_csv(RAW, encoding='latin1')
    raw['target_time'] = pd.to_datetime(raw['Date'], format='%d/%m/%Y') + pd.to_timedelta(raw['Hour'], unit='h')
    # Future observations below are for post-hoc diagnosis ONLY, never model inputs.
    context = raw[['target_time', 'Rainfall(mm)', 'Functioning Day']].rename(
        columns={'Rainfall(mm)': 'target_actual_rainfall', 'Functioning Day': 'target_actual_functioning'})
    p = p.merge(context, on='target_time', validate='one_to_one')
    p['hour'] = p.target_time.dt.hour
    p['period'] = p.hour.map(lambda h: 'commute (07-09,17-19)' if h in [7,8,9,17,18,19] else 'other')
    p['target_weather'] = p.target_actual_rainfall.map(lambda r: 'rain' if r > 0 else 'no rain')
    for name in MODELS:
        p[name + '_AE'] = (p[name] - p.target_count).abs()
    ae = [m + '_AE' for m in MODELS]
    hourly = p.groupby('hour')[ae].mean()
    hourly['rows'] = p.groupby('hour').size()
    hourly.to_csv(OUT / 'hourly_errors.csv')
    groups = []
    for key in ['period', 'target_weather', 'target_actual_functioning']:
        for value, part in p.groupby(key):
            groups.append({'group': key, 'value': value, 'rows': len(part),
                           **{m: part[m + '_AE'].mean() for m in MODELS}})
    groups = pd.DataFrame(groups)
    groups.to_csv(OUT / 'group_errors.csv', index=False)
    cols = ['input_time', 'target_time', 'current_count', 'target_count', *MODELS,
            'current_functioning', 'target_actual_functioning', 'Rainfall(mm)', 'target_actual_rainfall']
    worst = {}
    for name in MODELS:
        worst[name] = p.nlargest(5, name + '_AE')[cols + [name + '_AE']]
        worst[name].to_csv(OUT / f'worst_{name}.csv', index=False)
    fig, ax = plt.subplots(figsize=(9,4))
    ax.bar(metrics.model.map(model_name), metrics.MAE, color=['#94a3b8', '#38bdf8', '#14b8a6'])
    ax.set(ylabel='평균 절대 오차(대)', title='다음 1시간 대여량 예측: 시간순 최종 시험')
    for i, v in enumerate(metrics.MAE):
        ax.text(i, v + 2, f'{v:.1f}', ha='center')
    fig.tight_layout(); fig.savefig(OUT / 'model_comparison.png', dpi=160); plt.close(fig)
    # A fixed first-week window, not a cherry-picked best interval.
    week = p.iloc[:168]
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(week.target_time, week.target_count, label='실제 대여량', color='#0f172a')
    ax.plot(week.target_time, week[selected], label=model_name(selected), color='#0d9488', alpha=.85)
    ax.plot(week.target_time, week.Persistence, label='직전 대여량 기준', alpha=.4)
    ax.set(ylabel='대여량(대)', title='최종 시험의 첫 168시간: 실제 대여량과 예측값'); ax.legend()
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(OUT / 'first_week_predictions.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,4))
    for name in MODELS:
        ax.plot(hourly.index, hourly[name + '_AE'], marker='.', label=model_name(name))
    ax.set(xlabel='예측 대상 시각(시)', ylabel='평균 절대 오차(대)', title='시간대별 평균 예측 오차', xticks=range(0,24,2))
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / 'hourly_errors.png', dpi=160); plt.close(fig)
    baseline = metrics.set_index('model').loc['Persistence', 'MAE']
    best = metrics.set_index('model').loc[selected, 'MAE']
    lines = [
        '# 서울 자전거 다음 1시간 대여량 예측 결과 (초기 3개 모델)',
        '순환 신경망(RNN)·트랜스포머를 추가한 최신 결과는 [5개 모델 비교 보고서](sequence/COMPARISON.md)를 참조한다. 아래는 초기 실험의 기록이다.',
        '## 1. 문제와 데이터',
        '현재 시간대의 집계가 끝난 뒤, 그 기록으로 다음 시간대의 전체 대여 건수를 예측한다. '
        '특정 대여소의 재고나 남은 자전거 수를 예측하는 문제가 아니다. 9시 기록으로 10시를 맞히는 것은 모든 시간에 적용하는 규칙의 예다.',
        f"UCI 원본 {audit['source_rows']:,}행에서 {audit['examples']:,}개의 연속 1시간 사례를 구성했다. "
        f"결측 {audit['missing_cells']}칸, 중복 시간 {audit['duplicate_timestamps']}개, 누락 시간 간격 {audit['gap_pairs_excluded']}개. "
        f"운영하지 않는 원본 기록 {audit['non_operating_rows']}개도 유지한다. 마지막 행은 다음 시간 정답이 없어 제외한다.",
        '입력: 현재 대여량, 현재 기온·습도·풍속·가시거리·이슬점·일사량·강수·적설, 현재 휴일·계절·운영 상태, 다음 시간의 시각(sin/cos)·요일·월. '
        '미래 실측 날씨, 미래 대여량, 미래 실제 운영 상태는 입력하지 않는다. 시각은 23시와 0시가 가깝게 표현되도록 원형 변환한다.',
        '출처: https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand (CC BY 4.0). 원본 해시와 다운로드 내역은 data/raw/SOURCE.md.',
        '## 2. 자료 분리와 누수 방지', markdown_table(splits),
        '정답 시간순으로 70/15/15%를 나눴다. 무작위 셔플은 하지 않았다. 표준화와 범주 인코딩은 학습 구간으로만 적합한다. '
        '매시간 실제 현재 기록이 들어오는 1시간 예측을 반복한다. 시험 구간의 이전 실제 대여량을 다음 예측 입력으로 쓰는 것은 이 운영 조건에서 허용된다. '
        '며칠 앞을 한꺼번에 예측하거나, 과거 예측값만 연쇄 입력하는 시험과는 다르다.',
        '## 3. 비교 방법',
        '- 직전 대여량 기준: 현재 대여량을 그대로 다음 시간 예측값으로 사용.\n'
        '- 선형회귀: 동일한 입력으로 학습하는 선형회귀.\n'
        '- 그래디언트 부스팅: 깊이 3의 작은 결정트리 200개, 학습률 0.05, 시드 42. 비선형 관계를 학습하는 추가 비교 모델.\n'
        '- 학습 모델의 음수 예측은 0으로 제한하며 소수 예측은 반올림하지 않는다.',
        'MAE는 평균 절대 오차이며 단위는 대다. RMSE는 큰 오차에 더 민감한 보조 지표, R²는 설명력의 보조 지표다.',
        '### 검증 결과 (모델 선택용)', markdown_table(validation),
        f'검증 MAE가 가장 낮은 {model_name(selected)}를 선택하고 고정했다. 검증 자료로 재학습하지 않았으며 최종 시험 점수로 모델이나 설정을 바꾸지 않았다.',
        '### 최종 시험 결과', markdown_table(metrics),
        f'선택 모델의 MAE는 {best:.2f}대이며, 기준모델 {baseline:.2f}대보다 {(baseline-best)/baseline*100:.1f}% 작다. 이 비율은 MAE 감소율이며 예측 정확도가 아니다.',
        '![모델 비교](model_comparison.png)',
        '![첫 일주일](first_week_predictions.png)',
        '## 4. 오류 분석', '![시간대별 오차](hourly_errors.png)', markdown_table(groups),
        '위 그룹 표에서 ‘실제 다음 시간 강수 여부’와 ‘실제 다음 시간 운영 상태’은 예측 후 확인한 진단용 정보다. 모델 입력과 명확히 분리했다. '
        '강수·비운영 사례의 표본 수를 함께 확인해야 한다. 단순한 그룹 차이만으로 원인을 확정할 수 없다.',
        '기준모델은 수요가 일정할 때 유리하지만 급상승·급하락을 한 시간 늦게 따라간다. 선형회귀는 시간·날씨 효과가 일정하게 더해진다고 가정하므로 '
        '출퇴근 변화나 운영 중단 같은 급격한 변화를 충분히 설명하지 못할 수 있다. 트리 모델도 학습하지 않은 전환이나 급변에는 오차가 남는다.',
    ]
    commute = groups.loc[(groups['group'] == 'period') & (groups['value'] == 'commute (07-09,17-19)')].iloc[0]
    other = groups.loc[(groups['group'] == 'period') & (groups['value'] == 'other')].iloc[0]
    stopped = groups.loc[(groups['group'] == 'target_actual_functioning') & (groups['value'] == 'No')].iloc[0]
    lines += [f"선택 모델의 출퇴근 구간 MAE는 {commute[selected]:.2f}대, 나머지 시간은 {other[selected]:.2f}대다. "
              f"출퇴근 급변 구간에서 오차가 더 크다. 반면 실제 비운영 {stopped['rows']}개 사례에서는 "
              f"기준모델 {stopped['Persistence']:.2f}대가 선택 모델 {stopped[selected]:.2f}대보다 오차가 작다. "
              '종합 점수가 좋은 모델이 모든 조건에서 우수한 것은 아니다. 비가 온 시험 사례는 30개로 적어 강수에 대한 일반적인 결론은 유보한다.']
    for name in MODELS:
        lines += [f'### {model_name(name)}: 절대오차 상위 5개', markdown_table(worst[name])]
    # Numerical change/error evidence rather than an invented causal narrative.
    for name in MODELS:
        r = worst[name].iloc[0]
        lines += [f"{model_name(name)}의 최대 오차 사례: {r.target_time}, 현재 {r.current_count:.0f}대 → 실제 {r.target_count:.0f}대, "
                  f"예측 {r[name]:.1f}대, 오차 {r[name + '_AE']:.1f}대. "
                  f"관측 운영 상태는 {'운영' if r.current_functioning == 'Yes' else '비운영'} → {'운영' if r.target_actual_functioning == 'Yes' else '비운영'}. "
                  '이 숫자와 운영·강수 기록은 함께 움직인 현상에 대한 근거이며 인과관계를 증명하지 않는다.']
    lines += ['## 5. 한계와 다음 실험',
              '한 해 자료를 한 번 시간순 분할했으므로 학습·검증·시험의 계절 분포가 다르다. 최신 두 달의 결과를 다른 연도나 모든 계절에 일반화할 수 없다. '
              '현재 시간대 집계가 예측 시점에 제공된다는 가정도 필요하다. 원본에 기록된 집계 외에 실시간 제공 지연은 확인하지 않았다. '
              '과거 24시간을 사용하는 순환 신경망(RNN)·트랜스포머 추가 실험은 별도 비교 보고서에 정리했다. 향후 다수의 과거 시점 분할, 운영 일정 사전 확보를 검토할 수 있다. 이번 시험 결과를 보고 추가 튜닝하지 않았다.',
              '## 6. 재현과 기여',
              'README.md의 실행 순서로 원본 CSV에서 표·그림을 재생성한다. 팀원별 실제 기여와 협업 사례는 팀이 직접 기록해야 한다. '
              '성능 개선 결과는 이번 실행에서 계산했으며 기존 저장소의 결과를 재사용하지 않았다.']
    (OUT / 'REPORT.md').write_text('\n\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Wrote analysis to {OUT}')


if __name__ == '__main__':
    main()
