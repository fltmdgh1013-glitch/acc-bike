"""Post-hoc high-demand classification of frozen regression predictions."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from forecast_data import OUT, RAW
from analyze_forecast import markdown_table
from korean_labels import model_name

DEST = OUT / 'timing'
THRESHOLD = 400  # Teacher's example, not optimized on test scores.
MODELS = ['Persistence','LinearRegression','GradientBoosting','RNN','Transformer']


def ratio(a, b):
    return float(a / b) if b else float('nan')


def binary_scores(actual, predicted, threshold=THRESHOLD):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.shape != predicted.shape or actual.ndim != 1:
        raise ValueError('Expected aligned one-dimensional counts')
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('Counts must be finite')
    truth, alarm = actual >= threshold, predicted >= threshold
    tp, fp = int((truth & alarm).sum()), int((~truth & alarm).sum())
    fn, tn = int((truth & ~alarm).sum()), int((~truth & ~alarm).sum())
    recall, specificity = ratio(tp, tp+fn), ratio(tn, tn+fp)
    return dict(n=len(actual), positives=tp+fn, negatives=tn+fp,
                TP=tp, FP=fp, FN=fn, TN=tn,
                positive_rate=ratio(tp+fn, len(actual)),
                precision=ratio(tp, tp+fp), recall=recall,
                accuracy=ratio(tp+tn, len(actual)), F1=ratio(2*tp, 2*tp+fp+fn),
                specificity=specificity,
                balanced_accuracy=(recall+specificity)/2)


def timing_masks(times):
    times = pd.DatetimeIndex(times)
    h = times.hour
    weekday = times.day_of_week < 5
    morning, evening = (h >= 7) & (h <= 9), (h >= 17) & (h <= 19)
    return {'all': np.ones(len(times), dtype=bool),
            'morning_all_days': morning, 'evening_all_days': evening,
            'commute_all_days': morning | evening,
            'weekday_morning': weekday & morning, 'weekday_evening': weekday & evening,
            'weekday_commute': weekday & (morning | evening),
            'non_commute_all_days': ~(morning | evening)}


def percent_table(frame):
    frame = frame.copy()
    for c in ['positive_rate','precision','recall','accuracy','F1','specificity','balanced_accuracy']:
        if c in frame:
            frame[c] = frame[c].map(lambda v: '-' if pd.isna(v) else f'{100*v:.1f}%')
    return markdown_table(frame)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    p = pd.read_csv(OUT / 'sequence/all_test_predictions.csv', parse_dates=['input_time','target_time'])
    if p.target_time.duplicated().any() or not p.target_time.is_monotonic_increasing:
        raise ValueError('Ambiguous test timestamps')
    masks = timing_masks(p.target_time)
    raw = pd.read_csv(RAW, encoding='latin1')
    raw['target_time'] = pd.to_datetime(raw.Date, format='%d/%m/%Y') + pd.to_timedelta(raw.Hour, unit='h')
    context = p[['target_time']].merge(raw[['target_time','Functioning Day']], on='target_time', validate='one_to_one')
    # Actual future operating status is post-hoc diagnostic context only.
    active = (context['Functioning Day'] == 'Yes').to_numpy()
    masks['weekday_commute_actual_operating'] = masks['weekday_commute'] & active
    low = p.target_count < THRESHOLD
    low_commute = int((masks['weekday_commute'] & low).sum())
    low_closed = int((masks['weekday_commute'] & low & ~active).sum())
    # This untrained baseline exposes how easy 100% recall can be.
    p['AlwaysHigh'] = float(THRESHOLD)
    rows, hourly_rows = [], []
    for group, mask in masks.items():
        part = p.loc[mask]
        for name in MODELS + ['AlwaysHigh']:
            rows.append({'group': group, 'model': name,
                         **binary_scores(part.target_count, part[name])})
    for hour, part in p.groupby(p.target_time.dt.hour):
        for name in MODELS + ['AlwaysHigh']:
            hourly_rows.append({'hour': hour, 'model': name,
                                **binary_scores(part.target_count, part[name])})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(DEST / 'timing_metrics.csv', index=False)
    pd.DataFrame(hourly_rows).to_csv(DEST / 'hourly_metrics.csv', index=False)
    failures = []
    for name in MODELS:
        part = p.loc[masks['weekday_commute']].copy()
        part['model'] = name
        part['prediction'] = part[name]
        part['error_type'] = np.where(part[name] >= THRESHOLD, 'false_alarm', 'missed_high_demand')
        wrong = (part.target_count >= THRESHOLD) != (part[name] >= THRESHOLD)
        failures.append(part.loc[wrong, ['model','input_time','target_time','target_count','prediction','error_type']])
    errors = pd.concat(failures, ignore_index=True)
    errors.to_csv(DEST / 'weekday_commute_errors.csv', index=False)
    config = {'event': 'target_count >= 400 bikes/hour', 'threshold': THRESHOLD,
              'threshold_source': 'Week12 teacher example, no test threshold optimization',
              'morning_target_hours': [7,8,9], 'evening_target_hours': [17,18,19],
              'weekday_definition': 'Monday-Friday; public holidays are not excluded',
              'model_retrained': False, 'model_selection_changed': False,
              'undefined_denominator': 'NaN in CSV, dash in report',
              'scope': 'post-hoc descriptive evaluation of previously seen test period'}
    (DEST / 'protocol.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1,3,figsize=(14,6), sharey=True)
    colors = ['#94a3b8','#38bdf8','#14b8a6','#a78bfa','#f59e0b','#d1d5db']
    for ax, group, title in zip(axes, ['weekday_morning','weekday_evening','weekday_commute'],
                               ['평일 출근(07~09시)','평일 퇴근(17~19시)','평일 출퇴근 합계']):
        part = metrics.loc[metrics.group == group]
        x = np.arange(len(part))
        ax.bar(x-.18, part.recall*100, .36, label='재현율', color=colors)
        ax.bar(x+.18, part.precision*100, .36, label='정밀도', color=colors, alpha=.4, hatch='//')
        ax.set_xticks(x, part.model.map(model_name), rotation=65, ha='right', fontsize=8)
        ax.set_ylim(0,105); ax.set_title(title,fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=10)
    axes[0].set_ylabel('비율(%): 다음 시간 400대 이상을 고수요로 판단')
    fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(DEST / 'commute_precision_recall.png',dpi=160, bbox_inches='tight', pad_inches=.2); plt.close(fig)
    support = metrics.loc[metrics.model == 'Persistence', ['group','n','positives','negatives','positive_rate']]
    lines = ['# 주요 시간대의 고수요 탐지 비교',
             '## 무엇을 맞혔는가',
             '**다음 시간 대여량 400대 이상**을 양성(고수요)으로 정의한다. 교수님 자료의 분류 예시를 그대로 썼고, '
             '시험 점수를 높이기 위해 기준을 고르지 않았다. 기존 회귀 예측값에도 같은 400대 기준을 적용한다. '
             '새 분류 모델을 학습한 결과가 아니라, 이미 고정한 대여량 예측으로 고수요 여부를 판단한 결과다.',
             '출근 07·08·09시, 퇴근 17·18·19시는 **예측 대상 시간**이다. 예를 들어 06시 기록으로 07시를 예측한 사례는 출근 구간이다. '
             '평일은 월~금이며 공휴일은 별도로 제외하지 않았다. 전체 요일 결과도 CSV와 아래 표에 함께 제공한다.',
             '## 지표를 쉽게 읽는 법',
             '- 재현율(Recall): 실제 400대 이상이었던 시간 중 고수요라고 잡아낸 비율. TP / (TP+FN).\n'
             '- 정밀도(Precision): 고수요라고 예측한 시간 중 실제로 고수요였던 비율. TP / (TP+FP).\n'
             '- 정확도(Accuracy): 고수요와 비고수요 판단 전체가 맞은 비율. (TP+TN) / 전체.\n'
             '- F1: 정밀도와 재현율의 조화평균. 특이도는 실제 비고수요 중 비고수요로 맞힌 비율.\n'
             '- TP는 고수요 적중, FP는 잘못 울린 경보, FN은 놓친 고수요, TN은 비고수요 적중. 분모가 0이면 정의 불가(-)로 표시한다.',
             '정의 참고: [scikit-learn 공식 평가 지표 문서](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics).',
             '## 표본 수와 고수요 비율', percent_table(support),
             '## 평일 주요 시간대 비교',
             '![출퇴근 정밀도와 재현율](commute_precision_recall.png)']
    cols = ['model','precision','recall','accuracy','F1','TP','FP','FN','TN']
    for group, title in [('weekday_morning','출근'),('weekday_evening','퇴근'),('weekday_commute','출퇴근 합계')]:
        lines += [f'### 평일 {title}', percent_table(metrics.loc[metrics.group == group, cols])]
    lines += ['## 전체 요일 출퇴근 비교', percent_table(metrics.loc[metrics.group == 'commute_all_days', cols]),
              '## 해석과 한계',
              '400대는 수업의 예시 기준이며 실제 운영상 적절한 경계라고 검증한 값은 아니다. 출퇴근에는 대부분 400대 이상일 수 있다. '
              '따라서 항상 고수요라고만 말해도 재현율 100%가 된다. ‘항상 고수요로 예측’ 행과 고수요 비율, FP·TN을 함께 봐야 한다. '
              '정확도나 재현율 하나만 높다고 더 좋은 모델이라고 할 수 없다. 특히 비고수요가 적으면 그 탐지 성능은 소수 사례에 좌우된다.',
              f'실제 평일 출퇴근 비고수요 {low_commute}개 중 {low_closed}개는 실제 비운영 시간이었다. '
              '실제로 운영한 시간만 사후적으로 분리한 다음 표를 함께 확인해야 한다. 이 분리는 예측 뒤 확인한 운영 정보로 만든 진단용이며 모델 입력이나 의사결정 규칙으로 사용하지 않았다.',
              percent_table(metrics.loc[metrics.group == 'weekday_commute_actual_operating', cols]),
              '전체 대여량의 MAE가 가장 낮은 모델과 400대 경계를 잘 구분하는 모델은 다를 수 있다. '
              '399대와 401대를 혼동하면 2대 오차인데도 분류에서는 오답이고, 500대와 1500대는 1000대 차이여도 둘 다 고수요로 정답이다.',
              '이번 결과는 기존 시험 기간을 본 뒤 주요 시간대를 기술적으로 비교한 추가 분석이다. '
              '모델·학습 횟수·고수요 기준을 이 점수에 맞춰 다시 최적화하지 않았고 기존 검증 MAE 기반 선택도 변경하지 않았다. '
              '다른 계절·연도의 독립 자료에서 재검증이 필요하다.',
              '## 평일 출퇴근 오판 내역',
              '`weekday_commute_errors.csv`에 모든 오판을 저장했다. 아래는 모델별 놓친 고수요와 잘못 울린 경보다.']
    for name in MODELS:
        lines += [f'### {model_name(name)}', markdown_table(errors.loc[errors.model == name].head(10))]
    lines += ['## 실행', '`python src/evaluate_timing.py`로 재학습 없이 재생성한다. 전체 과정은 `python src/run_forecast.py`. '
              '`timing_metrics.csv`는 0~1 비율이며 보고서·그림은 %로 표현한다. `hourly_metrics.csv`에는 시각별 24개 구간의 지표도 들어 있다.']
    (DEST / 'REPORT.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    print(metrics.loc[metrics.group.str.startswith('weekday'), ['group',*cols]].to_string(index=False))


if __name__ == '__main__':
    main()
