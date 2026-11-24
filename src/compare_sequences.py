"""Combine frozen experiments on exactly the same validation/test targets."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from forecast_data import OUT
from sequence_models import SEQ_OUT
from solution import scores, select_model
from analyze_forecast import markdown_table
from korean_labels import model_name


def main():
    val = pd.read_csv(OUT / 'validation_metrics.csv')
    test = pd.read_csv(OUT / 'test_metrics.csv').drop(columns='selected_by_validation')
    pred = pd.read_csv(OUT / 'test_predictions.csv', parse_dates=['input_time', 'target_time'])
    metadata = []
    for name in ['RNN', 'Transformer']:
        m = pd.read_csv(SEQ_OUT / f'{name}_metrics.csv')
        val = pd.concat([val, m.loc[m.split == 'validation'].drop(columns='split')], ignore_index=True)
        test = pd.concat([test, m.loc[m.split == 'test'].drop(columns='split')], ignore_index=True)
        new = pd.read_csv(SEQ_OUT / f'{name}_test_predictions.csv', parse_dates=['input_time', 'target_time'])
        # Fail rather than compare different hours or labels by accident.
        pd.testing.assert_frame_equal(pred[['input_time', 'target_time', 'target_count']],
                                      new[['input_time', 'target_time', 'target_count']])
        pred[name] = new[name]
        measured = scores(pred.target_count, pred[name])
        assert abs(measured['MAE'] - float(m.loc[m.split == 'test', 'MAE'].iloc[0])) < 1e-5
        metadata.append(json.loads((SEQ_OUT / f'{name}_training.json').read_text(encoding='utf-8')))
    selected = select_model(val.to_dict('records'))
    test['selected_by_validation'] = test.model == selected
    val.to_csv(SEQ_OUT / 'all_validation_metrics.csv', index=False)
    test.to_csv(SEQ_OUT / 'all_test_metrics.csv', index=False)
    pred.to_csv(SEQ_OUT / 'all_test_predictions.csv', index=False)
    (SEQ_OUT / 'all_selection.json').write_text(json.dumps({
        'selected_model': selected, 'selection_metric': 'validation MAE',
        'test_targets_identical': True, 'test_rows': len(pred),
        'scope': 'Exploratory extension after initial test results were already seen; no test-driven retuning.'}, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(test.model.map(model_name), test.MAE, color=['#94a3b8','#38bdf8','#14b8a6','#a78bfa','#f59e0b'])
    for i, v in enumerate(test.MAE):
        ax.text(i, v+2, f'{v:.1f}', ha='center')
    ax.set(ylabel='평균 절대 오차(대)', title='5개 모델 비교: 같은 시험 시각 1,314개')
    ax.tick_params(axis='x', labelsize=9)
    fig.tight_layout(); fig.savefig(SEQ_OUT / 'five_model_comparison.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,4))
    for name in ['RNN','Transformer']:
        curve = pd.read_csv(SEQ_OUT / f'{name}_learning_curve.csv')
        ax.plot(curve.epoch, curve.validation_MAE, label=model_name(name))
    ax.set(xlabel='전체 학습 자료를 반복한 횟수', ylabel='검증 평균 절대 오차(대)', title='반복 학습에 따른 검증 오차: 가장 낮은 시점의 모델 사용')
    ax.legend(); fig.tight_layout(); fig.savefig(SEQ_OUT / 'learning_curves.png', dpi=160); plt.close(fig)
    hours = pred.target_time.dt.hour
    hourly = pd.DataFrame({'hour': hours})
    for name in test.model:
        hourly[name] = (pred[name] - pred.target_count).abs()
    hourly.groupby('hour').mean().to_csv(SEQ_OUT / 'all_hourly_errors.csv')
    fig, ax = plt.subplots(figsize=(11,4))
    for name in test.model:
        ax.plot(range(24), hourly.groupby('hour')[name].mean(), label=model_name(name))
    ax.set(xlabel='예측 대상 시각(시)', ylabel='평균 절대 오차(대)', title='5개 모델의 시간대별 예측 오차')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(SEQ_OUT / 'all_hourly_errors.png', dpi=160); plt.close(fig)
    epoch_table = pd.DataFrame([{'model': m['model'], 'train_rows': m['samples']['train'],
                                'best_epoch': m['best_epoch'], 'epochs_run': m['epochs_run'],
                                'parameters': m['parameters']} for m in metadata])
    lines = ['# 순환 신경망(RNN)·트랜스포머 포함 5개 모델 비교',
             '## 실험 조건',
             '직전 대여량 기준·선형회귀·그래디언트 부스팅에 순환 신경망(RNN)과 트랜스포머를 추가했다. '
             '신경망은 최근 24시간을 순서대로 입력하고 다음 1시간을 예측한다. 기존 세 모델은 현재 한 시간 입력이므로 '
             '구조와 입력 정보량이 모두 다르다. 이 비교만으로 신경망 구조 자체의 우열을 단정할 수 없다.',
             '검증·시험 날짜와 정답은 기존 실험과 완전히 동일한지 코드에서 검사했다. 검증 1,314개, 시험 1,314개다. '
             '신경망 학습은 첫 23개를 제외한 6,108개이며 기존 학습은 6,131개다. '
             '검증/시험 구간의 시작에도 이전 구간의 이미 관측한 과거 23시간을 이용할 수 있다.',
             '현재 시간 t까지의 관측만 입력한다. 학습과 정답 표준화는 학습 자료로만 적합하고, 검증 MAE로 최적 학습 반복 횟수를 선택한다. '
             '시드 42, CPU, 최대 40회 반복, 개선 없는 8회 반복 후 종료. 설정은 SEQUENCE_EXPERIMENT.md 참조.',
             markdown_table(epoch_table),
             '## 검증과 선택', markdown_table(val),
             f'5개 후보 중 검증 MAE가 가장 낮아 **{model_name(selected)}** 모델을 선택한다. 최종 시험 점수로 선택을 바꾸지 않는다.',
             '## 최종 시험 결과', markdown_table(test),
             '![5개 모델 MAE](five_model_comparison.png)',
             '![검증 학습 곡선](learning_curves.png)',
             '![시간대별 오차](all_hourly_errors.png)',
             '## 결과 해석',
             '더 복잡한 모델이 언제나 더 잘 맞는 것은 아니다. 검증과 시험의 순위가 달라질 수 있으며, '
             '시험 MAE가 가장 낮다는 이유로 검증에서 선택한 모델을 교체하면 안 된다. 이번 실험은 단일 설정·단일 시드 비교다.',
             '## 신경망 실패 사례']
    for name in ['RNN', 'Transformer']:
        tmp = pred.copy()
        tmp['absolute_error'] = (tmp[name] - tmp.target_count).abs()
        worst = tmp.nlargest(5, 'absolute_error')[['input_time','target_time','current_count','target_count', name,'absolute_error','current_functioning','Rainfall(mm)']]
        worst.to_csv(SEQ_OUT / f'worst_{name}.csv', index=False)
        lines += [f'### {model_name(name)}: 오차 상위 5개', markdown_table(worst)]
        row = worst.iloc[0]
        lines += [f"최대 오차 시각은 {row.target_time}: 직전 {row.current_count:.0f}대, 실제 {row.target_count:.0f}대, 예측 {row[name]:.1f}대다. "
                  '과거 24시간을 넣어도 급격한 변화나 과거에 드물었던 상황을 모두 예측할 수는 없다. 원인은 운영·기상 기록과 함께 검토해야 한다.']
    lines += ['## 해석의 한계',
              '기존 시험 점수를 확인한 뒤 요청에 따라 신경망 실험을 추가했다. 확장 실험은 탐색적 비교이며 완전히 새로 보지 않은 시험이라고 주장하지 않는다. '
              '시험 점수를 본 뒤 설정을 재조정하지 않았다. 다른 연도나 독립 기간으로 추가 검증해야 한다. '
              '계절 분포 차이, 한 해 자료, 매시간 집계가 제때 제공된다는 가정도 그대로 남는다.',
              '## 재현', '`python src/run_forecast.py`로 5개 모델 전체를 실행한다. '
              '이미 학습한 결과의 분석만 갱신하려면 `python src/compare_sequences.py`를 실행한다. '
              '학습 가중치, 학습 반복 횟수별 곡선, 설정과 라이브러리 버전, 모든 예측을 같은 폴더에 보관한다.']
    (SEQ_OUT / 'COMPARISON.md').write_text('\n\n'.join(lines)+'\n', encoding='utf-8')
    print(test.to_string(index=False))
    print('Selected by validation:', selected)


if __name__ == '__main__':
    main()
