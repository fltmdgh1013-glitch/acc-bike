"""외부자료 추가 전후의 고정된 예측 결과를 한국어로 비교한다."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_external_experiment import DEST, MODEL_LIST
from external_features import EXTERNAL_COLUMNS
from korean_labels import model_name, export_korean_csvs
from analyze_forecast import markdown_table
from evaluate_timing import percent_table


def main():
    summary = pd.read_csv(DEST/'comparison.csv')
    metrics = pd.read_csv(DEST/'metrics.csv')
    groups = pd.read_csv(DEST/'group_metrics.csv')
    timing = pd.read_csv(DEST/'timing_metrics.csv')
    selected = json.loads((DEST/'selection.json').read_text(encoding='utf-8'))
    metadata = json.loads((DEST/'training_metadata.json').read_text(encoding='utf-8'))
    protocol = json.loads((DEST/'protocol.json').read_text(encoding='utf-8'))
    before = pd.read_csv(DEST/'before_test_predictions.csv',parse_dates=['input_time','target_time'])
    after = pd.read_csv(DEST/'after_test_predictions.csv',parse_dates=['input_time','target_time'])
    pd.testing.assert_frame_equal(before[['input_time','target_time','target_count']],after[['input_time','target_time','target_count']])
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    x = np.arange(len(summary))
    for ax,split,title in zip(axes,['validation','test'],['검증 자료','최종 시험 자료']):
        ax.bar(x-.18,summary[f'{split}_before_MAE'],.36,label='외부자료 추가 전',color='#94a3b8')
        ax.bar(x+.18,summary[f'{split}_after_MAE'],.36,label='외부자료 추가 후',color='#0d9488')
        ax.set_xticks(x,summary.model.map(model_name),rotation=25,ha='right',fontsize=9)
        ax.set(ylabel='평균 절대 오차(대)',title=title+'의 대여량 오차')
        ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(DEST/'before_after_mae.png',dpi=160,bbox_inches='tight'); plt.close(fig)
    commute = groups.loc[(groups.split=='test') & (groups.group=='commute_all_days')]
    fig, ax = plt.subplots(figsize=(10,5))
    for offset,variant,label,color in [(-.18,'before','추가 전','#94a3b8'),(.18,'after','추가 후','#0d9488')]:
        values = commute.loc[commute.variant==variant].set_index('model').reindex(MODEL_LIST)
        ax.bar(x+offset,values.MAE,.36,label=label,color=color)
    ax.set_xticks(x,[model_name(m) for m in MODEL_LIST],rotation=20,ha='right')
    ax.set(title='출퇴근 시간의 대여량 오차 비교(모든 요일 330개)',ylabel='평균 절대 오차(대)')
    ax.legend(); fig.tight_layout(); fig.savefig(DEST/'commute_mae.png',dpi=160,bbox_inches='tight'); plt.close(fig)
    details=[]
    for name in MODEL_LIST:
        prior=(before[name]-before.target_count).abs()
        extra=(after[name]-after.target_count).abs()
        detail=before[['input_time','target_time','target_count','current_count',*EXTERNAL_COLUMNS]].copy()
        detail['before_prediction']=before[name]
        detail['after_prediction']=after[name]
        detail['before_absolute_error']=prior
        detail['after_absolute_error']=extra
        detail['error_change']=extra-prior
        detail.nlargest(5,'error_change').to_csv(DEST/f'{name}_most_worsened.csv',index=False)
        detail.nsmallest(5,'error_change').to_csv(DEST/f'{name}_most_improved.csv',index=False)
        details.append((name,detail))
    training = pd.DataFrame([{'variant':m['variant'],'model':m['model'],'best_epoch':m['best_epoch'],
                             'epochs_run':m['epochs_run'],'parameters':m['parameters']} for m in metadata])
    selected_test=metrics.loc[(metrics.split=='test')&(metrics.variant==selected['variant'])&(metrics.model==selected['model'])].iloc[0]
    lines=['# 외부 공휴일·연휴 정보 추가 전후 실험',
           '## 실험 방법',
           '선형회귀·그래디언트 부스팅·순환 신경망(RNN)·트랜스포머를 외부자료 추가 전후로 각각 학습했다. '
           '직전 대여량 기준도 함께 비교했다. 모델 구조·설정·시드 42는 고정하고 입력 정보만 추가했다. '
           '기존 실험 파일은 유지하고 모든 새 결과는 이 폴더에 별도로 저장했다.',
           '추가 정보: '+', '.join(EXTERNAL_COLUMNS)+'. '
           '공휴일명은 학습하지 않았고, 요일도 기존 입력과 겹치므로 추가하지 않았다. '
           '특별공휴일 여부 등 일부 값은 기존 정보와 중복된다.',
           '현재가 12월 24일 23시이고 다음 날 0시를 예측한다면 12월 25일의 달력 정보를 붙인다. '
           '달력은 미리 알 수 있는 일정이며 미래 실제 대여량·기상·운영 상태는 추가 입력하지 않았다. '
           '원본 외부자료의 출처·범위·해석은 저장소의 data/external/외부자료_출처와설명.json에 있다.',
           f"학습 {protocol['split_rows']['train']:,}개(신경망 {protocol['neural_train_rows']:,}개), 검증 1,314개, 시험 1,314개. "
           '모든 조건은 동일한 검증·시험 시각과 정답을 사용하며 코드에서 일치 여부를 검사한다. '
           '전처리는 각 조건의 학습 자료로만 계산한다. 신경망은 최대 40회 학습하고 검증 MAE 최소 시점의 가중치로 복원한다.',
           markdown_table(training),
           '신경망은 입력 차원이 늘어나 입력층 계수 수도 늘었다. 따라서 동일한 학습 규칙·구조를 사용했지만 계수 수까지 동일한 비교는 아니다.',
           '## 전체 결과',
           '오차 변화는 **추가 후 - 추가 전**이다. 음수는 개선, 양수는 악화다. 단위는 대이며 낮을수록 좋다.',
           markdown_table(summary),
           '![외부자료 추가 전후 평균 오차](before_after_mae.png)',
           '## 검증으로 선택한 조건',
           f"검증 오차만으로 선택한 조건은 **{'외부자료 추가 후' if selected['variant']=='after' else '외부자료 추가 전'} / {model_name(selected['model'])}**다. "
           f"검증 MAE {selected['MAE']:.2f}대, 이 고정된 조건의 시험 MAE {selected_test.MAE:.2f}대다. "
           '이 선택은 이번 실험의 시험 예측 전에 저장했다. 시험 점수가 더 좋은 조건이 따로 있어도 시험 결과로 선택을 바꾸지 않는다.',
           '## 주요 시간대와 휴일 구간',
           '![출퇴근 시간 평균 오차 비교](commute_mae.png)',
           markdown_table(groups.loc[(groups.split=='test')&groups.group.isin(['commute_all_days','weekday_commute','calendar_rest_day','calendar_work_day']),
                                    ['variant','model','group','rows','MAE']]),
           '## 평일 출퇴근 고수요 탐지',
           '다음 시간 400대 이상을 고수요로 고정했다. 새 분류 모델을 학습하거나 임계값을 조정하지 않았다. '
           '재현율뿐 아니라 정밀도·정확도와 잘못 울린 경보·놓친 고수요 수도 함께 비교한다.',
           percent_table(timing.loc[(timing.split=='test')&(timing.group=='weekday_commute'),
                                   ['variant','model','n','precision','recall','accuracy','TP','FP','FN','TN']]),
           '이 구간은 고수요 215개/240개로 고수요가 매우 많다. 재현율 100%가 곧 대여량 오차가 작다는 뜻은 아니다.',
           '## 모델별 해석']
    transformer = summary.loc[summary.model=='Transformer'].iloc[0]
    commute_transformer = groups.loc[(groups.split=='test')&(groups.group=='commute_all_days')&(groups.model=='Transformer')].set_index('variant')
    timing_transformer = timing.loc[(timing.split=='test')&(timing.group=='weekday_commute')&(timing.model=='Transformer')].set_index('variant')
    reduction = (transformer.test_before_MAE-transformer.test_after_MAE)/transformer.test_before_MAE*100
    lines += [f'이번 트랜스포머 실행에서는 전체 시험 평균 절대 오차가 {reduction:.1f}% 감소했다. '
              f"출퇴근 오차도 {commute_transformer.loc['before','MAE']:.2f}대에서 {commute_transformer.loc['after','MAE']:.2f}대로 줄었다. "
              f"반면 평일 출퇴근의 잘못 울린 경보는 {int(timing_transformer.loc['before','FP'])}개에서 {int(timing_transformer.loc['after','FP'])}개로 늘었고, "
              f"정밀도는 {timing_transformer.loc['before','precision']*100:.1f}%에서 {timing_transformer.loc['after','precision']*100:.1f}%로 낮아졌다. "
              '대여량 오차 개선과 고수요 경보 성능 개선은 같은 결과가 아닐 수 있다.']
    for row in summary.itertuples():
        change=row.test_delta_MAE
        direction='줄었다' if change<0 else ('늘었다' if change>0 else '같았다')
        lines += [f"- {model_name(row.model)}: 시험 MAE {row.test_before_MAE:.2f} → {row.test_after_MAE:.2f}대. "
                  f"평균 오차가 {abs(change):.2f}대 {direction}. 검증 변화는 {row.validation_delta_MAE:+.2f}대다."]
    lines += ['## 크게 달라진 예측 사례',
              '오차 변화는 추가 후 절대오차에서 추가 전 절대오차를 뺀 값이다. 음수는 개선, 양수는 악화다. '
              '표의 날짜별 사례는 결과를 이해하기 위한 관찰이며 공휴일 정보가 인과적으로 개선을 만들었다는 증명은 아니다.']
    for name,detail in details:
        if name=='Persistence': continue
        case_columns = ['target_time','current_count','target_count','before_prediction',
                        'after_prediction','error_change','특별공휴일여부','연속휴일길이_일']
        lines += [f'### {model_name(name)}: 개선 3개',markdown_table(detail.nsmallest(3,'error_change')[case_columns]),
                  f'### {model_name(name)}: 악화 3개',markdown_table(detail.nlargest(3,'error_change')[case_columns])]
    lines += ['## 결론과 한계',
              '외부자료를 넣으면 무조건 좋아지는 것은 아니다. 이번 결과에는 기존 휴일 정보와의 중복, 한 해에 적은 공휴일 표본, '
              '학습·검증·시험의 계절 차이, 신경망의 한 번의 시드 실행 등이 영향을 줄 수 있다. '
              '정확한 원인을 구분하려면 추가 특성을 하나씩 빼는 실험이나 여러 과거 구간 검증이 필요하다. 이번에는 결과를 보고 재튜닝하지 않았다.',
              '이미 기존 시험 점수를 확인한 뒤 확장한 탐색적 비교다. 시험 자료를 완전히 처음 보는 독립 검증처럼 주장하지 않는다. '
              '효과를 확정하려면 다른 기간이나 연도 자료로 추가 확인해야 한다.',
              '## 재현',
              '`python src/run_external_experiment.py`로 추가 전후 모델을 모두 학습·평가하고, '
              '`python src/analyze_external_experiment.py`로 이 보고서와 그래프를 생성한다. '
              '분석 명령은 재학습하지 않는다. 외부자료가 없다면 `python src/prepare_external_calendar.py`로 먼저 재생성한다.']
    (DEST/'REPORT.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    export_korean_csvs()
    print('외부자료 비교 보고서 생성 완료')


if __name__ == '__main__':
    main()
