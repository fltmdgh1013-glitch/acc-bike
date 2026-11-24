"""실제 미세먼지 추가 실험의 결과와 한계를 한국어로 보고한다."""
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from forecast_data import ROOT,OUT
from run_air_quality_experiment import DEST
from run_external_experiment import MODEL_LIST
from korean_labels import model_name,export_korean_csvs
from analyze_forecast import markdown_table
from evaluate_timing import percent_table


def label_conditions(frame, include_calendar=True):
    frame=frame.copy()
    if 'variant' in frame:
        labels=({'before':'기존 정보+달력','after':'기존 정보+달력+미세먼지'} if include_calendar else
                {'before':'기존 정보만','after':'기존 정보+미세먼지'})
        frame['variant']=frame.variant.map(labels)
    return frame


def four_condition_table(dest, before):
    original=pd.read_csv(OUT/'sequence/all_test_metrics.csv').set_index('model')
    calendar=pd.read_csv(OUT/'external/metrics.csv')
    calendar=calendar.loc[(calendar.split=='test')&(calendar.variant=='after')].set_index('model')
    both=pd.read_csv(OUT/'air_quality/metrics.csv')
    both=both.loc[(both.split=='test')&(both.variant=='after')].set_index('model')
    pm=pd.read_csv(dest/'metrics.csv')
    pm=pm.loc[(pm.split=='test')&(pm.variant=='after')].set_index('model')
    for path in [OUT/'sequence/all_test_predictions.csv',OUT/'external/after_test_predictions.csv',OUT/'air_quality/after_test_predictions.csv']:
        frame=pd.read_csv(path,parse_dates=['input_time','target_time'])
        pd.testing.assert_frame_equal(before[['input_time','target_time','target_count']],frame[['input_time','target_time','target_count']])
    result=pd.DataFrame([{'model':name,'원래 정보만(대)':original.loc[name,'MAE'],
                          '외부 달력만 추가(대)':calendar.loc[name,'MAE'],
                          '미세먼지만 추가(대)':pm.loc[name,'MAE'],
                          '달력과 미세먼지 모두 추가(대)':both.loc[name,'MAE']}
                         for name in MODEL_LIST])
    result.to_csv(dest/'four_conditions_comparison.csv',index=False)
    return result


def main(include_calendar=True):
    DEST=OUT/('air_quality' if include_calendar else 'air_quality_only')
    before_label='기존 정보+달력' if include_calendar else '기존 정보만'
    after_label='기존 정보+달력+미세먼지' if include_calendar else '기존 정보+미세먼지'
    summary=pd.read_csv(DEST/'comparison.csv')
    metrics=pd.read_csv(DEST/'metrics.csv')
    groups=pd.read_csv(DEST/'group_metrics.csv')
    timing=pd.read_csv(DEST/'timing_metrics.csv')
    selected=json.loads((DEST/'selection.json').read_text(encoding='utf-8'))
    protocol=json.loads((DEST/'protocol.json').read_text(encoding='utf-8'))
    source=json.loads((ROOT/'data/external/air_quality/미세먼지_출처와처리.json').read_text(encoding='utf-8'))
    before=pd.read_csv(DEST/'before_test_predictions.csv',parse_dates=['input_time','target_time'])
    after=pd.read_csv(DEST/'after_test_predictions.csv',parse_dates=['input_time','target_time'])
    pd.testing.assert_frame_equal(before[['input_time','target_time','target_count']],after[['input_time','target_time','target_count']])
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    x=np.arange(len(summary))
    for ax,split,title in zip(axes,['validation','test'],['검증','최종 시험']):
        title='검증' if split=='validation' else '최종 시험'
        ax.bar(x-.18,summary[f'{split}_before_MAE'],.36,label=before_label,color='#94a3b8')
        ax.bar(x+.18,summary[f'{split}_after_MAE'],.36,label=after_label,color='#0d9488')
        ax.set_xticks(x,summary.model.map(model_name),rotation=25,ha='right',fontsize=9)
        ax.set(ylabel='평균 절대 오차(대)',title=title+' 자료의 대여량 오차')
        ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(DEST/'pm_before_after_mae.png',dpi=160,bbox_inches='tight');plt.close(fig)
    commute=groups.loc[(groups.split=='test')&(groups.group=='commute_all_days')]
    fig,ax=plt.subplots(figsize=(10,5))
    for offset,variant,label,color in [(-.18,'before',before_label,'#94a3b8'),(.18,'after',after_label,'#0d9488')]:
        part=commute.loc[commute.variant==variant].set_index('model').reindex(MODEL_LIST)
        ax.bar(x+offset,part.MAE,.36,label=label,color=color)
    ax.set_xticks(x,[model_name(m) for m in MODEL_LIST],rotation=20,ha='right')
    ax.set(ylabel='평균 절대 오차(대)',title='미세먼지 추가 전후 출퇴근 시간 오차(모든 요일 330개)')
    ax.legend();fig.tight_layout();fig.savefig(DEST/'pm_commute_mae.png',dpi=160,bbox_inches='tight');plt.close(fig)
    selected_test=metrics.loc[(metrics.split=='test')&(metrics.variant==selected['variant'])&(metrics.model==selected['model'])].iloc[0]
    lines=['# 미세먼지·초미세먼지 추가 전후 실험'+(' (달력 포함)' if include_calendar else ' (외부 달력 제외)'),
           '## 이번에 추가한 자료',
           '**이번 실험에는 실제 PM10(미세먼지)과 PM2.5(초미세먼지)를 넣었다.** '
           +('기존 정보와 외부 달력 정보를 모두 포함한 조건을 기준으로 비교한다.' if include_calendar else
             '외부 공휴일·연휴 자료를 빼고 원래 입력에 미세먼지만 추가했다. 원본에 이미 있는 날짜·휴일 입력은 기존 조건 그대로다.'),
           '출처는 [에어코리아 최종확정 측정자료](https://www.airkorea.or.kr/web/last_amb_hour_data)다. '
           '전국 2017·2018년 공식 원본 ZIP에서 2017년 11월~2018년 11월의 서울 도시대기 측정소만 추출했다. '
           '2017년 11월은 과제 첫 시간의 지연 입력을 만들기 위해 확보했다. 도로변 측정소는 섞지 않았다.',
           f"서울 측정소 {source['범위']['측정소수']}개의 값을 시간별로 평균했다. PM10·PM2.5의 단위는 ㎍/㎥다. "
           '각 물질별로 -999 등 음수·비수치·빈칸을 제외하고 유효한 측정소만 평균했다. 시간별 유효 측정소 수는 원본 집계 CSV에 함께 보관했다.',
           '사용자용 시간별 자료는 저장소 최상위 **외부자료_미세먼지.csv**다. 기존 **외부자료.csv**는 날짜별 달력 자료로 유지한다. '
           '자전거 입력 시각 8,760개 모두 미세먼지 값이 연결되는지 확인했다.',
           '## 미래 정보를 피한 연결',
           '입력 시각이 오전 9시이면, 종료 표기가 오전 8시인 측정값을 사용해 다음 시간 대여량을 예측한다. '
           '다음 시간의 실제 미세먼지나 하루 전체 평균을 사용하지 않는다. 원본 24시 표기는 다음 날짜 0시로 해석한다.',
           '실제 관측값이 일부 빠졌을 때는 학습 구간의 중앙값으로만 대체하고 결측 표시를 함께 넣도록 구현했다. '
           '이번 과제 구간에서는 서울 평균값에 결측이 없어 이 대체는 실제로 적용되지 않았다. 유효 측정소 수는 시간마다 다르다.',
           '## 비교 조건',
           '- 추가 전: 현재 대여량·날씨·날짜 정보'+(' + 외부 공휴일·연휴 정보' if include_calendar else '')+'.\n'
           +'- 추가 후: 위 정보 + 한 시간 지연 PM10·PM2.5 + 두 물질의 결측 표시.\n'
           +'- 같은 날짜 분할: 학습 6,131개(신경망 6,108개), 검증 1,314개, 시험 1,314개.\n'
           '- 모델 구조·설정·시드 42·최대 40회 학습 등 기존 조건 고정. 전처리는 학습 자료로만 적합.\n'
           '- 검증 MAE 최소 시점의 신경망 가중치로 복원. 두 조건 내 모델 선택은 시험 평가 전에 고정.',
           '신경망 입력 차원이 늘어 계수 수도 늘어난다. 결측 표시 두 열은 이번 구간에서 모두 0이다. '
           '단일 시드 실험이므로 개선 또는 악화가 다른 초기화에서도 반복되는지 별도로 확인해야 한다.',
           '## 전체 대여량 오차',
           f'아래 표의 **추가 전은 {before_label}**, **추가 후는 {after_label}** 조건이다. '
           '오차 변화는 추가 후-추가 전으로, 음수는 개선이고 양수는 악화다.',
           markdown_table(summary),
           '![미세먼지 추가 전후 오차](pm_before_after_mae.png)',
           '## 검증으로 선택한 조건',
           f"이번 두 조건 내 검증 선택은 **{after_label if selected['variant']=='after' else before_label} / {model_name(selected['model'])}**다. "
           f"검증 MAE {selected['MAE']:.2f}대, 이 조건의 시험 MAE {selected_test.MAE:.2f}대다. 시험 점수로 선택을 바꾸지 않았다. "
           '다른 외부정보 실험의 모든 후보까지 합쳐 새로 전역 선택한 것은 아니다.',
           '## 주요 시간대 결과',
           '![출퇴근 오차](pm_commute_mae.png)',
           markdown_table(label_conditions(groups.loc[(groups.split=='test')&groups.group.isin(['commute_all_days','weekday_commute']),
                                                      ['variant','model','group','rows','MAE']],include_calendar)),
           '## 평일 출퇴근 고수요 판단',
           '다음 시간 400대 이상을 고수요로 고정한다. 새로운 경계를 시험 결과에 맞춰 조정하지 않았다.',
           percent_table(label_conditions(timing.loc[(timing.split=='test')&(timing.group=='weekday_commute'),
                                                      ['variant','model','n','precision','recall','accuracy','TP','FP','FN','TN']],include_calendar)),
           '평일 출퇴근은 고수요 215개/240개로 고수요가 많다. 재현율만 높다는 이유로 더 정확한 대여량 예측이라고 판단하지 않는다.',
           '## 모델별 변화']
    for row in summary.itertuples():
        direction='감소' if row.test_delta_MAE<0 else ('증가' if row.test_delta_MAE>0 else '변화 없음')
        lines += [f'- {model_name(row.model)}: 시험 MAE {row.test_before_MAE:.2f} → {row.test_after_MAE:.2f}대, '
                  f'{abs(row.test_delta_MAE):.2f}대 {direction}. 검증 변화 {row.validation_delta_MAE:+.2f}대.']
    if not include_calendar:
        lines += ['## 네 가지 입력 조건의 시험 오차',
                  '아래는 같은 시험 시각·정답을 사용했는지 확인한 뒤 기존 실험까지 모은 비교다. '
                  '외부 달력 없음은 원본 날짜·요일·휴일 입력까지 없앴다는 뜻이 아니라 외부자료.csv의 8개 추가 특성을 쓰지 않았다는 뜻이다. '
                  '여러 차례 실험에서 관찰한 시험 결과이므로 이 표에서 가장 낮은 숫자로 모델을 다시 선택하지 않는다.',
                  markdown_table(four_condition_table(DEST,before))]
    learned=summary.loc[summary.model!='Persistence']
    if (learned.test_delta_MAE>0).all() and (learned.validation_delta_MAE>0).all():
        lines += ['이번 고정 설정·단일 시드 실험에서는 학습 모델 4개 모두 검증·시험 대여량 오차가 늘었다. '
                  f'따라서 현재 실험의 근거로는 {before_label} 조건을 유지하는 쪽이 타당하다. '
                  '미세먼지가 자전거 이용에 영향을 주지 않는다는 뜻은 아니다.']
    t=timing.loc[(timing.split=='test')&(timing.group=='weekday_commute')&(timing.model=='Transformer')].set_index('variant')
    lines += [f"트랜스포머의 평일 출퇴근 재현율은 {t.loc['before','recall']*100:.1f}% → {t.loc['after','recall']*100:.1f}%이고, "
              f"잘못 울린 경보는 {int(t.loc['before','FP'])}개 → {int(t.loc['after','FP'])}개였다. "
              '대여량 오차가 늘어도 특정 분류 지표는 좋아질 수 있으므로 두 결과를 구분해야 한다.']
    lines += ['## 예측이 크게 달라진 사례']
    for name in MODEL_LIST:
        if name=='Persistence':continue
        detail=after[['input_time','target_time','target_count','current_count','미세먼지_PM10_1시간지연','초미세먼지_PM25_1시간지연']].copy()
        detail['before_prediction']=before[name]
        detail['after_prediction']=after[name]
        detail['error_change']=(after[name]-after.target_count).abs()-(before[name]-before.target_count).abs()
        detail.nlargest(5,'error_change').to_csv(DEST/f'{name}_most_worsened.csv',index=False)
        detail.nsmallest(5,'error_change').to_csv(DEST/f'{name}_most_improved.csv',index=False)
        lines += [f'### {model_name(name)}: 개선 사례',markdown_table(detail.nsmallest(3,'error_change')),
                  f'### {model_name(name)}: 악화 사례',markdown_table(detail.nlargest(3,'error_change'))]
    lines += ['## 해석의 한계',
              '**최종확정 자료는 당시 실시간 게시값과 다를 수 있다.** 한 시간 지연을 가정했지만 실제 게시 지연을 복원한 것은 아니다. '
              '따라서 이번 결과를 실시간 운영에서 그대로 달성할 성능이라고 주장하지 않는다.',
              '서울 측정소의 단순 평균은 이용자별 실제 노출량과 다르다. 한 해 자료와 한 번의 시드 실행이며, '
              '기존 시험 점수를 이미 확인한 뒤 추가한 탐색적 비교다. 시험 결과를 보고 설정을 재조정하지 않았다. '
              '새 기간 검증·여러 시드 반복·특성별 제외 실험이 더 필요하다.',
              '## 재현',
              '커밋된 서울 측정소 추출본으로 `python src/prepare_air_quality.py --from-extracted`를 실행하면 '
              '전국 원본을 다시 내려받지 않고 시간별 평균을 재생성한다. '
              f'`python src/run_air_quality_experiment.py{"" if include_calendar else " --without-calendar"}`로 두 조건을 학습·평가하고 '
              f'`python src/analyze_air_quality_experiment.py{"" if include_calendar else " --without-calendar"}`로 이 보고서와 그래프를 생성한다.',
              '전국 원본부터 확인하려면 `python src/download_air_quality.py` 후 `python src/prepare_air_quality.py`를 실행한다. '
              '전국 ZIP은 용량이 커 로컬에만 보관하고 Git에서는 제외했다. 다운로드 주소·파일 크기·SHA256과 서울 추출본을 저장소에 보관했다.']
    (DEST/'REPORT.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    export_korean_csvs()
    print('미세먼지 실험 보고서 생성 완료')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--without-calendar',action='store_true',help='외부 달력을 제외한 미세먼지만의 효과 보고')
    args=parser.parse_args()
    main(include_calendar=not args.without_calendar)
