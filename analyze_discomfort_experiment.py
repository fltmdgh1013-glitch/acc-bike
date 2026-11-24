"""불쾌지수 추가 전후 수치·계절 한계·오판을 한국어 독립 보고서로 만든다."""
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from reportlab.platypus import SimpleDocTemplate,PageBreak,Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from create_condition_reports import ROOT,BASE,NAMES,MODELS,p,table,setup_fonts,footer,metrics

DEST=BASE/'discomfort'
PDF=ROOT/'output/pdf/조건별보고서/자전거대여량예측_보고서_불쾌지수만_추가.pdf'
DI='현재_불쾌지수'
BANDS=[('68 미만','di_below68'),('68 이상 75 미만','di_68_75'),('75 이상 80 미만','di_75_80'),('80 이상','di_80_plus')]


def markdown(df):
    def fmt(v):return '-' if pd.isna(v) else f'{v:.2f}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+
                     ['| '+' | '.join(map(fmt,row))+' |' for row in df.itertuples(index=False,name=None)])


def main(render=False):
    all_metrics=pd.read_csv(DEST/'metrics.csv');comparison=pd.read_csv(DEST/'comparison.csv')
    timing=pd.read_csv(DEST/'timing_metrics.csv');groups=pd.read_csv(DEST/'group_metrics.csv')
    selected=json.loads((DEST/'selection.json').read_text(encoding='utf-8'))
    source=json.loads((ROOT/'data/external/discomfort/SOURCE.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((ROOT/source['계산파일']).read_bytes()).hexdigest()==source['계산파일sha256']
    before=pd.read_csv(DEST/'before_test_predictions.csv',parse_dates=['input_time','target_time'])
    after=pd.read_csv(DEST/'after_test_predictions.csv',parse_dates=['input_time','target_time'])
    reference=pd.read_csv(BASE/'sequence/all_test_predictions.csv',parse_dates=['input_time','target_time'])
    for frame in [before,after]:
        pd.testing.assert_frame_equal(frame[['input_time','target_time','target_count','current_count']],
                                      reference[['input_time','target_time','target_count','current_count']],check_dtype=False)
    primary=(after.target_time.dt.day_of_week<5)&after.target_time.dt.hour.isin([7,8,9,17,18,19])
    for m in MODELS:
        assert np.allclose(before[m],reference[m],atol=1e-4,rtol=0)
        expected=all_metrics[(all_metrics.split=='test')&(all_metrics.variant=='after')&(all_metrics.model==m)].iloc[0]
        assert abs(metrics(after.target_count,after[m])['MAE']-expected.MAE)<1e-4
        saved=timing[(timing.split=='test')&(timing.variant=='after')&(timing.model==m)&(timing.group=='weekday_commute')].iloc[0]
        calc=metrics(after.loc[primary,'target_count'],after.loc[primary,m])
        for c in ['TP','FP','FN','TN','precision','recall','accuracy']:assert abs(calc[c]-saved[c])<1e-9
    inputs=pd.read_csv(DEST/'input_examples.csv',parse_dates=['input_time','target_time'])
    distribution=[]
    for label,group in BANDS:
        row={'현재 불쾌지수':label}
        for split in ['train','validation','test']:
            v=inputs.loc[inputs.split==split,DI]
            mask={'di_below68':v<68,'di_68_75':(v>=68)&(v<75),'di_75_80':(v>=75)&(v<80),'di_80_plus':v>=80}[group]
            row[{'train':'학습','validation':'검증','test':'시험'}[split]]=int(mask.sum())
        distribution.append(row)
    distribution=pd.DataFrame(distribution)
    assert distribution[['학습','검증','시험']].sum().tolist()==[6131,1314,1314]
    distribution.to_csv(DEST/'불쾌지수_구간분포_한국어.csv',index=False,encoding='utf-8-sig')
    summary=comparison.rename(columns={'model':'모델','validation_before_MAE':'검증_기본','validation_after_MAE':'검증_추가후',
                                       'validation_delta_MAE':'검증_오차증가','test_before_MAE':'시험_기본','test_after_MAE':'시험_추가후','test_delta_MAE':'시험_오차증가'}).copy()
    summary['모델']=summary['모델'].map(NAMES);summary.to_csv(DEST/'비교표_한국어.csv',index=False,encoding='utf-8-sig')
    t=timing[(timing.split=='test')&(timing.group=='weekday_commute')].copy()
    t['model']=t.model.map(NAMES);t['variant']=t.variant.map({'before':'기본','after':'불쾌지수 추가'})
    t.rename(columns={'model':'모델','variant':'조건','recall':'재현율','precision':'정밀도','accuracy':'정확도'}).to_csv(
        DEST/'출퇴근_고수요_한국어.csv',index=False,encoding='utf-8-sig')
    human_groups=groups.copy();human_groups['model']=human_groups.model.map(NAMES)
    human_groups.to_csv(DEST/'구간별오차_한국어.csv',index=False,encoding='utf-8-sig')
    label='불쾌지수 추가' if selected['variant']=='after' else '기본 정보만'
    chosen=all_metrics[(all_metrics.split=='test')&(all_metrics.variant==selected['variant'])&(all_metrics.model==selected['model'])].iloc[0]
    choice=(f'이번 두 조건의 검증 선택은 {label} 조건의 {NAMES[selected["model"]]} 모델이다. 검증 MAE {selected["MAE"]:.2f}대, '
            f'선택한 조건의 시험 MAE는 {chosen.MAE:.2f}대다. 시험 예측 전에 선택을 기록했으며 시험 점수로 바꾸지 않았다.')
    hot=int((after[DI]>=68).sum());maximum=float(after[DI].max())
    season=(f'최종 시험은 2018년 10월 7일~11월 30일이다. 현재 불쾌지수 68 이상은 {hot}/1,314개이고 최댓값은 {maximum:.2f}이다. '
            '따라서 이 시험 결과로 여름철 무더위 구간의 예측 효과를 검증했다고 주장할 수 없다.')
    intro=('기본 입력의 기온·습도는 그대로 두고 현재 불쾌지수 한 열만 추가했다. '
           '외부 달력·미세먼지·일출일몰은 추가하지 않았다. 새 외부 관측이 아니라 기존 정보의 파생 특성 실험이다.')
    formula='DI = 1.8 × T - 0.55 × (1 - H/100) × (1.8 × T - 26) + 32'
    limits=('시드 42·기존 모델 설정·같은 시간순 분할로 비교했다. 학습 6,131개(신경망 6,108개), 검증·시험은 각 1,314개다. '
            '학습 전처리는 학습 자료로만 적합했다. 기존 시험을 확인한 뒤 확장한 단일 시드 비교이므로 독립된 새 기간 검증이 아니다.')
    remarks=('기온·습도에 이미 있는 정보를 조합한 것이므로 새로운 관측 정보를 얻은 것은 아니다. 산식에는 기온과 습도의 곱이 있어 '
             '선형 모델에 관계를 다른 형태로 제공할 수 있다. 모델마다 효과가 같다고 보장되지 않는다. 신경망 입력 차원이 늘면 계수 수도 달라진다.')
    improved=[f'{NAMES[r.model]}({r.test_before_MAE:.2f} → {r.test_after_MAE:.2f}대)' for r in comparison.itertuples()
              if r.model!='Persistence' and r.test_delta_MAE<0]
    finding='이번 실행에서 시험 MAE가 줄어든 학습 모델: '+(', '.join(improved) if improved else '없음')+'.'
    lines=['# 자전거 대여량 예측 보고서: 불쾌지수만 추가',intro,'## 계산과 시각',formula,
           'T는 현재 기온 °C, H는 현재 상대습도 %다. input_time으로 연결했고 다음 시간의 실제 기온·습도는 사용하지 않았다. '
           '겨울 값을 임의로 자르거나 제거하지 않았다. 공식 발표 지수를 복원한 것이 아니며 구별 기상 분포를 추가 수집한 것도 아니다.',
           '## 전체 오차',markdown(summary),'오차증가=추가 후-기본. 음수일 때 개선.',finding,choice,
           '## 평일 출퇴근 고수요',markdown(t[['variant','model','n','TP','FP','FN','TN','recall','precision','accuracy']]),
           '고수요 400대 이상. 월~금 07·08·09·17·18·19시 240개, 공휴일 포함. 비율은 0~1. 실제 고수요 215개이므로 오차와 오판 수도 함께 평가한다.',
           '## 불쾌지수 구간 분포',markdown(distribution),season,
           '## 한계',limits,remarks,'2011년 부산지방기상청 설명자료의 산식은 여름철 무더위 기준이며 복사·바람을 반영하지 않는다. 서울 원본의 기온·습도에 산식만 적용했다. 겨울 값을 여름 체감 단계로 해석하지 않는다.',
           '## 출처와 재현','산식: data/external/weather_candidates/kma_discomfort_index_20110714.pdf, PDF 2쪽. 원본·계산 CSV와 해시는 data/external/discomfort/SOURCE.json 참조.',
           'python src/prepare_discomfort.py\npython src/run_discomfort_experiment.py\npython src/analyze_discomfort_experiment.py --render',
           '구현·분석·문서에 AI 도구 활용. 작성일 2026-11-20.']
    (DEST/'REPORT.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    setup_fonts()
    story=[p('인공지능과머신러닝 텀프로젝트','small'),p('자전거 대여량 예측 보고서\n불쾌지수만 추가','title'),p(intro),
           p('1 입력과 계산식','h'),p(formula,'small'),p('T는 현재 기온(°C), H는 현재 상대습도(%)다. '
             '현재 input_time의 값으로 계산해 다음 시간 대여량을 예측한다. 원본과 같은 8,760시간을 보관했고 예측 사례는 8,759개다.'),
           p('2 통제 조건','h'),p(limits),p('3 모델별 평균 절대 오차 MAE','h')]
    rows=[[NAMES[r.model],f'{r.validation_before_MAE:.2f}',f'{r.validation_after_MAE:.2f}',f'{r.test_before_MAE:.2f}',f'{r.test_after_MAE:.2f}'] for r in comparison.itertuples()]
    story += [table(['모델','검증\n기본','검증\n추가 후','시험\n기본','시험\n추가 후'],rows,[48,31,31,31,31]),Spacer(1,6),
              p('MAE는 실제 대여량과 예측값 차이의 크기를 평균한 값이다. 단위는 대이며 낮을수록 좋다.','small'),p(finding),p('검증으로 선택한 조건','h'),p(choice),
              p('주의: 시험 기간의 더위 표본','h'),p(season),PageBreak(),
              p('불쾌지수만 추가','small'),p('주요 시간대의 성능','title'),p('4 평일 출퇴근 고수요 판단','h'),
              p('예측 대상이 월~금 07·08·09·17·18·19시인 240개다. 공휴일도 포함한다. 400대 이상을 고수요로 정했다. '
                '화살표는 기본 → 불쾌지수 추가 조건을 나타낸다.')]
    indexed=timing[(timing.split=='test')&(timing.group=='weekday_commute')].set_index(['variant','model'])
    rows=[]
    for m in MODELS:
        a,b=indexed.loc[('before',m)],indexed.loc[('after',m)]
        rows.append([NAMES[m]]+[f'{a[c]*100:.1f} → {b[c]*100:.1f}' for c in ['recall','precision','accuracy']]+[f'{int(a.FN)} → {int(b.FN)}',f'{int(a.FP)} → {int(b.FP)}'])
    story += [table(['모델','재현율 %','정밀도 %','정확도 %','놓친\n고수요','잘못 울린\n경보'],rows,[41,28,27,27,22,27]),Spacer(1,6),
              p('재현율은 실제 고수요를 잡은 비율, 정밀도는 고수요 경보가 맞은 비율, 정확도는 전체 판단이 맞은 비율이다. '
                '실제 고수요가 215/240개라 항상 고수요라고 해도 재현율은 100%다.','small'),p('5 시간대별 MAE: 기본 → 추가 후','h')]
    g=groups[groups.split=='test'].set_index(['variant','model','group']);rows=[]
    for m in MODELS:
        row=[NAMES[m]]
        for group in ['morning_all_days','evening_all_days','commute_all_days','non_commute_all_days']:
            a,b=g.loc[('before',m,group)],g.loc[('after',m,group)]
            row.append(f'{a.MAE:.1f} → {b.MAE:.1f}')
        rows.append(row)
    story += [table(['모델','출근','퇴근','출퇴근 합계','출퇴근 외'],rows,[48,31,31,31,31]),Spacer(1,7),
              p('위 표는 모든 요일이다. 출근·퇴근은 각 165개, 출퇴근 합계 330개, 그 외 984개다. 낮은 MAE와 높은 재현율은 서로 다른 평가다.','small'),
              p('같은 정보를 다른 형태로 제공','h'),p(remarks),PageBreak(),
              p('불쾌지수만 추가','small'),p('계절 분포와 오류 사례','title'),p('6 현재 불쾌지수 구간별 사례 수','h'),
              table(distribution.columns.tolist(),distribution.astype(str).values.tolist(),[58,38,38,38]),Spacer(1,6),p(season),
              p('7 추가 후 모델의 큰 오류 사례','h'),p('검증으로 선택한 모델 종류의 불쾌지수 추가 버전에서 절대오차 상위 4개를 확인했다. 시험으로 모델을 재선택한 것이 아니다.','small')]
    model=selected['model'];worst=after.assign(error=(after[model]-after.target_count).abs()).nlargest(4,'error')
    rows=[[r.target_time.strftime('%Y-%m-%d %H시'),str(int(r.target_count)),f'{r[model]:.1f}',f'{r.error:.1f}',f'{r[DI]:.1f}'] for _,r in worst.iterrows()]
    story += [table(['예측 대상 시각','실제','예측','절대 오차','현재 DI'],rows,[60,28,28,28,28]),Spacer(1,6),
              p('8 해석의 한계와 재현','h'),p('공식 과거 설명의 여름철 산식을 적용한 것이며, 현재 공식 예보 서비스의 지수를 복원한 것은 아니다. '
                '복사·바람을 반영하지 않고 구별 기상 분포도 추가하지 않았다. 겨울 값은 그대로 보존하되 여름 체감 단계로 해석하지 않는다. '
                '별도의 여름 시험 기간과 여러 시드에서 재검증해야 한다.'),
              p('산식 근거: 부산지방기상청 2011-07-14 자료 PDF 2쪽. 보관 원문과 계산 CSV의 해시는 data/external/discomfort/SOURCE.json 참조.','small'),
              p('계산: python src/prepare_discomfort.py\n학습: python src/run_discomfort_experiment.py\n보고서: python src/analyze_discomfort_experiment.py --render','small'),
              p('상세 수치·입력·가중치: reports/forecast/discomfort/\n구현·분석·문서에 AI 도구 활용. 작성일 2026년 11월 20일.','small')]
    PDF.parent.mkdir(parents=True,exist_ok=True)
    SimpleDocTemplate(str(PDF),pagesize=A4,leftMargin=19*mm,rightMargin=19*mm,topMargin=17*mm,bottomMargin=22*mm,
                      title='자전거 대여량 예측 보고서 불쾌지수만 추가',author='자전거 대여량 예측 프로젝트').build(story,onFirstPage=footer,onLaterPages=footer)
    pages=None
    if render:
        import pypdfium2 as pdfium
        pdf=pdfium.PdfDocument(str(PDF));pages=len(pdf)
        qa=ROOT/'tmp/pdfs/discomfort';qa.mkdir(parents=True,exist_ok=True)
        for i,page in enumerate(pdf):page.render(scale=1.5).to_pil().save(qa/f'page-{i+1}.png')
        assert pages==3,pages
    manifest={'file':str(PDF.relative_to(ROOT)),'sha256':hashlib.sha256(PDF.read_bytes()).hexdigest(),'pages':pages,
              'test_rows':len(after),'test_current_di_ge68':hot,'test_current_di_max':maximum,'selection':selected}
    (DEST/'report_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary.to_string(index=False));print(choice);print(season);print(PDF)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--render',action='store_true');main(parser.parse_args().render)
