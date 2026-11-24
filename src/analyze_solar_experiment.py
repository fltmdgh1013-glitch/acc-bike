"""일출·일몰만 추가한 실험의 한국어 표와 독립 PDF를 저장된 예측에서 만든다."""
import argparse
import json
import numpy as np
import pandas as pd
from reportlab.platypus import SimpleDocTemplate,PageBreak,Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from create_condition_reports import ROOT,BASE,NAMES,MODELS,p,table,setup_fonts,footer,metrics

DEST=BASE/'solar'
PDF=ROOT/'output/pdf/조건별보고서/자전거대여량예측_보고서_일출일몰만_추가.pdf'
FEATURES=['일출_분','일몰_분','주간여부','일출일몰_가까운경계까지_분']


def markdown(frame):
    def fmt(v):
        return f'{v:.2f}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
                     ['| '+' | '.join(map(fmt,row))+' |' for row in frame.itertuples(index=False,name=None)])


def main(render=False):
    all_metrics=pd.read_csv(DEST/'metrics.csv')
    timing=pd.read_csv(DEST/'timing_metrics.csv')
    group=pd.read_csv(DEST/'group_metrics.csv')
    comparison=pd.read_csv(DEST/'comparison.csv')
    selection=json.loads((DEST/'selection.json').read_text(encoding='utf-8'))
    selected=selection['model'];variant=selection['variant']
    before=pd.read_csv(DEST/'before_test_predictions.csv',parse_dates=['input_time','target_time'])
    after=pd.read_csv(DEST/'after_test_predictions.csv',parse_dates=['input_time','target_time'])
    reference=pd.read_csv(BASE/'sequence/all_test_predictions.csv',parse_dates=['input_time','target_time'])
    for frame in [before,after]:
        pd.testing.assert_frame_equal(reference[['input_time','target_time','target_count','current_count']],
                                      frame[['input_time','target_time','target_count','current_count']],check_dtype=False)
    for m in MODELS:
        assert np.allclose(before[m],reference[m],atol=1e-4,rtol=0)
        saved=all_metrics[(all_metrics.split=='test')&(all_metrics.variant=='after')&(all_metrics.model==m)].iloc[0]
        assert abs(metrics(after.target_count,after[m])['MAE']-saved.MAE)<1e-4
    counts={label:int(mask.sum()) for label,mask in {
        '낮':after['주간여부']==1,'밤':after['주간여부']==0,
        '일출일몰1시간이내':after[FEATURES[-1]]<=60}.items()}
    labels={'before':'기본 정보만','after':'일출·일몰 추가'}
    condition=labels[variant]
    intro=('기본 입력에 일출·일몰 정보만 추가한 비교다. 외부 달력 8개 특성과 미세먼지는 추가하지 않았다. '
           '원본에 이미 있는 시각·계절·휴일·현재 일사량은 두 조건 모두 유지했다.')
    method=('Open-Meteo 서울 대표 좌표의 2017-12-01~2018-11-30 천문 계산 시각 365일을 KST 날짜로 연결했다. '
            '예측 대상 정시 순간이 일출 이상·일몰 미만이면 낮이다. 시간 전체의 실제 밝기나 구름 관측이 아니다.')
    inputs=('추가 특성은 자정부터 일출까지의 분, 자정부터 일몰까지의 분, 예측 대상 시각의 주간 여부, '
            '그 시각에서 더 가까운 일출 또는 일몰까지의 절대 시간 차(분)다. 이 네 값은 날짜·위치·예측 시각으로 미리 알 수 있다.')
    limits=('한 번의 시간순 분할과 시드 42에서의 탐색적 비교다. 기본 시험 결과를 확인한 뒤 추가한 조건이므로 '
            '독립적인 최종 성능 검증으로 해석하지 않는다. 모델 설정은 고정했고 시험 점수로 재조정하지 않았다. '
            '신경망은 입력 차원이 늘며 계수 수도 달라진다. 개선이 다른 초기화와 기간에서도 유지되는지 확인해야 한다.')
    summary=comparison.copy();summary['model']=summary['model'].map(NAMES)
    summary=summary.rename(columns={'model':'모델','validation_before_MAE':'검증_기본','validation_after_MAE':'검증_일출일몰',
                                    'validation_delta_MAE':'검증_오차증가','test_before_MAE':'시험_기본','test_after_MAE':'시험_일출일몰','test_delta_MAE':'시험_오차증가'})
    summary.to_csv(DEST/'비교표_한국어.csv',index=False,encoding='utf-8-sig')
    test_timing=timing[(timing.split=='test')&(timing.group=='weekday_commute')].copy()
    test_timing['model']=test_timing['model'].map(NAMES);test_timing['variant']=test_timing['variant'].map(labels)
    test_timing[['variant','model','n','TP','FP','FN','TN','recall','precision','accuracy']].rename(
        columns={'variant':'조건','model':'모델','n':'사례수','recall':'재현율','precision':'정밀도','accuracy':'정확도'}
    ).to_csv(DEST/'출퇴근_고수요_한국어.csv',index=False,encoding='utf-8-sig')
    solar_groups=group[(group.split=='test')&group.group.isin(['solar_daylight','solar_dark','solar_near_boundary','commute_all_days'])].copy()
    solar_groups['model']=solar_groups.model.map(NAMES)
    solar_groups['variant']=solar_groups.variant.map(labels)
    solar_groups['group']=solar_groups.group.map({'solar_daylight':'낮','solar_dark':'밤','solar_near_boundary':'일출·일몰 1시간 이내','commute_all_days':'출퇴근(모든 요일)'})
    solar_groups.to_csv(DEST/'낮밤_시간대별오차_한국어.csv',index=False,encoding='utf-8-sig')
    selected_test=all_metrics[(all_metrics.split=='test')&(all_metrics.variant==variant)&(all_metrics.model==selected)].iloc[0]
    selection_note=(f'이번 두 조건 내 검증 선택: {condition} / {NAMES[selected]}. 검증 MAE {selection["MAE"]:.2f}대, '
                    f'해당 조건 시험 MAE {selected_test.MAE:.2f}대다. 검증으로 선택을 고정한 뒤 시험을 평가했다. '
                    '모든 과거 외부자료 실험을 합쳐 새 전역 모델을 선택한 결과는 아니다.')
    conclusion=('이번 고정 설정에서는 선형회귀의 전체 오차만 감소했고 부스팅·RNN·트랜스포머는 증가했다. '
                '선형회귀도 평일 출퇴근 고수요 재현율은 94.0%에서 93.0%로 낮아져 전체 오차 개선과 고수요 포착 개선이 일치하지 않았다.')
    lines=['# 자전거 대여량 예측 보고서: 일출·일몰만 추가',intro,method,inputs,
           '## 비교 방법','학습 6,131개(신경망 6,108개), 검증 1,314개, 시험 1,314개. 과거 24시간 입력은 신경망에만 적용. 동일 설정·시드·전처리 규칙. 추가 전의 전체 예측이 기존 저장 결과와 일치하는지 검사했다.',
           '## 전체 오차',markdown(summary),'오차증가 = 추가 후 - 추가 전. 음수이면 개선이다.',conclusion,selection_note,
           '## 평일 출퇴근 고수요',markdown(test_timing[['variant','model','n','recall','precision','accuracy','FN','FP']]),
           '400대 이상을 고수요로 판단한다. 재현율·정밀도·정확도는 위 표에서 0~1 비율이다. 평일 월~금의 예측 대상 07·08·09·17·18·19시 240개를 평가한다. 공휴일을 제외하지 않았다.',
           '## 낮·밤과 일출·일몰 근처',markdown(solar_groups[['variant','model','group','rows','MAE']]),
           f'시험의 낮 {counts["낮"]}시간, 밤 {counts["밤"]}시간. 일출·일몰 1시간 이내 {counts["일출일몰1시간이내"]}시간은 낮·밤 구간과 겹치므로 합산하지 않는다.',
           '## 한계',limits,'대표 좌표의 천문 시각은 도시 전체의 조명·건물 그림자·지형·구름·실제 일조시간을 설명하지 못한다. 시간·월·계절과 겹치는 정보라 별도 효과가 작을 수도 있다.',
           '## 출처와 재현','[Open-Meteo 공식 문서](https://open-meteo.com/en/docs/historical-weather-api), 보관 원문과 좌표·시간대·해시는 data/external/solar/SOURCE.json 참조.',
           'python src/prepare_solar.py\npython src/run_solar_experiment.py\npython src/analyze_solar_experiment.py --render',
           '모델 재학습은 두 번째 명령만 수행한다. PDF 작성은 requirements-reports.txt의 별도 의존성을 사용한다. 구현·분석에 AI 도구 활용. 작성일 2026-11-20.']
    (DEST/'REPORT.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    setup_fonts()
    story=[p('인공지능과머신러닝 텀프로젝트','small'),p('자전거 대여량 예측 보고서\n일출·일몰만 추가','title'),
           p(intro),p('1 자료와 추가 입력','h'),p(method),p(inputs),
           p('2 같은 조건으로 비교','h'),p('학습 6,131개(신경망 6,108개), 검증 1,314개, 시험 1,314개를 시간순으로 나눴다. '
             '모델 구조·학습 설정·시드 42를 고정했다. 신경망은 과거 24시간을 사용한다. 추가 전 예측은 기존 결과와 일치하는지 확인했다.'),
           p('3 모델별 평균 절대 오차 MAE','h')]
    rows=[[NAMES[r.model],f'{r.validation_before_MAE:.2f}',f'{r.validation_after_MAE:.2f}',f'{r.test_before_MAE:.2f}',f'{r.test_after_MAE:.2f}'] for r in comparison.itertuples()]
    story += [table(['모델','검증\n기본','검증\n추가 후','시험\n기본','시험\n추가 후'],rows,[48,31,31,31,31]),Spacer(1,6),
              p('MAE는 실제 대여량과 예측값의 차이 크기를 평균한 값이며 단위는 대다. 낮을수록 좋다.','small'),
              p(conclusion),p('검증으로 선택한 조건','h'),p(selection_note),PageBreak(),
              p('일출·일몰만 추가','small'),p('출퇴근 고수요와 낮·밤 비교','title'),
              p('4 평일 출퇴근의 고수요 판단','h'),p('고수요는 실제 400대 이상이다. 예측 대상이 월~금 07·08·09·17·18·19시인 240개이며 공휴일도 포함한다. '
                '각 칸의 화살표는 기본 조건에서 일출·일몰 추가 조건으로의 변화를 뜻한다.')]
    before_t=timing[(timing.split=='test')&(timing.variant=='before')&(timing.group=='weekday_commute')].set_index('model')
    after_t=timing[(timing.split=='test')&(timing.variant=='after')&(timing.group=='weekday_commute')].set_index('model')
    rows=[]
    for m in MODELS:
        a,b=before_t.loc[m],after_t.loc[m]
        rows.append([NAMES[m]]+[f'{100*a[c]:.1f} → {100*b[c]:.1f}' for c in ['recall','precision','accuracy']]+[f'{int(a.FN)} → {int(b.FN)}',f'{int(a.FP)} → {int(b.FP)}'])
    story += [table(['모델','재현율 %','정밀도 %','정확도 %','놓친\n고수요','잘못 울린\n경보'],rows,[41,28,27,27,22,27]),Spacer(1,7),
              p('재현율은 실제 고수요 중 잡아낸 비율, 정밀도는 고수요 경보 중 맞은 비율, 정확도는 전체 판단 중 맞은 비율이다. '
                '실제 고수요가 215/240개이므로 재현율만으로 좋고 나쁨을 판단하지 않는다.','small'),
              p('5 시간대별 MAE: 기본 → 추가 후','h')]
    rows=[]
    for m in MODELS:
        row=[NAMES[m]]
        for g in ['commute_all_days','solar_daylight','solar_dark','solar_near_boundary']:
            a=group[(group.split=='test')&(group.variant=='before')&(group.model==m)&(group.group==g)].iloc[0]
            b=group[(group.split=='test')&(group.variant=='after')&(group.model==m)&(group.group==g)].iloc[0]
            row.append(f'{a.MAE:.1f} → {b.MAE:.1f}')
        rows.append(row)
    story += [table(['모델','출퇴근 전체요일','낮','밤','일출·일몰\n1시간 이내'],rows,[48,31,31,31,31]),Spacer(1,7),
              p(f'출퇴근은 모든 요일 330개, 낮은 {counts["낮"]}개, 밤은 {counts["밤"]}개, 경계 1시간 이내는 {counts["일출일몰1시간이내"]}개다. '
                '경계 구간과 출퇴근 구간은 낮·밤과 겹친다. 행의 구간을 합산하지 않는다.','small'),
              p('같은 모델이라도 전체·출퇴근·밤의 결과가 다를 수 있다. 시간대별 분석은 설명용이며 시험 지표에 맞춰 설정을 바꾸지 않는다.'),PageBreak(),
              p('일출·일몰만 추가','small'),p('오류 사례와 해석의 한계','title'),
              p('6 추가 후 모델의 큰 오차 사례','h'),p('검증으로 선택한 모델 종류에서 일출·일몰을 추가한 버전의 상위 절대오차 5개다. 시험 점수로 모델 종류를 다시 고르지 않았다.')]
    worst=after.assign(error=(after[selected]-after.target_count).abs()).nlargest(5,'error')
    rows=[[r.target_time.strftime('%Y-%m-%d %H시'),f'{r.target_count:.0f}',f'{r[selected]:.1f}',f'{r.error:.1f}','낮' if r['주간여부'] else '밤'] for _,r in worst.iterrows()]
    story += [table(['예측 대상 시각','실제','예측','절대 오차','낮/밤'],rows,[60,28,28,28,28]),Spacer(1,8),
              p('7 한계와 다음 확인','h'),p(limits),
              p('대표 한 지점의 천문 시각은 건물 그림자·지형·조명·구름·실제 일조시간을 반영하지 않는다. 정시의 낮·밤 판정이라 '
                '해가 뜨거나 지는 시간이 포함된 한 시간 전체를 완벽하게 설명하지도 않는다.'),
              p('원래 입력의 시각·월·계절·현재 일사량과 겹치는 정보가 있다. 성능 변화가 일출·일몰이 자전거 이용에 미치는 인과 효과를 뜻하지 않는다. '
                '추가 후가 나빠져도 입력이 불필요하다고 일반화할 수 없다.'),
              p('8 출처와 재현','h'),
              p('출처: Open-Meteo Historical Weather API, 서울 요청 좌표 37.5665°N, 126.9780°E. 응답 격자좌표·원문·해시는 data/external/solar/SOURCE.json에 보관했다.','small'),
              p('공식 문서: https://open-meteo.com/en/docs/historical-weather-api','small'),
              p('외부자료: 외부자료_일출일몰.csv\n학습: python src/run_solar_experiment.py\n보고서: python src/analyze_solar_experiment.py --render','small'),
              p('상세 수치·선택 기록·모델 가중치: reports/forecast/solar/\n구현·분석·문서에 AI 도구 활용. 작성일 2026년 11월 20일.','small')]
    PDF.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(PDF),pagesize=A4,leftMargin=19*mm,rightMargin=19*mm,topMargin=17*mm,bottomMargin=22*mm,
                          title='자전거 대여량 예측 보고서 일출일몰만 추가',author='자전거 대여량 예측 프로젝트')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    if render:
        import pypdfium2 as pdfium
        pdf=pdfium.PdfDocument(str(PDF));dest=ROOT/'tmp/pdfs/solar';dest.mkdir(parents=True,exist_ok=True)
        for i,page in enumerate(pdf):page.render(scale=1.5).to_pil().save(dest/f'page-{i+1}.png')
        assert len(pdf)==3,len(pdf)
    print(summary.to_string(index=False));print(selection_note);print(PDF)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--render',action='store_true');main(parser.parse_args().render)
