"""일출일몰 혼합 3조건의 독립 보고서와 8조건 전체 비교를 만든다. 재학습 없음."""
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from reportlab.platypus import SimpleDocTemplate,PageBreak,Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from create_condition_reports import ROOT,BASE,NAMES,MODELS,p,table,setup_fonts,footer,metrics

DEST=BASE/'solar_mix'
LABELS={'base':'기본 정보만','calendar':'달력','air':'미세먼지','calendar_air':'달력+미세먼지','solar':'일출일몰',
        'solar_calendar':'일출일몰+달력','solar_air':'일출일몰+미세먼지','solar_calendar_air':'일출일몰+달력+미세먼지'}
CONTROLS={'solar_calendar':'calendar','solar_air':'air','solar_calendar_air':'calendar_air'}
OUTPUT=ROOT/'output/pdf/조건별보고서'


def markdown(df):
    def fmt(v):return f'{v:.2f}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+
                     ['| '+' | '.join(map(fmt,row))+' |' for row in df.itertuples(index=False,name=None)])


def main(render=False):
    score=pd.read_csv(DEST/'metrics.csv')
    timing=pd.read_csv(DEST/'timing_metrics.csv')
    groups=pd.read_csv(DEST/'group_metrics.csv')
    selected=json.loads((DEST/'selection.json').read_text(encoding='utf-8'))
    protocol=json.loads((DEST/'protocol.json').read_text(encoding='utf-8'))
    for path,sha in protocol['source_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
    for path,sha in json.loads((DEST/'cached_prediction_hashes.json').read_text(encoding='utf-8')).items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
    test=score[score.split=='test'].set_index(['condition','model'])
    val=score[score.split=='validation'].set_index(['condition','model'])
    assert len(test)==len(val)==40 and not test.index.duplicated().any()
    reference=pd.read_csv(BASE/'solar/after_test_predictions.csv',parse_dates=['input_time','target_time'])
    frozen_note=(f'8조건 전체 검증 최소는 {LABELS[selected["condition"]]} / {NAMES[selected["model"]]}이며 '
                 f'검증 MAE {selected["MAE"]:.2f}대다. 신규 시험 예측 전에 선택을 기록했다. '
                 '기존 시험을 이미 확인한 뒤 확장한 비교이므로 새 기간 독립 검증은 아니다.')
    limits=('같은 분할·시드 42·설정에서 입력 조합을 비교했다. 학습 6,131개(신경망 6,108개), 검증·시험은 각 1,314개다. '
            '새 세 조합만 학습하고 기존 다섯 조건은 저장된 결과를 재사용했다. 모든 시험 시각·정답을 대조했고 '
            '각 모델의 새로운 시험 예측에서 MAE를 다시 계산했다. 신경망은 입력 차원과 학습 계수 수가 달라진다.')
    pivot=test.MAE.unstack('model').reindex(LABELS).reindex(columns=MODELS)
    pivot.index=pivot.index.map(LABELS);pivot.columns=pivot.columns.map(NAMES)
    pivot.reset_index(names='조건').to_csv(DEST/'8조건_시험오차_한국어.csv',index=False,encoding='utf-8-sig')
    valpivot=val.MAE.unstack('model').reindex(LABELS).reindex(columns=MODELS)
    valpivot.index=valpivot.index.map(LABELS);valpivot.columns=valpivot.columns.map(NAMES)
    valpivot.reset_index(names='조건').to_csv(DEST/'8조건_검증오차_한국어.csv',index=False,encoding='utf-8-sig')
    intro=['# 일출·일몰과 다른 외부요인의 혼합 실험',limits,frozen_note,
           '## 전체 8조건 시험 MAE',markdown(pivot.reset_index(names='조건')),
           '## 전체 8조건 검증 MAE',markdown(valpivot.reset_index(names='조건')),
           '추가 요인이 좋아 보이는 시험 결과만 골라 최종 성능으로 확정하지 않는다. 다음 독립 기간과 여러 시드에서 확인해야 한다.']
    changes=[];manifest=[]
    setup_fonts()
    for key,control in CONTROLS.items():
        label=LABELS[key];folder=DEST/key
        pred=pd.read_csv(folder/'test_predictions.csv',parse_dates=['input_time','target_time'])
        columns=['input_time','target_time','target_count','current_count']
        pd.testing.assert_frame_equal(pred[columns],reference[columns],check_dtype=False)
        weekday=(pred.target_time.dt.day_of_week<5)&pred.target_time.dt.hour.isin([7,8,9,17,18,19])
        for m in MODELS:
            b=metrics(pred.target_count,pred[m]);assert abs(b['MAE']-test.loc[(key,m),'MAE'])<1e-4
            saved=timing[(timing.split=='test')&(timing.condition==key)&(timing.model==m)&(timing.group=='weekday_commute')].iloc[0]
            calculated=metrics(pred.loc[weekday,'target_count'],pred.loc[weekday,m])
            for field in ['TP','FP','FN','TN','precision','recall','accuracy']:
                assert abs(saved[field]-calculated[field])<1e-9
            changes.append({'조건':label,'모델':NAMES[m],'model':m,
                            '시험_MAE':test.loc[(key,m),'MAE'],
                            '기본대비_오차증가':test.loc[(key,m),'MAE']-test.loc[('base',m),'MAE'],
                            '일출일몰단독대비_오차증가':test.loc[(key,m),'MAE']-test.loc[('solar',m),'MAE'],
                            '같은외부정보_일출일몰추가_오차증가':test.loc[(key,m),'MAE']-test.loc[(control,m),'MAE']})
        local=val.loc[key].MAE.idxmin()
        findings=[]
        for m in MODELS[1:]:
            now=test.loc[(key,m),'MAE'];solar=test.loc[('solar',m),'MAE'];other=test.loc[(control,m),'MAE']
            findings.append(f'{NAMES[m]}: 시험 MAE {now:.2f}대. 일출일몰만 대비 {now-solar:+.2f}대, '
                            f'{LABELS[control]} 대비 {now-other:+.2f}대 변화.')
        sources=('일출·일몰: 외부자료_일출일몰.csv와 data/external/solar/SOURCE.json. '
                 '달력: 외부자료.csv와 data/external/외부자료_출처와설명.json. '
                 '미세먼지: data/external/air_quality/의 집계·출처 기록. 기존 자료.zip에도 원본이 보관돼 있다.')
        caveat=('일출·일몰은 예측 대상 KST 정시의 천문 시각으로, 지형·건물·구름·실제 일조량을 반영하지 않는다. '
                '달력은 예측 대상 날짜에 연결한다. 미세먼지는 입력 시각보다 한 시간 앞서 끝난 최종확정 관측이며 '
                '실제 게시 지연을 복원하지 않았다. 다음 시간의 실제 날씨와 실제 운영 상태는 입력하지 않는다.')
        rows=[[NAMES[m]]+[f'{test.loc[(c,m),"MAE"]:.2f}' for c in ['base','solar',control,key]]+
              [f'{val.loc[(key,m),"MAE"]:.2f}'] for m in MODELS]
        comparison=pd.DataFrame(rows,columns=['모델','기본','일출일몰만',LABELS[control],'현재조합','조합검증'])
        lines=[f'# 자전거 대여량 예측 보고서: {label}',limits,
               '## 같은 모델에서 네 조건 비교',markdown(comparison),
               '시험 MAE는 낮을수록 좋다. 변화량은 추가 후-비교 조건으로 음수일 때 개선이다.',*findings,
               '## 선택 기록',frozen_note,f'이 조합 내부의 검증 최소 모델은 {NAMES[local]}다.',
               '## 입력 연결과 한계',caveat,
               '## 수치 출처와 재현',sources,
               '학습: python src/run_solar_mix_experiment.py\n분석: python src/analyze_solar_mix.py --render',
               '구현·분석에 AI 도구 활용. 단일 시드 탐색 비교로 인과 효과나 일반적인 우열을 확정하지 않는다. 작성일 2026-11-20.']
        story=[p('인공지능과머신러닝 텀프로젝트','small'),p('자전거 대여량 예측 보고서\n'+label,'title'),
               p('1 실험 조건','h'),p(limits),p('일출·일몰 4개 특성에 '+LABELS[control]+' 정보를 더했다. '
                 '원본에 이미 있는 휴일·시각·계절·현재 일사량은 모든 조건에 유지한다.'),p('2 같은 모델의 네 조건 비교','h'),
               table(['모델','시험\n기본','시험\n일출일몰만','시험\n'+LABELS[control].replace('+','+\n'),'시험\n현재조합','검증\n현재조합'],rows,[42,26,26,26,26,26]),Spacer(1,7),
               p('MAE는 실제 대여량과 예측값의 차이 크기를 평균한 값이며 단위는 대다. 낮을수록 좋다.','small'),
               p('검증으로 선택한 조건','h'),p(frozen_note),
               p(f'이 조합 안에서는 {NAMES[local]}의 검증 오차가 가장 작다. 전체 8조건의 선택과 구분한다.','small'),PageBreak(),
               p(label,'small'),p('주요 시간대의 성능','title'),p('3 평일 출퇴근 고수요','h'),
               p('월~금 07·08·09·17·18·19시 240개 중 400대 이상을 고수요로 판단했다. 공휴일도 포함한다. '
                 '재현율 칸은 일출일몰만 → 현재 조합이며 다른 지표는 현재 조합의 값이다.')]
        t=timing[(timing.split=='test')&(timing.group=='weekday_commute')].set_index(['condition','model'])
        rows=[]
        for m in MODELS:
            a,b=t.loc[('solar',m)],t.loc[(key,m)]
            rows.append([NAMES[m],f'{a.recall*100:.1f} → {b.recall*100:.1f}',f'{b.precision*100:.1f}%',f'{b.accuracy*100:.1f}%',str(int(b.FN)),str(int(b.FP))])
        story += [table(['모델','재현율 %','정밀도','정확도','놓친\n고수요','잘못 울린\n경보'],rows,[41,28,27,27,22,27]),Spacer(1,7),
                  p('실제 고수요는 215/240개다. 항상 고수요라고 해도 재현율이 100%이므로 대여량 오차·정밀도·정확도·오판 수를 함께 본다.','small'),
                  p('4 시간대별 MAE: 일출일몰만 → 현재 조합','h')]
        g=groups[groups.split=='test'].set_index(['condition','model','group'])
        rows=[]
        for m in MODELS:
            row=[NAMES[m]]
            for group in ['commute_all_days','solar_daylight','solar_dark','solar_near_boundary']:
                a,b=g.loc[('solar',m,group)],g.loc[(key,m,group)]
                row.append(f'{a.MAE:.1f} → {b.MAE:.1f}')
            rows.append(row)
        story += [table(['모델','출퇴근\n전체 요일','낮','밤','일출일몰\n1시간 이내'],rows,[48,31,31,31,31]),Spacer(1,7),
                  p('출퇴근은 모든 요일 330개, 낮 585개, 밤 729개, 경계 1시간 이내 221개다. 경계·출퇴근은 낮·밤과 겹치므로 합산하지 않는다.','small'),
                  p('전체 오차가 줄어도 출퇴근 재현율은 낮아질 수 있다. 시간대별 표는 설명용이며 이 결과로 모델 설정을 다시 바꾸지 않았다.'),PageBreak(),
                  p(label,'small'),p('오류 사례와 해석','title'),p('5 조합 내부 검증 최소 모델의 오류','h'),
                  p(f'이 조합의 {NAMES[local]}에서 절대오차가 큰 시험 사례 4개다. 시험 점수로 모델을 다시 고른 것이 아니다.')]
        worst=pred.assign(error=(pred[local]-pred.target_count).abs()).nlargest(4,'error')
        rows=[[r.target_time.strftime('%Y-%m-%d %H시'),str(int(r.target_count)),f'{r[local]:.1f}',f'{r.error:.1f}'] for _,r in worst.iterrows()]
        story += [table(['예측 대상 시각','실제','예측','절대 오차'],rows,[70,34,34,34]),Spacer(1,7),
                  p('6 입력과 해석의 한계','h'),p(caveat),
                  p('기존 시험 결과를 본 뒤 수행한 단일 시드 확장이다. 여러 조합을 시험한 뒤 시험 성적이 좋은 경우만 고르면 우연한 성능을 과대평가할 수 있다. '
                    '독립된 새 기간과 여러 초기값으로 재검증해야 한다.'),
                  p('이 비교는 모델 성능의 변화이며 외부요인이 자전거 이용량을 직접 얼마나 바꾸는지 증명한 인과 분석은 아니다. '
                    '추가 입력에 따라 신경망 계수 수와 최적화 경로도 달라진다.'),
                  p('7 출처와 재현','h'),p(sources,'small'),
                  p('공식 출처: UCI 자전거 데이터, 한국천문연구원 월력요항, 에어코리아 최종확정 측정자료, Open-Meteo 일출·일몰 API. 정확한 URL과 해시는 각 출처 기록에 보관했다.','small'),
                  p('학습: python src/run_solar_mix_experiment.py\n보고서: python src/analyze_solar_mix.py --render\n상세 수치·가중치: reports/forecast/solar_mix/','small'),
                  p('구현·분석·문서에 AI 도구 활용. 작성일 2026년 11월 20일.','small')]
        lines += ['## 출퇴근 고수요 상세',markdown(t.loc[key].reset_index()[['model','n','TP','FP','FN','TN','recall','precision','accuracy']])]
        (folder/'REPORT.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
        OUTPUT.mkdir(parents=True,exist_ok=True)
        filename='자전거대여량예측_보고서_'+label.replace('+','_')+'.pdf';path=OUTPUT/filename
        SimpleDocTemplate(str(path),pagesize=A4,leftMargin=19*mm,rightMargin=19*mm,topMargin=17*mm,bottomMargin=22*mm,
                          title='자전거 대여량 예측 보고서 '+label,author='자전거 대여량 예측 프로젝트').build(story,onFirstPage=footer,onLaterPages=footer)
        if render:
            import pypdfium2 as pdfium
            pdf=pdfium.PdfDocument(str(path));qa=ROOT/'tmp/pdfs/solar_mix'/key;qa.mkdir(parents=True,exist_ok=True)
            for i,page in enumerate(pdf):page.render(scale=1.5).to_pil().save(qa/f'page-{i+1}.png')
            assert len(pdf)==3,(key,len(pdf))
        manifest.append({'condition':key,'file':str(path.relative_to(ROOT)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        print(path,flush=True)
    changes=pd.DataFrame(changes)
    changes.drop(columns='model').to_csv(DEST/'조합별_오차변화_한국어.csv',index=False,encoding='utf-8-sig')
    human=timing[(timing.split=='test')&(timing.group=='weekday_commute')].copy()
    human['condition']=human.condition.map(LABELS);human['model']=human.model.map(NAMES)
    human.rename(columns={'condition':'조건','model':'모델','recall':'재현율','precision':'정밀도','accuracy':'정확도'}).to_csv(DEST/'8조건_출퇴근고수요_한국어.csv',index=False,encoding='utf-8-sig')
    intro += ['## 새 조합의 오차 변화',markdown(changes.drop(columns='model')),
              '오차증가는 현재 조합 - 비교 조건이다. 음수면 개선이다. 일출일몰만 대비와 같은 외부정보에 일출일몰을 더한 효과를 구분한다.',
              '## 재현','python src/run_solar_mix_experiment.py\npython src/analyze_solar_mix.py --render',
              '기존 결과는 원래 폴더에 유지한다. source_sha256과 cached_prediction_hashes.json에서 입력과 재사용 결과의 일치 여부를 확인한다.']
    (DEST/'REPORT.md').write_text('\n\n'.join(intro)+'\n',encoding='utf-8')
    (DEST/'report_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pivot.to_string());print(frozen_note)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--render',action='store_true');main(parser.parse_args().render)
