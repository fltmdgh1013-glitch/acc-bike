"""저장된 수치에서 네 조건별 독립 PDF 보고서를 생성하고 페이지를 렌더링한다."""
from pathlib import Path
from html import escape
import argparse
import hashlib
import json
from importlib.metadata import version
import numpy as np
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Flowable,KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT,TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'reports/forecast'
OUTPUT=ROOT/'output/pdf/조건별보고서'
QA=ROOT/'tmp/pdfs/condition_reports'
MODELS=['Persistence','LinearRegression','GradientBoosting','RNN','Transformer']
NAMES=dict(zip(MODELS,['직전 대여량 기준','선형회귀','그래디언트 부스팅','순환 신경망 RNN','트랜스포머']))
CONDITIONS=[
 ('기본 정보만 사용',BASE/'sequence','base',False,False),
 ('외부 달력 정보 추가',BASE/'external','after',True,False),
 ('미세먼지만 추가',BASE/'air_quality_only','after',False,True),
 ('외부 달력과 미세먼지 추가',BASE/'air_quality','after',True,True),
]
WIDTH=A4[0]-38*mm
S={}


def setup_fonts():
    fonts=Path('C:/Windows/Fonts')
    regular=fonts/'malgun.ttf'; bold=fonts/'malgunbd.ttf'
    if not regular.exists():
        raise FileNotFoundError('한글 PDF 생성을 위한 맑은 고딕 글꼴 경로가 필요합니다.')
    pdfmetrics.registerFont(TTFont('Korean',str(regular)))
    pdfmetrics.registerFont(TTFont('KoreanBold',str(bold if bold.exists() else regular)))
    for name,size,leading,space,font in [('body',10,14.5,5,'Korean'),('small',8.4,11.5,4,'Korean'),
                                         ('title',18,24,9,'KoreanBold'),('h',11.5,16,6,'KoreanBold'),
                                         ('cell',9.2,13.5,0,'Korean'),('head',9.1,13.5,0,'KoreanBold')]:
        S[name]=ParagraphStyle(name,fontName=font,fontSize=size,leading=leading,spaceAfter=space,
                               wordWrap='CJK',textColor=colors.black,keepWithNext=(name=='h'))


def p(text,style='body'):
    return Paragraph(escape(str(text)).replace('\n','<br/>'),S[style])


def table(headers,rows,widths):
    data=[[p(h,'head') for h in headers]]+[[p(x,'cell') for x in r] for r in rows]
    t=Table(data,colWidths=[x*mm for x in widths],repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DDEAF1')),
                          ('GRID',(0,0),(-1,-1),.45,colors.HexColor('#D9D9D9')),
                          ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),7),
                          ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),4),
                          ('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    for row in range(2,len(data),2):
        t.setStyle(TableStyle([('BACKGROUND',(0,row),(-1,row),colors.HexColor('#F6F8FA'))]))
    return t


class MetricBars(Flowable):
    def __init__(self,metrics):
        Flowable.__init__(self);self.width=WIDTH;self.height=112;self.metrics=metrics
    def draw(self):
        c=self.canv; maximum=max(self.metrics.loc[m,'MAE'] for m in MODELS)
        left=112; width=WIDTH-left-40
        for i,m in enumerate(MODELS):
            y=94-i*20
            c.setFont('Korean',8.5);c.setFillColor(colors.black);c.drawString(0,y+3,NAMES[m])
            value=self.metrics.loc[m,'MAE'];c.setFillColor(colors.HexColor('#228B88') if m!='Persistence' else colors.HexColor('#94A3B8'))
            c.rect(left,y,width*value/maximum,12,fill=1,stroke=0)
            c.setFillColor(colors.black);c.setFont('Korean',8.5);c.drawString(left+width*value/maximum+5,y+2,f'{value:.1f}')


def metrics(actual,pred):
    actual=np.asarray(actual);pred=np.asarray(pred)
    truth=actual>=400;alarm=pred>=400
    tp=int((truth&alarm).sum());fp=int((~truth&alarm).sum());fn=int((truth&~alarm).sum());tn=int((~truth&~alarm).sum())
    return dict(MAE=float(np.abs(actual-pred).mean()),precision=tp/(tp+fp) if tp+fp else float('nan'),
                recall=tp/(tp+fn) if tp+fn else float('nan'),accuracy=(tp+tn)/len(actual),TP=tp,FP=fp,FN=fn,TN=tn)


def read_condition(folder,variant):
    if variant=='base':
        val=pd.read_csv(folder/'all_validation_metrics.csv').set_index('model')
        test=pd.read_csv(folder/'all_test_metrics.csv').set_index('model')
        pred=pd.read_csv(folder/'all_test_predictions.csv',parse_dates=['input_time','target_time'])
    else:
        data=pd.read_csv(folder/'metrics.csv')
        val=data.loc[(data.split=='validation')&(data.variant==variant)].set_index('model')
        test=data.loc[(data.split=='test')&(data.variant==variant)].set_index('model')
        pred=pd.read_csv(folder/f'{variant}_test_predictions.csv',parse_dates=['input_time','target_time'])
    for m in MODELS:
        assert abs(metrics(pred.target_count,pred[m])['MAE']-test.loc[m,'MAE'])<1e-4
    return val,test,pred


def footer(c,doc):
    c.saveState();c.setFont('Korean',8);c.setFillColor(colors.HexColor('#555555'))
    c.drawString(19*mm,14*mm,'서울 자전거 다음 1시간 대여량 예측')
    c.drawRightString(A4[0]-19*mm,14*mm,str(doc.page));c.restoreState()


def build_report(condition,folder,variant,calendar,air):
    val,test,pred=read_condition(folder,variant)
    selected=val.MAE.idxmin();lowest=test.MAE.idxmin()
    title='자전거 대여량 예측 보고서 '+condition
    filename=f'자전거대여량예측_보고서_{condition.replace(" ","_")}.pdf'
    dest=OUTPUT/filename
    story=[p('인공지능과머신러닝 텀프로젝트','small'),p(title,'title')]
    story += [p(f'이 보고서는 {condition} 조건에서 다음 1시간 대여량을 예측한 결과를 정리한다. '
                f'검증 오차가 가장 낮은 모델은 {NAMES[selected]}이며 검증 평균 오차는 {val.loc[selected,"MAE"]:.2f}대다. '
                f'최종 시험에서 가장 낮은 평균 오차를 기록한 모델은 {NAMES[lowest]}로 {test.loc[lowest,"MAE"]:.2f}대다.')]
    story += [p('1 문제와 입력 조건','h'),p('현재 시간의 대여량과 날씨 등을 이용해 다음 시간의 서울 전체 대여 건수를 예측한다. '
                '예를 들어 9시 기록으로 10시 대여량을 맞힌다. 다음 시간의 실제 날씨와 실제 운영 결과는 입력하지 않는다.')]
    inputs='공통 입력은 현재 대여량과 기상 기록, 원본의 휴일·계절·운영 상태, 다음 시간의 시각·요일·월이다. '
    if calendar:
        inputs+='외부자료.csv에서 주말·공휴일·휴일 전후·연속 휴일 길이와 순번 등 8개 숫자를 예측 대상 날짜에 연결했다. '
    if air:
        inputs+='에어코리아 서울 도시대기 측정소 평균 PM10·PM2.5와 두 결측 표시를 추가했다. 입력 시각보다 한 시간 이전에 끝난 측정값을 사용했다. '
    if not calendar and not air: inputs+='외부 달력 자료나 미세먼지 자료를 추가하지 않은 기준 조건이다.'
    elif not calendar: inputs+='외부 달력의 8개 특성은 추가하지 않았다. 원본에 이미 있는 날짜·휴일 입력은 포함한다.'
    story += [p(inputs),p('2 자료와 평가 방법','h')]
    split=pd.read_csv(BASE/'split_summary.csv')
    rows=[]
    for r in split.itertuples():
        count=f'{r.rows:,}' + ('\n신경망 6,108' if r.split=='train' else '')
        rows.append([{'train':'학습','validation':'검증','test':'최종 시험'}[r.split],count,
                     f'{r.target_start} ~ {r.target_end}'])
    story += [table(['자료 구분','사례 수','예측 대상 시간 범위'],rows,[22,30,120]),Spacer(1,6),
              p('원본은 2017년 12월 1일부터 2018년 11월 30일까지 8,760시간이다. 다음 시간 정답이 있는 8,759개를 시간순으로 나눴다. '
                '신경망은 과거 24시간이 필요한 첫 23개 학습 사례를 제외한다. 검증·시험 사례는 모든 모델이 동일하다.','small'),
              p('3 모델별 대여량 오차','h')]
    rows=[[NAMES[m],f'{val.loc[m,"MAE"]:.2f}',f'{test.loc[m,"MAE"]:.2f}',f'{test.loc[m,"RMSE"]:.2f}',f'{test.loc[m,"R2"]:.3f}'] for m in MODELS]
    story += [table(['모델','검증 MAE\n대','시험 MAE\n대','시험 RMSE\n대','결정계수\nR²'],rows,[48,31,31,31,31]),Spacer(1,6),
              p('MAE는 실제값과 예측값 차이의 크기를 평균한 값이다. 낮을수록 좋다. RMSE는 큰 오차에 더 민감하며 R²는 정답률이 아니다.','small'),
              p('검증 최소 모델과 시험 최소 모델의 순위가 다를 수 있다. 시험 점수를 보고 모델을 다시 고르지 않는다.','small'),PageBreak()]
    # Page 2: identical time masks across all conditions.
    h=pred.target_time.dt.hour;weekday=pred.target_time.dt.day_of_week<5
    morning=h.isin([7,8,9]);evening=h.isin([17,18,19]);commute=morning|evening
    primary=weekday&commute
    actual=pred.loc[primary,'target_count']
    story += [p(condition,'small'),p('주요 시간대의 성능','title'),
              p('예측 대상 시간 기준으로 출근은 07·08·09시, 퇴근은 17·18·19시다. '
                '예를 들어 06시 기록으로 07시를 예측한 사례가 출근 구간에 들어간다. 출퇴근 전체 요일은 330개이고 평일은 240개다. '
                '평일은 월요일부터 금요일이며 공휴일을 별도로 제외하지 않았다.'),
              p('4 시간대별 대여량 오차','h')]
    rows=[]
    for m in MODELS:
        e=(pred[m]-pred.target_count).abs()
        rows.append([NAMES[m],f'{e.mean():.2f}',f'{e[morning].mean():.2f}',f'{e[evening].mean():.2f}',f'{e[commute].mean():.2f}'])
    story += [table(['모델','전체\n대','출근\n대','퇴근\n대','출퇴근 합계\n대'],rows,[48,31,31,31,31]),Spacer(1,7),
              p('위 표는 모든 요일을 포함한다. 같은 모델도 특정 시간대의 오차가 전체 평균보다 클 수 있다.','small'),
              p('5 평일 출퇴근의 고수요 판단','h'),
              p('교수님 자료의 예시대로 다음 시간 400대 이상을 고수요로 정했다. 새 분류 모델을 학습한 것이 아니라 고정된 대여량 예측에 400대 경계를 적용했다. '
                '재현율은 실제 고수요를 잡아낸 비율, 정밀도는 고수요 경보가 맞은 비율, 정확도는 모든 판단이 맞은 비율이다.')]
    rows=[];timings={}
    for m in MODELS:
        b=metrics(actual,pred.loc[primary,m]);timings[m]=b
        rows.append([NAMES[m],f'{b["recall"]*100:.1f}%',f'{b["precision"]*100:.1f}%',f'{b["accuracy"]*100:.1f}%',str(b['FN']),str(b['FP'])])
    story += [table(['모델','재현율','정밀도','정확도','놓친\n고수요','잘못 울린\n경보'],rows,[47,25,25,25,25,25]),Spacer(1,8),
              p(f'평일 출퇴근 240개 중 실제 고수요는 {int((actual>=400).sum())}개다. 항상 고수요라고 답해도 재현율은 100%가 된다. '
                '따라서 재현율만으로 모델을 평가하지 않고 오판 수와 대여량 오차를 함께 확인한다.'),
              p('지표를 함께 읽는 예','h'),
              p('실제 500대를 1,500대로 예측하면 대여량 오차는 1,000대다. 그러나 두 값 모두 400대 이상이므로 고수요 판단은 맞는다. '
                '반대로 실제 399대를 401대로 예측하면 오차는 2대지만 고수요 판단은 틀린다. 두 평가가 같은 순위를 보장하지 않는 이유다.'),
              p('부족한 표본의 한계','h'),
              p('비고수요 사례가 적어 몇 개의 경보 오류만으로 정밀도와 정확도가 달라질 수 있다. '
                '이번 기간의 100% 재현율이 다른 계절이나 연도에서도 유지된다고 해석하지 않는다.'),PageBreak()]
    # Page 3: failure cases, limits and traceable sources.
    story += [p(condition,'small'),p('오류 분석과 해석의 한계','title'),
              p('6 큰 오차가 난 사례','h'),p('이 조건에서 검증 오차가 가장 낮았던 모델과 시험 오차가 가장 낮았던 모델의 절대오차 상위 사례를 확인했다. '
                '표의 선택은 오류를 설명하기 위한 것이며 시험 자료로 모델을 재선택한 것이 아니다.')]
    for m in dict.fromkeys([selected,lowest]):
        tmp=pred.copy();tmp['error']=(tmp[m]-tmp.target_count).abs();worst=tmp.nlargest(3,'error')
        rows=[[r.target_time.strftime('%Y-%m-%d %H시'),f'{r.current_count:.0f}',f'{r.target_count:.0f}',f'{r[m]:.1f}',f'{r.error:.1f}'] for _,r in worst.iterrows()]
        story += [p(NAMES[m],'h'),table(['예측 대상 시각','직전 대여량','실제 대여량','예측 대여량','절대 오차'],rows,[60,28,28,28,28]),Spacer(1,6)]
    story += [p('출퇴근 급증·급감이나 운영 중단 같은 변화는 큰 오차로 이어질 수 있다. '
                '오차 사례의 날짜·날씨·운영 기록을 함께 검토하되, 이 표만으로 특정 요인이 원인이라고 단정하지 않는다.'),
              p('7 실험의 한계','h'),
              p('한 해 자료를 한 번 시간순으로 분할한 결과다. 학습·검증·시험의 계절 분포가 다르며, 기본 비교 이후의 확장 실험은 시험 결과를 본 뒤 추가한 탐색적 비교다. '
                '모델 설정은 고정했고 시험 결과에 맞춰 다시 조정하지 않았다. 여러 시드와 독립된 새 기간 검증이 필요하다.')]
    if air:
        story += [p('미세먼지는 최종확정 자료라 당시 실시간 게시값과 다를 수 있다. 한 시간 지연을 가정했지만 실제 게시 지연을 복원하지 않았다. '
                    '서울 25개 측정소의 단순 평균은 이용자별 실제 노출량과도 다르다. 이 실험을 실시간 운영 성능으로 그대로 해석하지 않는다.')]
    elif calendar:
        story += [p('일부 공휴일 정보는 원본 휴일 입력과 중복된다. 달력상 휴일은 모든 이용자의 실제 휴무나 자전거 운영 중단을 뜻하지 않는다. '
                    '추가 정보의 효과는 모델·시간대·평가 지표마다 다를 수 있다.')]
    else:
        story += [p('선형회귀와 부스팅은 현재 한 시간 입력을 사용하고 신경망은 최근 24시간을 사용한다. '
                    '모델 구조와 입력 정보량이 함께 다르므로 순수한 구조만의 우열 비교는 아니다.')]
    story += [p('8 출처와 재현','h')]
    sources=[('원본 자전거 데이터','https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand')]
    if calendar:
        sources += [('2018년 공휴일 한국천문연구원','https://astro.kasi.re.kr/life/post/almanac?search_year=2018'),
                    ('2017년 12월 공휴일 교차 확인','https://www.timeanddate.com/holidays/south-korea/2017')]
    if air: sources += [('공식 미세먼지 자료 에어코리아','https://www.airkorea.or.kr/web/last_amb_hour_data')]
    for label,url in sources:
        story.append(Paragraph(f'{escape(label)}: <link href="{escape(url,quote=True)}" color="#18516A">{escape(url)}</link>',S['small']))
    if variant=='base': command='python src/run_forecast.py'
    elif calendar and not air: command='python src/run_external_experiment.py'
    else: command='python src/run_air_quality_experiment.py'+(' --without-calendar' if not calendar else '')
    story += [p('실험 재실행: '+command,'small'),
              p('수치 근거: '+str(folder.relative_to(ROOT)).replace('\\','/')+'의 성능표와 전체 예측 CSV. 보고서 재생성: python src/create_condition_reports.py','small'),
              p('문서 작성일 2026년 11월 20일. 구현과 분석에 AI 도구를 활용했다. 팀원별 실제 담당과 기여는 별도로 기록한다.','small')]
    doc=SimpleDocTemplate(str(dest),pagesize=A4,rightMargin=19*mm,leftMargin=19*mm,topMargin=17*mm,bottomMargin=22*mm,
                          title=title,author='자전거 대여량 예측 프로젝트')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    return {'condition':condition,'file':filename,'calendar':calendar,'air':air,
            'validation_min_model':selected,'test_min_model':lowest,'test_rows':len(pred),
            'validation_metrics':val.MAE.to_dict(),'test_metrics':test.MAE.to_dict(),
            'weekday_commute_metrics':timings,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--render',action='store_true');args=parser.parse_args()
    setup_fonts();OUTPUT.mkdir(parents=True,exist_ok=True)
    manifests=[]
    common=None
    for condition,folder,variant,cal,air in CONDITIONS:
        _,_,pred=read_condition(folder,variant)
        labels=pred[['input_time','target_time','target_count']]
        if common is None:common=labels
        else:pd.testing.assert_frame_equal(common,labels)
        item=build_report(condition,folder,variant,cal,air)
        if args.render:
            import pypdfium2 as pdfium
            pdf=pdfium.PdfDocument(str(OUTPUT/item['file']))
            target=QA/Path(item['file']).stem;target.mkdir(parents=True,exist_ok=True)
            item['pages']=len(pdf)
            for i,page in enumerate(pdf):page.render(scale=1.5).to_pil().save(target/f'page-{i+1}.png')
            assert len(pdf)==3,(item['file'],len(pdf))
        manifests.append(item)
        print(item['file'],item.get('pages',''),flush=True)
    (OUTPUT/'보고서_생성기록.json').write_text(json.dumps(manifests,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'requirements-reports.txt').write_text('\n'.join(f'{name}=={version(name)}' for name in ['reportlab','pandas','pypdfium2'])+'\n',encoding='utf-8')


if __name__=='__main__':main()
