"""기존 예측을 날짜별로 짝지어 외부요인의 조건별 성능 차이를 분석한다."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from create_condition_reports import ROOT, BASE, NAMES, p, table, setup_fonts, footer
from reportlab.platypus import SimpleDocTemplate, PageBreak, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

DEST = BASE / 'external_factors'
PDF = ROOT / 'output/pdf/자전거대여량예측_외부요인별_효과분석.pdf'
MODELS = ['LinearRegression', 'GradientBoosting', 'RNN', 'Transformer']
CONDITIONS = {'기본': 'sequence', '달력': 'external', '미세먼지': 'air_quality_only', '달력+미세먼지': 'air_quality'}
PAIRS = [('달력 효과', '기본', '달력'), ('미세먼지만의 효과', '기본', '미세먼지'),
         ('달력에 미세먼지 추가 효과', '달력', '달력+미세먼지')]
PM10 = '미세먼지_PM10_1시간지연'
PM25 = '초미세먼지_PM25_1시간지연'


def load_data():
    frames = {}
    for name, folder in CONDITIONS.items():
        filename = 'all_test_predictions.csv' if folder == 'sequence' else 'after_test_predictions.csv'
        frames[name] = pd.read_csv(BASE/folder/filename, parse_dates=['input_time', 'target_time'])
    reference = frames['기본'][['input_time', 'target_time', 'target_count', 'current_count']]
    for frame in frames.values():
        pd.testing.assert_frame_equal(reference, frame[reference.columns], check_dtype=False)
        assert not frame.input_time.duplicated().any()
    raw = pd.read_csv(ROOT/'data/raw/SeoulBikeData.csv', encoding='latin1')
    raw['input_time'] = pd.to_datetime(raw.Date, format='%d/%m/%Y')+pd.to_timedelta(raw.Hour, unit='h')
    raw = raw.sort_values('input_time').reset_index(drop=True)
    raw['next_functioning'] = raw['Functioning Day'].shift(-1)
    calendar = pd.read_csv(ROOT/'외부자료.csv', parse_dates=['날짜'])
    air = pd.read_csv(ROOT/'외부자료_미세먼지.csv', parse_dates=['입력시각', '측정자료종료시각'])
    assert not air['입력시각'].duplicated().any()
    assert (air['입력시각']-air['측정자료종료시각'] == pd.Timedelta(hours=1)).all()
    data = reference.merge(raw[['input_time', 'Rainfall(mm)', 'Temperature(°C)', 'Functioning Day', 'next_functioning']],
                           on='input_time', validate='one_to_one')
    data['날짜'] = data.target_time.dt.normalize()
    data = data.merge(calendar, on='날짜', how='left', validate='many_to_one')
    data = data.merge(air, left_on='input_time', right_on='입력시각', how='left', validate='one_to_one')
    assert len(data)==1314 and data[[PM10, PM25, '휴일여부_주말포함']].notna().all().all()
    pd.testing.assert_series_equal(data.target_time, reference.target_time)
    train_times = raw.input_time.iloc[:6131]
    train_air = air.set_index('입력시각').loc[train_times]
    thresholds = {col: train_air[col].quantile([1/3, 2/3]).tolist() for col in [PM10, PM25]}
    errors = {name: {m: (frame[m]-frame.target_count).abs().to_numpy() for m in MODELS}
              for name, frame in frames.items()}
    return data, thresholds, errors


def group_masks(data, thresholds):
    result = {'전체': np.ones(len(data), dtype=bool),
              '일반 평일': ((data['주말여부']==0)&(data['특별공휴일여부']==0)).to_numpy(),
              '주말(공휴일 제외)': ((data['주말여부']==1)&(data['특별공휴일여부']==0)).to_numpy(),
              '특별 공휴일': (data['특별공휴일여부']==1).to_numpy(),
              '현재 비 없음': (data['Rainfall(mm)']==0).to_numpy(),
              '현재 비 있음': (data['Rainfall(mm)']>0).to_numpy()}
    for label, col in [('PM10', PM10), ('PM2.5', PM25)]:
        low, high = thresholds[col]
        result[label+' 낮음'] = (data[col]<=low).to_numpy()
        result[label+' 중간'] = ((data[col]>low)&(data[col]<=high)).to_numpy()
        result[label+' 높음'] = (data[col]>high).to_numpy()
    return result


def block_interval(delta, dates, seed=42, repetitions=2000):
    """동일 날짜의 시간들을 함께 재표집. 날짜간 독립 가정의 조건부 구간이다."""
    daily = pd.DataFrame({'day': dates.to_numpy(), 'delta': delta}).groupby('day').delta.agg(['sum', 'count'])
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(daily), size=(repetitions, len(daily)))
    values = daily['sum'].to_numpy()[ix].sum(axis=1)/daily['count'].to_numpy()[ix].sum(axis=1)
    return np.quantile(values, [.025, .975])


def analyze():
    data, thresholds, errors = load_data()
    masks = group_masks(data, thresholds)
    for labels in [('일반 평일','주말(공휴일 제외)','특별 공휴일'),
                   ('현재 비 없음','현재 비 있음'),('PM10 낮음','PM10 중간','PM10 높음'),
                   ('PM2.5 낮음','PM2.5 중간','PM2.5 높음')]:
        assert np.all(np.stack([masks[label] for label in labels]).sum(axis=0)==1)
    comparisons, slices = [], []
    for model in MODELS:
        for label, before, after in PAIRS:
            delta = errors[before][model]-errors[after][model]
            folder = CONDITIONS[after]
            saved = pd.read_csv(BASE/folder/'metrics.csv')
            saved_mae = saved.loc[(saved.split=='test')&(saved.variant=='after')&(saved.model==model),'MAE'].item()
            assert abs(saved_mae-errors[after][model].mean())<1e-4
            low, high = block_interval(delta, data['날짜'])
            comparisons.append({'모델': NAMES[model], 'model': model, '비교': label,
                                '추가전_MAE': errors[before][model].mean(), '추가후_MAE': errors[after][model].mean(),
                                '개선량_대': delta.mean(), '개선율_퍼센트': 100*delta.mean()/errors[before][model].mean(),
                                '날짜블록_하한': low, '날짜블록_상한': high})
        for label, mask in masks.items():
            part = data.loc[mask]
            row = {'모델': NAMES[model], 'model': model, '구간': label, '시간수': len(part),
                   '날짜수': part['날짜'].nunique(), '소표본_주의': len(part)<100 or part['날짜'].nunique()<10,
                   '실제대여량_평균': part.target_count.mean()}
            row.update({name+'_MAE': values[model][mask].mean() if mask.any() else np.nan for name, values in errors.items()})
            row['달력_개선량'] = row['기본_MAE']-row['달력_MAE']
            row['미세먼지_개선량'] = row['기본_MAE']-row['미세먼지_MAE']
            row['달력에미세먼지_개선량'] = row['달력_MAE']-row['달력+미세먼지_MAE']
            slices.append(row)
    summary, slices = pd.DataFrame(comparisons), pd.DataFrame(slices)
    # 원래 검증으로 선정된 기본 RNN을 고정하고 운영 전환 사례를 사후 진단한다.
    transitions = (data['Functioning Day'] != data.next_functioning).to_numpy()
    residuals = []
    for name, mask in [('운영 상태 유지', ~transitions), ('운영 상태 전환(사후 진단)', transitions)]:
        part = data.loc[mask]
        residuals.append({'구간': name, '시간수': len(part), '날짜수': part['날짜'].nunique(),
                          '기본RNN_전체절대오차중비중_퍼센트':100*errors['기본']['RNN'][mask].sum()/errors['기본']['RNN'].sum(),
                          **{condition: e['RNN'][mask].mean() if mask.any() else np.nan for condition, e in errors.items()}})
    residuals = pd.DataFrame(residuals)
    DEST.mkdir(parents=True, exist_ok=True)
    summary.to_csv(DEST/'외부자료_전후효과.csv', index=False, encoding='utf-8-sig')
    slices.to_csv(DEST/'요인별_오차비교.csv', index=False, encoding='utf-8-sig')
    residuals.to_csv(DEST/'운영전환_사후진단.csv', index=False, encoding='utf-8-sig')
    details = {'시간수':len(data), '날짜수':data['날짜'].nunique(), '시작':str(data.target_time.min()),
               '종료':str(data.target_time.max()), '농도경계':thresholds, '농도경계기준':'학습 6131개 입력시각의 하위/상위 3분위, 건강등급 아님',
               '구간표본':{label:{'시간수':int(mask.sum()),'날짜수':int(data.loc[mask,'날짜'].nunique())} for label,mask in masks.items()},
               '부트스트랩':{'반복수':2000,'시드':42,'단위':'예측 대상 날짜','제한':'날짜 간 독립 가정, 재학습 및 다중비교 보정 제외'}}
    (DEST/'분석설정.json').write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding='utf-8')
    return data, thresholds, summary, slices, residuals


def md_table(frame):
    def fmt(x):
        return f'{x:.2f}' if isinstance(x, (float, np.floating)) else str(x)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |', '| '+' | '.join(['---']*len(frame.columns))+' |']+
                     ['| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False, name=None)])


def create_report(data, thresholds, summary, slices, residuals, render=False):
    setup_fonts()
    intro = ('기본·달력·미세먼지·달력+미세먼지의 같은 시험 1,314시간을 짝지어 비교했다. '
             '개선량 = 추가 전 MAE - 추가 후 MAE이며 양수는 개선, 음수는 악화다. '
             '현재 대여량·날씨·날짜 정보는 모든 조건에 이미 들어 있다.')
    caveat = ('이는 외부자료를 넣었을 때 고정된 모델의 예측 성능이 어떻게 달라졌는지에 대한 탐색적 분석이다. '
              '미세먼지나 공휴일이 이용량을 직접 얼마나 바꿨는지 증명하는 인과 분석이 아니다. '
              '시험 결과를 이미 본 뒤 진행했으며 모델을 다시 선택하거나 조정하지 않았다.')
    holiday = data[data['특별공휴일여부']==1]
    holiday_note = (f'시험의 특별 공휴일은 {holiday["날짜"].nunique()}일, {len(holiday)}시간뿐이며 '
                    f'실제 대여량이 0인 사례가 {(holiday.target_count==0).sum()}시간이다. '
                    '이 구간의 결과는 공휴일 자체와 운영 상태의 영향을 분리하지 못한다.')
    transition_share = residuals.iloc[1]['기본RNN_전체절대오차중비중_퍼센트']
    findings = ('트랜스포머에서 미세먼지만 추가한 효과는 일반 평일에 개선, 주말에는 악화로 나타났다. '
                'PM2.5가 학습 기준 상위 3분위인 시간에는 오히려 악화됐다. '
                '미세먼지가 높을수록 이 입력이 더 도움이 된다고 해석할 수 없다.')
    lines = ['# 자전거 대여량 예측: 외부요인별 효과 분석', intro, caveat, '## 주요 발견', findings, holiday_note,
             '## 전체 효과와 날짜 단위 재표집 구간', md_table(summary.drop(columns='model')),
             '95% 구간은 시험 날짜를 2,000회 재표집한 조건부 범위다. 시간별 독립을 가정하지 않지만 날짜 간 상관과 학습의 불확실성은 반영하지 않는다. 다중비교를 보정한 유의성 검정이 아니다.',
             '## 요인별 분석', '미세먼지 구간은 학습 입력의 3분위로 고정했다. 건강상 좋음·나쁨 등급이 아니다. PM10과 PM2.5는 따로 나눴으며 강한 상관 때문에 서로의 독립 효과가 아니다.',
             md_table(slices.drop(columns='model')), '## 운영 전환의 사후 진단',md_table(residuals),
             f'운영 전환 8시간은 기본 RNN 전체 절대오차의 {transition_share:.2f}%다. 건당 오차가 크지만 전체 오차의 대부분을 설명하지는 않는다.',
             '운영 전환은 정답 시각 운영 상태를 오류 설명에만 사용한다. 모델 입력이나 실시간 사전 경보로 사용하지 않는다.',
             '## 다음 실험',
             '- 달력과 미세먼지 입력을 함께 넣으면 각각의 이점이 더해진다고 가정하지 않는다. 각 모델과 시간대에서 따로 검증한다.',
             '- 시간별로 알려진 운영·정비 공지가 과거에도 존재했는지 먼저 확인한다. 미래 운영 결과를 그대로 넣으면 안 된다.',
             '- 강수 변화가 큰 오류와 겹치는지 확인한 뒤, 예측 시각 이전에 발행된 강수 예보나 최근 2~3시간의 강수 추세를 비교한다. 현재 비 관측은 기존 입력이다.',
             '- 미세먼지는 농도 단독보다 최근 상승·하락과 출퇴근·주말의 결합 특성을 학습/검증 구간에서만 시험한다. 현재 결과를 근거로 개선을 보장하지 않는다.',
             '- 새 기간의 독립 평가와 여러 시드로 재검증한다. 같은 시험 구간을 계속 확인하며 고른 결과는 최종 성능으로 확정하지 않는다.',
             '## 재현 및 출처', 'python src/analyze_external_factors.py --render',
             '수치 출처: reports/forecast/sequence, external, air_quality_only, air_quality의 저장된 시험 예측 CSV; data/raw/SeoulBikeData.csv; 외부자료.csv; 외부자료_미세먼지.csv. 원본 출처는 각 자료의 보관 메타데이터 참조.',
             '구현·분석에 AI 도구를 사용했다. 작성일: 2026-11-20.']
    (DEST/'REPORT.md').write_text('\n\n'.join(lines)+'\n', encoding='utf-8')
    story=[p('인공지능과머신러닝 텀프로젝트','small'),p('외부요인별 효과 분석','title'),p(intro),p(caveat),
           p('1 전체 시험 기간에서의 효과','h'),p('아래 숫자는 MAE 감소량(대)이다. 원본 휴일·시각 정보 위에 달력 자료를 추가한 효과를 비교한다.','small')]
    rows=[]
    for model in MODELS:
        part=summary[summary.model==model].set_index('비교')
        rows.append([NAMES[model]]+[f'{part.loc[label,"개선량_대"]:+.2f}' for label,_,_ in PAIRS])
    story += [table(['모델','달력만 추가','미세먼지만 추가','달력에 미세먼지'],rows,[52,40,40,40]),Spacer(1,12),
              p('2 결과가 얼마나 흔들리는가','h'),p('트랜스포머의 개선량과 날짜 단위 재표집 95% 범위다. 같은 날의 시간들은 함께 뽑았다. 날짜를 2,000회 재표집했으며 학습은 반복하지 않았다.')]
    part=summary[summary.model=='Transformer']
    rows=[[r['비교'],f'{r["개선량_대"]:+.2f}',f'{r["날짜블록_하한"]:+.2f} ~ {r["날짜블록_상한"]:+.2f}'] for _,r in part.iterrows()]
    story += [table(['비교','개선량(대)','재표집 범위(대)'],rows,[85,32,55]),Spacer(1,8),
              p('트랜스포머는 앞선 실험에서 조건 차이가 컸기 때문에 설명용으로 함께 살폈다. 시험 점수로 새로운 최종 모델을 고른 것이 아니다. '
                '모든 모델의 범위와 개선율은 함께 저장한 상세 표에 있다.'),
              p('재표집 범위는 날짜 간 독립을 가정한다. 연속된 오염·기상 사건, 계절 편중, 재학습에 따른 변동은 반영하지 못한다. '
                '여러 모델·구간을 동시에 본 결과이므로 통계적 유의성을 확정하는 근거로 사용하지 않는다.'),
              p('읽는 방법','h'),p('달력만 추가했을 때 좋아져도 미세먼지를 더 넣으면 나빠질 수 있다. '
                '다른 모델의 결과도 같을 필요는 없다. 추가 정보가 많다는 사실만으로 더 좋은 예측이 보장되지는 않는다.'),PageBreak()]
    story += [p('외부요인별 효과 분석','small'),p('어떤 상황에서 도움이 됐나','title'),
              p('각 구간에서 기본 조건보다 줄어든 MAE다. 양수는 개선이다. 달력만 또는 미세먼지만 추가한 효과를 각각 비교한다. '
                '서로 다른 구간은 중복될 수 있으므로 행을 합산하지 않는다.'),
              p('3 구간별 개선량','h')]
    rows=[]
    for label in group_masks(data,thresholds):
        a=slices[(slices.model=='GradientBoosting')&(slices['구간']==label)].iloc[0]
        b=slices[(slices.model=='Transformer')&(slices['구간']==label)].iloc[0]
        rows.append([label+(' *' if a['소표본_주의'] else ''),f'{a["시간수"]}/{a["날짜수"]}',
                     f'{a["달력_개선량"]:+.1f}',f'{a["미세먼지_개선량"]:+.1f}',f'{b["달력_개선량"]:+.1f}',f'{b["미세먼지_개선량"]:+.1f}'])
    story += [table(['구간','시간/날짜','부스팅\n달력','부스팅\n미세먼지','트랜스포머\n달력','트랜스포머\n미세먼지'],rows,[46,22,26,26,26,26]),Spacer(1,8),
              p('* 100시간 미만 또는 10날짜 미만은 소표본이다. 시간 수가 많아도 한 사건의 여러 시간일 수 있어 독립적인 관측 수와 다르다.','small'),
              p('4 미세먼지 구간의 기준','h')]
    for label,col in [('PM10',PM10),('PM2.5',PM25)]:
        low,high=thresholds[col]
        story.append(p(f'{label}: 낮음 ≤ {low:.2f}, 중간 {low:.2f} 초과 ~ {high:.2f} 이하, 높음 > {high:.2f} µg/m³.','small'))
    story += [p('경계는 학습 6,131개 입력 시간의 하위·상위 3분위에서 정했다. 건강 기준의 좋음·나쁨 등급이 아니다. '
                '예측 입력 시각보다 한 시간 앞서 끝난 서울 평균 측정값을 썼다. 두 물질을 따로 넣어 재학습한 실험은 아니므로 PM10과 PM2.5의 독립 효과를 분리할 수 없다.'),
              p('공휴일의 한계','h'),p(holiday_note),PageBreak()]
    story += [p('외부요인별 효과 분석','small'),p('남은 오류와 다음 실험','title'),
              p('5 기본 RNN의 운영 전환 진단','h'),p('앞서 검증으로 선택한 기본 RNN을 고정해, 현재와 다음 시간의 운영 상태가 달라지는 사례를 따로 확인했다. 아래 숫자는 개선량이 아닌 MAE(대)다.')]
    rows=[[r['구간'],str(r['시간수'])]+[f'{r[c]:.2f}' for c in CONDITIONS] for _,r in residuals.iterrows()]
    story += [table(['구간','시간수','기본','달력','미세먼지','둘 다'],rows,[57,19,24,24,24,24]),Spacer(1,8),
              p('운영 전환은 다음 시간의 실제 상태를 사용한 사후 진단이다. 사전에 알 수 있는 특성이라는 뜻이 아니다. '
                f'8시간이 기본 RNN 전체 절대오차에서 차지하는 비중은 {transition_share:.2f}%다. 건당 오차가 커도 전체 오차의 대부분은 아니다.'),
              p('6 다음에 비교할 방법','h'),
              p('첫째, 과거 시점에 공개된 운영·정비 공지가 있는지 확인한다. 실제 다음 시간 운영 결과는 입력하지 않는다.'),
              p('둘째, 현재 강수량은 이미 모델 입력이다. 추가 실험은 예측 전에 발행된 강수 예보 또는 최근 2~3시간의 강수 변화로 구분한다.'),
              p('셋째, 미세먼지 상승·하락이나 출퇴근·주말과의 결합 특성을 학습·검증 구간에서 비교한다. 이번 한 번의 결과로 개선을 보장하지 않는다.'),
              p('7 해석할 때 지켜야 할 조건','h'),
              p('휴일·비·오염 농도는 계절·시간대·운영 여부와 함께 움직인다. 예를 들어 오염이 높은 시간의 대여량이 적더라도 원인이 오염이라고 단정할 수 없다. '
                '농도 구간별 모델 성능 차이도 특정 입력 하나의 중요도 점수가 아니다.'),
              p('미세먼지는 최종확정 관측으로 실제 게시 지연을 복원하지 않았다. 새 기간과 여러 시드의 검증이 필요하다. '
                '현재 시험 구간을 반복해 확인하며 고른 결과는 최종 성능으로 확정하지 않는다.'),
              p('출처와 재현','h'),p('기존 네 조건의 시험 예측 CSV, UCI 원본, 외부자료.csv, 외부자료_미세먼지.csv를 사용했다. 새 자료를 내려받거나 모델을 재학습하지 않았다.','small'),
              p('상세 표·설정: reports/forecast/external_factors/\n재현: python src/analyze_external_factors.py --render','small'),
              p('분석·문서 작성에 AI 도구 활용. 작성일 2026년 11월 20일.','small')]
    PDF.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(PDF),pagesize=A4,leftMargin=19*mm,rightMargin=19*mm,topMargin=17*mm,bottomMargin=22*mm,
                          title='자전거 대여량 예측 외부요인별 효과 분석',author='자전거 대여량 예측 프로젝트')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    if render:
        import pypdfium2 as pdfium
        pdf=pdfium.PdfDocument(str(PDF));dest=ROOT/'tmp/pdfs/external_factors';dest.mkdir(parents=True,exist_ok=True)
        for i,page in enumerate(pdf):page.render(scale=1.5).to_pil().save(dest/f'page-{i+1}.png')
        assert len(pdf)==3, len(pdf)
    print(summary[['모델','비교','개선량_대','날짜블록_하한','날짜블록_상한']].to_string(index=False))
    print(residuals.to_string(index=False))
    print(PDF)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--render',action='store_true');args=parser.parse_args()
    create_report(*analyze(),render=args.render)
