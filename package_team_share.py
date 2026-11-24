"""조원 공유용 폴더와 전체/가벼운 ZIP을 실제 저장된 결과에서 만든다."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import zipfile
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MODELS=['Persistence','LinearRegression','GradientBoosting','RNN','Transformer']
NAMES=dict(zip(MODELS,['직전 대여량 기준','선형회귀','그래디언트 부스팅','순환 신경망 RNN','트랜스포머']))
LABELS={'base':'기본 정보만','calendar':'달력','air':'미세먼지','calendar_air':'달력+미세먼지','solar':'일출일몰',
        'solar_calendar':'일출일몰+달력','solar_air':'일출일몰+미세먼지','solar_calendar_air':'일출일몰+달력+미세먼지','discomfort':'불쾌지수'}
SKIP_PARTS={'__pycache__','.venv','.git','node_modules','.ipynb_checkpoints'}


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def is_large_original(path):
    return path.name in ['airkorea_2017.zip','airkorea_2018.zip']


def copy_file(source,target):
    if not source.resolve().is_relative_to(ROOT.parent.resolve()):raise ValueError('허용 범위 밖의 원본')
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)


def table_text(df):
    headers=list(df.columns)
    rows=[[f'{v:.2f}' if isinstance(v,(float,np.floating)) else str(v) for v in row] for row in df.itertuples(index=False,name=None)]
    return '\n'.join([' / '.join(headers)]+[' / '.join(row) for row in rows])


def load_results():
    base=ROOT/'reports/forecast'
    mixed=pd.read_csv(base/'solar_mix/metrics.csv')
    latest=pd.read_csv(base/'discomfort/metrics.csv')
    di=latest[latest.variant=='after'].copy();di['condition']='discomfort'
    columns=['condition','split','model','MAE','RMSE','R2']
    result=pd.concat([mixed[columns],di[columns]],ignore_index=True)
    assert len(result)==90 and not result.duplicated(['condition','split','model']).any()
    assert set(result.condition)==set(LABELS) and set(result.model)==set(MODELS)
    reference=latest[latest.variant=='before'].set_index(['split','model'])
    old=result[result.condition=='base'].set_index(['split','model'])
    for field in ['MAE','RMSE','R2']:np.testing.assert_allclose(old[field],reference.reindex(old.index)[field],atol=1e-4,rtol=0)
    timing=pd.read_csv(base/'solar_mix/timing_metrics.csv')
    timing=timing[(timing.split=='test')&(timing.group=='weekday_commute')].copy()
    extra=pd.read_csv(base/'discomfort/timing_metrics.csv')
    extra=extra[(extra.split=='test')&(extra.group=='weekday_commute')&(extra.variant=='after')].copy()
    extra['condition']='discomfort'
    fields=['condition','model','n','positives','negatives','TP','FP','FN','TN','recall','precision','accuracy']
    timing=pd.concat([timing[fields],extra[fields]],ignore_index=True)
    assert len(timing)==45 and not timing.duplicated(['condition','model']).any()
    assert (timing.n==240).all() and (timing.positives==215).all()
    assert (timing[['TP','FP','FN','TN']].sum(axis=1)==timing.n).all()
    return result,timing


def main(date):
    target=ROOT.parent/f'조원공유_자전거과제_{date}'
    full_zip=target.with_name(target.name+'_전체원본포함.zip')
    light_zip=target.with_name(target.name+'_공유용.zip')
    if target.exists() or full_zip.exists() or light_zip.exists():raise FileExistsError('같은 이름의 공유 결과가 이미 있습니다. 다른 --date 이름을 사용하세요.')
    result,timing=load_results()
    target.mkdir()
    summary=target/'01_전체비교표';summary.mkdir()
    pdf_dir=target/'02_PDF보고서';pdf_dir.mkdir()
    project=target/'03_실행코드와전체자료';project.mkdir()
    assignment=target/'04_교수님과제안내';assignment.mkdir()
    pivots={}
    for split,label in [('test','시험'),('validation','검증')]:
        frame=result[result.split==split].pivot(index='condition',columns='model',values='MAE').reindex(index=LABELS,columns=MODELS)
        frame.index=frame.index.map(LABELS);frame.columns=frame.columns.map(NAMES)
        frame=frame.reset_index(names='입력 조건');pivots[split]=frame
        frame.to_csv(summary/f'9조건_{label}_평균오차.csv',index=False,encoding='utf-8-sig')
    details=result.copy();details['condition']=details.condition.map(LABELS);details['model']=details.model.map(NAMES)
    details['split']=details.split.map({'test':'최종 시험','validation':'검증'})
    details.rename(columns={'condition':'입력 조건','model':'모델','split':'자료 구분','MAE':'평균절대오차_MAE_대','RMSE':'평균제곱근오차_RMSE_대','R2':'결정계수_R2'}).to_csv(summary/'전체90개_성능기록.csv',index=False,encoding='utf-8-sig')
    t=timing.copy();t['condition']=t.condition.map(LABELS);t['model']=t.model.map(NAMES)
    for field in ['recall','precision','accuracy']:t[field]=100*t[field]
    t.rename(columns={'condition':'입력 조건','model':'모델','n':'사례수','positives':'실제고수요','negatives':'실제비고수요',
                       'TP':'고수요적중','FP':'잘못울린경보','FN':'놓친고수요','TN':'비고수요적중',
                       'recall':'재현율_pct','precision':'정밀도_pct','accuracy':'정확도_pct'}).to_csv(summary/'9조건_평일출퇴근_고수요비교.csv',index=False,encoding='utf-8-sig')
    validation=result[result.split=='validation']
    chosen=validation.sort_values(['MAE','condition','model']).iloc[0]
    chosen_test=result[(result.split=='test')&(result.condition==chosen.condition)&(result.model==chosen.model)].iloc[0]
    best=result[result.split=='test'].sort_values(['MAE','condition','model']).iloc[0]
    for p in sorted((ROOT/'output/pdf').rglob('*.pdf')):copy_file(p,pdf_dir/p.name)
    assert len(list(pdf_dir.glob('*.pdf')))==10
    for folder in ['src','tests','docs','notebooks','data','reports','output/pdf']:
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file() and not (set(p.relative_to(ROOT).parts)&SKIP_PARTS):copy_file(p,project/p.relative_to(ROOT))
    roots=['README.md','외부요인_리스트.md','원본데이터.xlsx','외부자료.csv','외부자료_미세먼지.csv','외부자료_일출일몰.csv',
           '외부자료_불쾌지수.csv','requirements.txt','requirements-forecast.txt','requirements-forecast-lock.txt',
           'requirements-reports.txt','LICENSE_DATASET.txt','app.py']
    for name in roots:copy_file(ROOT/name,project/name)
    copy_file(ROOT.parent/'Week12_Term project.pdf',assignment/'교수님_텀프로젝트_안내.pdf')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    history=subprocess.check_output(['git','log','--oneline'],cwd=ROOT,text=True,encoding='utf-8')
    (project/'docs/공유시점_커밋이력.txt').write_text(f'공유 원본 커밋: {head}\n\n'+history,encoding='utf-8-sig')
    howto=r'''실행 방법 (Windows)

보고서와 CSV를 읽는 데 Python은 필요하지 않습니다.
실험을 다시 학습할 때만 아래 명령을 사용합니다.
PowerShell의 작업 위치는 이 파일이 들어 있는 03_실행코드와전체자료 폴더입니다.

py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-forecast.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -q

# 기본, 달력, 미세먼지, 달력+미세먼지
.\.venv\Scripts\python.exe src/run_forecast.py --with-air
# 일출일몰만
.\.venv\Scripts\python.exe src/run_solar_experiment.py
# 일출일몰과 다른 외부요인의 세 조합
.\.venv\Scripts\python.exe src/run_solar_mix_experiment.py
# 불쾌지수만
.\.venv\Scripts\python.exe src/prepare_discomfort.py
.\.venv\Scripts\python.exe src/run_discomfort_experiment.py

이미 계산된 결과는 reports/forecast에 있으므로 다시 학습할 필요 없이 확인할 수 있습니다.
PDF 재생성의 별도 의존성은 requirements-reports.txt입니다. 문서 생성은 Windows 맑은 고딕을 사용합니다.
개별 보고서의 재생성 명령은 프로젝트 README와 해당 실험의 REPORT.md에 있습니다.

공유용 ZIP에서는 data/external/air_quality/airkorea_2017.zip 및 airkorea_2018.zip 두 파일만 빠집니다.
서울 측정소 추출본, 시간별 집계본, 출처·해시와 모델 입력은 모두 들어 있습니다.
기존 모델 실험은 집계본으로 실행할 수 있습니다. 원문부터 재집계하려면 전체본을 쓰거나
src/download_air_quality.py로 원본을 다시 확보합니다. 서울 추출본만 재집계할 때는
src/prepare_air_quality.py --from-extracted를 사용합니다.

강수예보는 2017년 자료 미확보로 보류했습니다. run_rain_forecast_experiment.py는 실행 대상이 아닙니다.
프로젝트 폴더에는 .git 이력 저장소와 개인 가상환경을 넣지 않았으며 커밋 이력은 docs에 텍스트로 보관했습니다.
'''
    (project/'실행방법.txt').write_text(howto,encoding='utf-8-sig')
    guide=f'''자전거 대여량 예측 과제 — 조원 공유 자료
정리일: {date}
공유 원본 커밋: {head}

1. 무엇을 한 과제인가
현재 시간의 대여량·기상정보 등으로 다음 한 시간의 서울 자전거 대여 건수를 예측했습니다.
예를 들어 9시 기록으로 10시 대여 건수를 예측합니다.
원본은 2017-12-01~2018-11-30, 8,760시간입니다.
다음 시간 정답이 있는 사례는 8,759개입니다.

2. 읽는 순서
01_전체비교표: 아홉 입력 조건과 다섯 모델의 결과를 한곳에 모았습니다.
02_PDF보고서: 조건별 보고서 9개와 외부요인 분석 보고서 1개입니다.
03_실행코드와전체자료: 원본 엑셀/CSV, 공식 근거, 외부자료, 코드, 예측, 가중치, 검사, 작업 기록입니다.
04_교수님과제안내: 교수님이 제공한 과제 설명 PDF입니다.

3. 비교한 다섯 방법
- 직전 대여량 기준: 현재 500대면 다음 시간도 500대로 예측합니다.
- 선형회귀: 현재 대여량·날씨·시간 정보에 계수를 곱해 합산합니다.
- 그래디언트 부스팅: 여러 결정나무를 차례로 학습합니다.
- RNN과 Transformer: 최근 24시간의 기록을 시간순으로 넣습니다.
선형회귀·부스팅은 현재 시간 입력, 신경망은 최근 24시간 입력을 사용합니다.
따라서 모델 구조와 입력 정보량이 함께 다른 비교입니다.

4. 시행한 아홉 입력 조건
기본 / 달력 / 미세먼지 / 달력+미세먼지 / 일출일몰 /
일출일몰+달력 / 일출일몰+미세먼지 / 일출일몰+달력+미세먼지 / 불쾌지수.
공통 기본 입력에는 기온·습도·강수량·일사량·원본 휴일 정보가 이미 들어 있습니다.
불쾌지수는 기존 기온·습도를 조합해 만든 파생 특성이며 새 외부 관측이 아닙니다.

5. 결과를 읽는 방법
MAE는 실제값과 예측값 차이의 크기를 평균한 것으로 단위는 대이며 낮을수록 좋습니다.
검증 자료는 모델 선택에 사용했고, 시험 자료는 선택한 설정을 평가하는 데 사용했습니다.
전체 기록에서 검증 MAE가 가장 낮은 기존 선택은 {LABELS[chosen.condition]} / {NAMES[chosen.model]}입니다.
검증 MAE {chosen.MAE:.2f}대, 그 조건의 시험 MAE {chosen_test.MAE:.2f}대입니다.
관찰한 시험 점수 중 가장 낮은 값은 {LABELS[best.condition]} / {NAMES[best.model]}의 {best.MAE:.2f}대입니다.
시험 점수가 더 낮다는 이유로 기존 선택을 바꾸지는 않았습니다.
여러 외부요인을 모두 섞는다고 항상 좋은 결과가 나오지는 않았습니다.
불쾌지수는 트랜스포머 시험 MAE를 94.83대에서 91.80대로 줄였습니다.

시험 MAE 요약 (대, 낮을수록 좋음)
{table_text(pivots['test'].drop(columns='직전 대여량 기준'))}

6. 출퇴근 평가
예측 대상 시각이 07·08·09시이면 출근, 17·18·19시이면 퇴근입니다.
평일은 월~금이며 공휴일을 제외하지 않았습니다. 평일 출퇴근 시험 사례는 240개입니다.
시간당 400대 이상을 고수요로 정했습니다. 실제 고수요는 215개입니다.
재현율: 실제 고수요 중 잡아낸 비율.
정밀도: 고수요 경보 중 맞은 비율.
정확도: 전체 판단 중 맞은 비율.
모두 고수요라고 해도 재현율은 100%이므로 MAE, 경보 오류, 놓친 고수요도 함께 확인합니다.
비교표의 _pct 열은 퍼센트 단위입니다.

7. 결과의 한계
시간순 학습 6,131개(신경망 6,108개), 검증 1,314개, 시험 1,314개로 나눴습니다.
원본 다음 시간의 실제 날씨와 실제 운영 상태는 예측 입력에 넣지 않았습니다.
시드 42의 한 번의 분할이며 기존 시험 결과를 본 뒤 추가한 확장 실험입니다.
여러 시드와 독립된 새 기간에서 검증한 결과는 아닙니다.
최종 시험은 10~11월이고 불쾌지수 68 이상이 0개라 여름 무더위 효과는 확인하지 못했습니다.
미세먼지는 최종확정 자료를 한 시간 늦춰 사용했으나 실제 게시 지연을 복원하지 못했습니다.
강수예보는 2017년 예보자료 미확보로 보류했습니다. 자료 삭제 여부는 확인되지 않았고 실험 점수도 없습니다.
서울 전체 수요에 맞춰 서울 전체 고유 예보 격자를 집계하는 준비 코드만 남아 있습니다.

8. 무엇을 보내면 되나
이 폴더는 전국 미세먼지 원본 ZIP 두 개까지 포함한 전체본입니다.
함께 만든 _공유용.zip은 그 두 대용량 ZIP만 제외합니다. 서울 추출본·집계본·근거와 나머지 결과는 같습니다.
실험 코드와 보고서는 공유용에도 모두 있어 보통 조원들에게는 공유용 ZIP을 보내면 됩니다.
전국 원본 파일을 직접 확인할 때는 _전체원본포함.zip 또는 이 전체 폴더를 사용합니다.
파일목록.csv의 SHA256으로 개별 파일을 확인할 수 있으며 공유용_제외파일.csv에 빠진 두 파일을 적었습니다.
개인 가상환경·임시 이미지·.git·기존 자료.zip 중복 묶음은 넣지 않았습니다.

9. 조원들과 확인할 부분
교수님 과제 요구와 현재 결과를 대조하고, 발표에 쓸 조건·시간대·오류 사례를 정하면 됩니다.
03 폴더의 기존 발표 자료는 작업 과정의 초안이며 최신 전체 수치는 01_전체비교표를 기준으로 봅니다.
실제 팀원별 역할과 기여는 docs/WORK_LOG.md에 사실대로 작성할 부분이 남아 있습니다.
구현·자료 정리·분석·문서 작성에 AI 도구를 사용했습니다.
이 묶음을 만들면서 모델을 재학습하거나 실험 결과를 수정하지 않았습니다.
'''
    (target/'00_먼저읽기.txt').write_text(guide,encoding='utf-8-sig')
    files=sorted(p for p in target.rglob('*') if p.is_file())
    records=[{'파일경로':p.relative_to(target).as_posix(),'크기_바이트':p.stat().st_size,'SHA256':sha256(p)} for p in files]
    def write_manifest(path,rows):
        with path.open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['파일경로','크기_바이트','SHA256']);writer.writeheader();writer.writerows(rows)
    write_manifest(target/'파일목록.csv',records)
    excluded=[r for r in records if is_large_original(Path(r['파일경로']))]
    assert len(excluded)==2
    write_manifest(target/'공유용_제외파일.csv',excluded)
    files=sorted(p for p in target.rglob('*') if p.is_file())
    for archive_path,light in [(full_zip,False),(light_zip,True)]:
        included=[p for p in files if not (light and is_large_original(p))]
        with zipfile.ZipFile(archive_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for p in included:
                archive.write(p,(Path(target.name)/p.relative_to(target)).as_posix(),
                              compress_type=zipfile.ZIP_STORED if p.suffix.lower() in ['.zip','.gz','.xlsx','.pt','.pdf'] else zipfile.ZIP_DEFLATED)
        with zipfile.ZipFile(archive_path) as archive:
            assert archive.testzip() is None
            assert len(archive.namelist())==len(included)
            for p in included:
                h=hashlib.sha256()
                with archive.open((Path(target.name)/p.relative_to(target)).as_posix()) as stream:
                    for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
                assert h.hexdigest()==sha256(p),p
        print(f'{archive_path}: {archive_path.stat().st_size:,} bytes, {len(included)} files, CRC and SHA256 verified',flush=True)
    meta={'folder':str(target),'full_zip':str(full_zip),'light_zip':str(light_zip),'source_commit':head,
          'conditions':9,'models':5,'score_rows':len(result),'commute_rows':len(timing),'pdf_reports':10,
          'file_count':len(files),'excluded_large_originals':excluded}
    (ROOT/'reports/TEAM_SHARE_MANIFEST.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(target,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--date',default='2026-11-22');main(parser.parse_args().date)
