"""원본·출처·가공 자료를 자료.zip 하나로 묶고 무결성을 검사한다."""
import csv
import hashlib
import io
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'자료.zip'


def main():
    sources=sorted([p for folder in ['data/raw','data/external'] for p in (ROOT/folder).rglob('*') if p.is_file()])
    sources += [ROOT/name for name in ['외부자료.csv','외부자료_미세먼지.csv','외부자료_일출일몰.csv','외부자료_불쾌지수.csv','외부요인_리스트.md','LICENSE_DATASET.txt']]
    if len(sources)!=len(set(sources)) or any(not p.exists() for p in sources):
        raise ValueError('근거자료 누락 또는 중복')
    manifest=io.StringIO(newline='')
    writer=csv.writer(manifest)
    writer.writerow(['압축파일내경로','원본크기_바이트','SHA256'])
    description='''자전거 대여량 예측 과제 근거자료

이 압축파일은 프로젝트에 실제 보관된 원본, 출처 기록, 가공 자료를 모은 것입니다.
기준 폴더 구조를 유지하므로 출처 기록에 적힌 상대 경로로 원본을 찾을 수 있습니다.

1. data/raw/: UCI 자전거 원본 CSV, 내려받은 ZIP, 출처 및 다운로드 기록
2. data/external/: 공휴일 확인 원문 HTML, 확인 목록, 출처 설명
3. data/external/air_quality/: 에어코리아 2017·2018 전국 원본 ZIP,
   서울 측정소 추출본, 시간별 집계본, 출처 및 처리 설명
4. data/external/solar/: 일출·일몰 API 원본 JSON, 좌표·시간대·출처·해시
5. 외부자료.csv: 날짜별 공휴일·연휴 가공 자료
6. 외부자료_미세먼지.csv: 한 시간 지연 미세먼지 입력 자료
7. 외부자료_일출일몰.csv: 날짜별 일출·일몰 가공 자료
8. LICENSE_DATASET.txt: 원본 데이터 라이선스 안내
9. 외부요인_리스트.md: 기상 요인 후보·기존 입력 여부·추가 확보 상태
10. data/external/weather_candidates/: 기상청 ASOS·불쾌지수 설명 원문과 출처 기록
11. 파일목록.csv: 각 파일의 경로·크기·SHA256 확인값
12. data/external/rain_forecast/: 강수예보 수집 안내·공식 설명 원문·출처
    실제 과거 예보 원본의 확보 여부는 이 폴더의 README.md와 SOURCE.json 참조
13. 외부자료_불쾌지수.csv와 data/external/discomfort/: 원본 기온·습도로 계산한
    시간별 불쾌지수와 계산 근거·해시. 새 외부 관측이나 공식 예보값이 아닙니다.

출처의 웹 페이지를 모두 복제한 것은 아닙니다. 실제 확보한 원문과 각 출처 기록을
함께 보관합니다. 원문을 확보하지 못한 항목이나 해석상의 한계는 해당 출처 설명을
확인하세요. 일출·일몰은 천문 계산값이며 실제 일조시간 관측이 아닙니다.

과제의 자전거 분석 기간: 2017-12-01 ~ 2018-11-30
미세먼지 전국 원본은 2017·2018년 전체 자료이며, 서울 추출·연결 기간은 설명 참조.
모델 코드·학습 결과·보고서는 프로젝트에 별도로 보관합니다.

재생성: python src/package_evidence.py
'''
    with zipfile.ZipFile(OUTPUT,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sources:
            digest=hashlib.sha256()
            with path.open('rb') as source:
                for chunk in iter(lambda:source.read(1024*1024),b''):
                    digest.update(chunk)
            relative=path.relative_to(ROOT).as_posix()
            writer.writerow([relative,path.stat().st_size,digest.hexdigest()])
            archive.write(path,relative,compress_type=zipfile.ZIP_STORED if path.suffix in ['.zip','.gz'] else zipfile.ZIP_DEFLATED)
        archive.writestr('자료설명.txt',description.encode('utf-8-sig'))
        archive.writestr('파일목록.csv',manifest.getvalue().encode('utf-8-sig'))
    with zipfile.ZipFile(OUTPUT) as archive:
        assert archive.testzip() is None,'압축파일 무결성 검사 실패'
        records=list(csv.DictReader(io.StringIO(archive.read('파일목록.csv').decode('utf-8-sig'))))
        assert len(archive.namelist())==len(sources)+2
        for row in records:
            digest=hashlib.sha256()
            with archive.open(row['압축파일내경로']) as source:
                for chunk in iter(lambda:source.read(1024*1024),b''):
                    digest.update(chunk)
            assert digest.hexdigest()==row['SHA256'],row['압축파일내경로']
    print(f'{OUTPUT}\n근거파일 {len(sources)}개 + 자료설명·파일목록\n크기 {OUTPUT.stat().st_size:,} 바이트\nCRC 및 SHA256 전체 검증 완료')


if __name__=='__main__':
    main()
