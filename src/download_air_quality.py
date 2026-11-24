"""공식 원본 다운로드. 보관된 서울 추출본으로 학습할 때는 실행할 필요가 없다."""
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path
from prepare_air_quality import DATA


def main():
    DATA.mkdir(parents=True,exist_ok=True)
    manifest=DATA/'미세먼지_출처와처리.json'
    expected={}
    if manifest.exists():
        expected={x['year']:x['sha256'] for x in json.loads(manifest.read_text(encoding='utf-8'))['원본']}
    for year in [2017,2018]:
        dest=DATA/f'airkorea_{year}.zip'
        if dest.exists() and zipfile.is_zipfile(dest):
            if year in expected and hashlib.sha256(dest.read_bytes()).hexdigest()!=expected[year]:
                raise ValueError(f'{year} 원본 해시가 보관 기록과 다릅니다.')
            print('Already downloaded:',dest.name,flush=True)
            continue
        url=f'https://www.airkorea.or.kr/jfile/readDownloadFile.do?fileId=1657eebba9237&fileSeq={str(year)[-2:]}'
        part=dest.with_suffix('.zip.part')
        with urllib.request.urlopen(url,timeout=60) as response,part.open('wb') as output:
            while chunk:=response.read(1024*1024): output.write(chunk)
        if not zipfile.is_zipfile(part):
            raise ValueError('다운로드 결과가 ZIP이 아닙니다. 서버 응답을 확인하세요.')
        if year in expected and hashlib.sha256(part.read_bytes()).hexdigest()!=expected[year]:
            raise ValueError(f'{year} 원본이 이후 수정됐습니다. 처리 기록을 검토한 뒤 재추출하세요.')
        part.replace(dest)
        print('Downloaded:',dest.name,dest.stat().st_size,flush=True)


if __name__=='__main__': main()
