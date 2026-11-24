"""강수예보의 발표·대상 시각을 구분하여 과거에 발표된 예보만 연결한다."""
import pandas as pd
from forecast_data import ROOT

RAIN_FORECAST_FILE=ROOT/'data/external/rain_forecast/서울_강수예보_정규화.csv'
SEOUL_GRID_FILE=ROOT/'data/external/rain_forecast/서울_대상격자.csv'
RAIN_FORECAST_COLUMNS=['서울_강수예보격자비율','서울_눈예보격자비율']
FORECAST_AUDIT_COLUMNS=['사용예보_발표시각','사용예보_가용시각','예보_집계격자수']
PUBLICATION_LAG_MINUTES=30


def kst_naive(values):
    result=pd.to_datetime(values,errors='raise')
    if result.isna().any():raise ValueError('예보 시각 결측')
    if result.dt.tz is not None:result=result.dt.tz_convert('Asia/Seoul').dt.tz_localize(None)
    return result


def aggregate_city_forecasts(forecasts,expected_grids):
    """같은 발표·대상 시각의 서울 고유 격자를 동일 가중치로 집계한다.

    고정 대상 격자가 모두 있는 발표본만 사용한다. 여러 행정동이 같은 격자에
    대응해도 한 번만 집계하며, 누락 격자를 무강수로 취급하지 않는다.
    """
    xy=['격자X','격자Y']
    if not set(xy).issubset(expected_grids.columns):raise ValueError('서울 대상 격자 목록에 X/Y가 필요합니다.')
    grids=expected_grids[xy].copy()
    for c in xy:
        grids[c]=pd.to_numeric(grids[c],errors='raise')
        if grids[c].isna().any() or not ((grids[c]>0)&(grids[c]%1==0)).all():
            raise ValueError('유효하지 않은 대상 격자 좌표')
    grids=grids.drop_duplicates()
    if len(grids)<2:raise ValueError('단일 대표 격자를 서울 전체 예보로 사용할 수 없습니다.')
    required=['발표시각_KST','예보대상시각_KST','강수형태코드',*xy]
    if not set(required).issubset(forecasts.columns):raise ValueError('격자별 정규화 강수예보 필수 열 누락')
    right=forecasts[required].copy()
    for c in xy:
        right[c]=pd.to_numeric(right[c],errors='raise')
        if right[c].isna().any() or not ((right[c]>0)&(right[c]%1==0)).all():raise ValueError('유효하지 않은 예보 격자 좌표')
    expected=set(map(tuple,grids.to_numpy()))
    actual=set(map(tuple,right[xy].drop_duplicates().to_numpy()))
    if actual-expected:raise ValueError('서울 고정 대상 목록 밖의 예보 격자가 있습니다.')
    if expected-actual:raise ValueError('서울 고정 대상 격자의 예보가 통째로 누락됐습니다.')
    right['발표시각_KST']=kst_naive(right['발표시각_KST'])
    right['예보대상시각_KST']=kst_naive(right['예보대상시각_KST'])
    right['강수형태코드']=pd.to_numeric(right['강수형태코드'],errors='raise')
    if not right['강수형태코드'].isin([0,1,2,3,4,5,6,7]).all():raise ValueError('유효하지 않은 강수형태 코드')
    lead=right['예보대상시각_KST']-right['발표시각_KST']
    if not ((lead>pd.Timedelta(0))&(lead<=pd.Timedelta(hours=6))).all():raise ValueError('초단기예보의 대상 시각 범위 오류')
    pairs=['발표시각_KST','예보대상시각_KST']
    if (right.groupby(pairs+xy)['강수형태코드'].nunique()>1).any():raise ValueError('같은 격자·발표·대상 시각의 예보가 서로 다릅니다.')
    right=right.drop_duplicates(pairs+xy)
    right[RAIN_FORECAST_COLUMNS[0]]=(right['강수형태코드']!=0).astype(float)
    right[RAIN_FORECAST_COLUMNS[1]]=right['강수형태코드'].isin([2,3,6,7]).astype(float)
    grouped=right.groupby(pairs,sort=True)
    result=grouped[RAIN_FORECAST_COLUMNS].mean()
    result['예보_집계격자수']=grouped.size()
    incomplete=result['예보_집계격자수']!=len(grids)
    result=result.loc[~incomplete].reset_index()
    if result.empty:raise ValueError('서울 대상 격자가 모두 있는 발표본이 없습니다.')
    result.attrs['불완전_발표대상쌍_제외수']=int(incomplete.sum())
    result.attrs['서울_고유격자수']=len(grids)
    return result


def attach_rain_forecast(frame,forecasts,publication_lag_minutes=PUBLICATION_LAG_MINUTES,expected_grids=None):
    """입력 시각 이하의 가용시각 중 같은 대상 시각의 최신 예보를 선택한다.

    정규화 CSV의 시각은 KST. 30분 게시 지연은 검증 전의 보수적 가정이며
    당시 실제 게시 지연을 복원했다고 주장하지 않는다. 누락을 0으로 바꾸지 않는다.
    """
    if publication_lag_minutes<0:raise ValueError('예보 가용 지연은 음수일 수 없습니다.')
    if len(frame)==0:raise ValueError('예측 사례가 없습니다.')
    if any(c in frame for c in RAIN_FORECAST_COLUMNS+FORECAST_AUDIT_COLUMNS):
        raise ValueError('강수예보 입력이 이미 연결되어 있습니다.')
    if expected_grids is None:
        if not SEOUL_GRID_FILE.exists():raise FileNotFoundError('서울 전체 선택 지역의 고정 대상 격자 목록이 필요합니다.')
        expected_grids=pd.read_csv(SEOUL_GRID_FILE)
    right=aggregate_city_forecasts(forecasts,expected_grids).rename(columns={'발표시각_KST':'사용예보_발표시각','예보대상시각_KST':'target_time'})
    right['사용예보_가용시각']=right['사용예보_발표시각']+pd.Timedelta(minutes=publication_lag_minutes)
    left=frame.copy();left['_rain_order']=range(len(left))
    if left[['input_time','target_time']].isna().any().any():raise ValueError('사례의 입력·정답 시각 결측')
    result=pd.merge_asof(left.sort_values('input_time'),right.sort_values('사용예보_가용시각'),
                         left_on='input_time',right_on='사용예보_가용시각',by='target_time',direction='backward',allow_exact_matches=True)
    missing=result[RAIN_FORECAST_COLUMNS[0]].isna()
    if missing.any():
        sample=result.loc[missing,'target_time'].head(3).astype(str).tolist()
        raise ValueError(f'사전에 가용한 동일 대상시각 예보가 없는 사례 {int(missing.sum())}개: {sample}. 결측을 무강수로 채우지 않습니다.')
    assert (result['사용예보_가용시각']<=result.input_time).all()
    result=result.sort_values('_rain_order').drop(columns=['_rain_order'])
    result.index=frame.index
    pd.testing.assert_frame_equal(result[frame.columns],frame)
    return result


def enrich_rain_forecast_sets(sets,forecasts):
    grids=pd.read_csv(SEOUL_GRID_FILE)
    return {name:attach_rain_forecast(frame,forecasts,expected_grids=grids) for name,frame in sets.items()}
