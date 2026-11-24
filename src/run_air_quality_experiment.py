"""달력까지 포함한 입력에 과거 미세먼지를 더한 통제 비교 실험."""
import json
import argparse
import hashlib
import platform
from importlib.metadata import version
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from forecast_data import OUT, ROOT, RAW, NUMERIC, CATEGORICAL, prepare
from external_features import EXTERNAL_FILE, EXTERNAL_COLUMNS, enrich_sets
from air_quality_features import AIR_FILE, AIR_COLUMNS, enrich_air_sets, export_hourly_companion
from solution import make_pipeline,scores
from sequence_models import CONFIG,sequence_data,RNNForecast,TransformerForecast,train_model
from run_external_experiment import seed_everything,predict_frozen,MODEL_LIST
from evaluate_timing import binary_scores,timing_masks

DEST=OUT/'air_quality'


def main(include_calendar=True):
    DEST=OUT/('air_quality' if include_calendar else 'air_quality_only')
    DEST.mkdir(parents=True,exist_ok=True)
    original=prepare()
    calendar=(enrich_sets(original,pd.read_csv(EXTERNAL_FILE)) if include_calendar else original)
    calendar_columns=EXTERNAL_COLUMNS if include_calendar else []
    hourly=pd.read_csv(AIR_FILE)
    dust,imputation=enrich_air_sets(calendar,hourly)
    sets={'before':calendar,'after':dust}
    # Save a human-readable hourly companion; preserve missing values rather than fill with future data.
    export_hourly_companion(hourly)
    protocol={
        '내용':'기존 입력에 과거 미세먼지 추가 비교',
        '외부달력정보사용':include_calendar,
        '조건':{'before':'기존 정보+달력' if include_calendar else '기존 정보만',
                'after':'기존 정보+달력+미세먼지' if include_calendar else '기존 정보+미세먼지'},
        '추가특성':AIR_COLUMNS,'시간연결':'입력 시각 t에 대해 종료시각 t-1의 서울 평균을 사용',
        '자료출처':'에어코리아 최종확정 측정자료, 서울 도시대기 측정소 평균',
        '기상실측미래값사용':False,'결측처리':imputation,
        '미세먼지파일sha256':hashlib.sha256(AIR_FILE.read_bytes()).hexdigest(),
        '원본sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),
        '달력sha256':hashlib.sha256(EXTERNAL_FILE.read_bytes()).hexdigest() if include_calendar else None,
        '시드':42,'신경망설정':CONFIG,
        '부스팅설정':dict(n_estimators=200,max_depth=3,learning_rate=.05,random_state=42),
        '자료수':{k:len(v) for k,v in original.items()},'신경망학습수':len(original['train'])-23,
        '선택':'두 조건의 검증 MAE 최소 모델/조건. 이 실험의 시험 평가 전 고정. 시험 점수로 변경하지 않음.',
        '고수요경계':400,'python':platform.python_version(),
        '패키지':{p:version(p) for p in ['numpy','pandas','scikit-learn','torch','openpyxl']},
        '한계':'최종확정값을 한 시간 늦춰 쓴 과거자료 실험. 실제 게시값·지연을 복원하지 못함. 이미 시험 결과를 본 뒤 확장한 탐색적 단일 시드 비교.'}
    (DEST/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    vals,frozen,metadata=[],[],[]
    for variant,split_sets in sets.items():
        numeric=NUMERIC+calendar_columns+(AIR_COLUMNS if variant=='after' else [])
        features=numeric+CATEGORICAL
        batches,mean,std=sequence_data(split_sets,numeric=numeric)
        for name in MODEL_LIST:
            seed_everything()
            if name=='Persistence': model,model_batches=None,None
            elif name in ['LinearRegression','GradientBoosting']:
                estimator=(LinearRegression() if name=='LinearRegression' else GradientBoostingRegressor(
                    n_estimators=200,max_depth=3,learning_rate=.05,random_state=42))
                model=make_pipeline(estimator,numeric=numeric)
                model.fit(split_sets['train'][features],split_sets['train'].target_count)
                model_batches=None
            else:
                inputs=batches['train'][0].shape[2]
                model=RNNForecast(inputs,32) if name=='RNN' else TransformerForecast(inputs,32,24)
                model_batches=batches
                detail=train_model(name,model,{k:batches[k] for k in ['train','validation']},mean,std,output_dir=DEST/variant)
                detail.update(variant=variant,features=features,target_mean=mean,target_std=std,
                              parameters=sum(p.numel() for p in model.parameters()))
                metadata.append(detail)
            item=(variant,name,model,features,model_batches,mean,std)
            frozen.append(item)
            vals.append({'variant':variant,'model':name,**scores(split_sets['validation'].target_count,predict_frozen(item,'validation',sets))})
            print(variant,name,'validation MAE:',vals[-1]['MAE'],flush=True)
    pd.DataFrame(vals).to_csv(DEST/'validation_metrics.csv',index=False)
    selected=min(vals,key=lambda r:(r['MAE'],r['variant'],r['model']))
    (DEST/'selection.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    (DEST/'training_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    results,timing_rows,group_rows=[],[],[]
    for split in ['validation','test']:
        frame=dust[split].reset_index(drop=True)
        masks=timing_masks(frame.target_time)
        if include_calendar:
            masks['calendar_rest_day']=frame['휴일여부_주말포함'].to_numpy()==1
            masks['calendar_work_day']=~masks['calendar_rest_day']
        pred_frames={v:frame[['input_time','target_time','target_count','current_count',*calendar_columns,*AIR_COLUMNS]].copy() for v in sets}
        for item in frozen:
            variant,name=item[:2]
            pred=predict_frozen(item,split,sets)
            if item[4] is not None: pd.testing.assert_series_equal(item[4][split][2].target_time,frame.target_time)
            pred_frames[variant][name]=pred
            results.append({'split':split,'variant':variant,'model':name,**scores(frame.target_count,pred)})
            for group,mask in masks.items():
                actual,estimated=frame.target_count.to_numpy()[mask],pred[mask]
                timing_rows.append({'split':split,'variant':variant,'model':name,'group':group,**binary_scores(actual,estimated)})
                group_rows.append({'split':split,'variant':variant,'model':name,'group':group,'rows':int(mask.sum()),**scores(actual,estimated)})
        for variant,pred in pred_frames.items(): pred.to_csv(DEST/f'{variant}_{split}_predictions.csv',index=False)
    results=pd.DataFrame(results)
    results.to_csv(DEST/'metrics.csv',index=False)
    pd.DataFrame(timing_rows).to_csv(DEST/'timing_metrics.csv',index=False)
    pd.DataFrame(group_rows).to_csv(DEST/'group_metrics.csv',index=False)
    if include_calendar:
        reference=pd.read_csv(OUT/'external/metrics.csv')
        reference=reference.loc[(reference.split=='test')&(reference.variant=='after')].set_index('model')
    else:
        reference=pd.read_csv(OUT/'sequence/all_test_metrics.csv').set_index('model')
    for row in results.loc[(results.split=='test')&(results.variant=='before')].itertuples():
        assert abs(row.MAE-float(reference.loc[row.model,'MAE']))<1e-4,(row.model,row.MAE)
    summary=[]
    for name in MODEL_LIST:
        row={'model':name}
        for split in ['validation','test']:
            for variant in ['before','after']:
                row[f'{split}_{variant}_MAE']=float(results.loc[(results.model==name)&(results.split==split)&(results.variant==variant),'MAE'].iloc[0])
            row[f'{split}_delta_MAE']=row[f'{split}_after_MAE']-row[f'{split}_before_MAE']
        summary.append(row)
    pd.DataFrame(summary).to_csv(DEST/'comparison.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    print('Validation-selected condition:',selected,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--without-calendar',action='store_true',help='외부 달력 자료 없이 미세먼지만 추가')
    args=parser.parse_args()
    main(include_calendar=not args.without_calendar)
