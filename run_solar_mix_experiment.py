"""일출·일몰과 달력/미세먼지의 세 조합을 고정 설정으로 비교한다."""
import json
import hashlib
import platform
from importlib.metadata import version
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from forecast_data import ROOT,OUT,RAW,NUMERIC,CATEGORICAL,prepare
from solar_features import SOLAR_FILE,SOLAR_COLUMNS,enrich_solar_sets
from external_features import EXTERNAL_FILE,EXTERNAL_COLUMNS,enrich_sets
from air_quality_features import AIR_FILE,AIR_COLUMNS,enrich_air_sets
from solution import make_pipeline,scores
from sequence_models import CONFIG,sequence_data,RNNForecast,TransformerForecast,train_model,infer
from run_external_experiment import seed_everything,MODEL_LIST
from evaluate_timing import timing_masks,binary_scores

DEST=OUT/'solar_mix'
VARIANTS={
    'solar_calendar':{'label':'일출일몰+달력','calendar':True,'air':False,'control':'external'},
    'solar_air':{'label':'일출일몰+미세먼지','calendar':False,'air':True,'control':'air_quality_only'},
    'solar_calendar_air':{'label':'일출일몰+달력+미세먼지','calendar':True,'air':True,'control':'air_quality'},
}
CACHED={'base':('기본 정보만','sequence'), 'calendar':('달력','external'),
        'air':('미세먼지','air_quality_only'), 'calendar_air':('달력+미세먼지','air_quality'),
        'solar':('일출일몰','solar')}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_mix_sets(original,key):
    spec=VARIANTS[key]
    sets=enrich_solar_sets(original,pd.read_csv(SOLAR_FILE))
    extra=SOLAR_COLUMNS.copy();audit={}
    if spec['calendar']:
        sets=enrich_sets(sets,pd.read_csv(EXTERNAL_FILE));extra+=EXTERNAL_COLUMNS
    if spec['air']:
        sets,audit=enrich_air_sets(sets,pd.read_csv(AIR_FILE));extra+=AIR_COLUMNS
    for split,frame in original.items():
        pd.testing.assert_frame_equal(frame,sets[split][frame.columns])
        assert set(sets[split])-set(frame)==set(extra)
        assert sets[split][extra].notna().all().all()
    return sets,extra,audit


def cached_metrics(folder,split):
    if folder=='sequence':return pd.read_csv(OUT/folder/f'all_{split}_metrics.csv')
    df=pd.read_csv(OUT/folder/'metrics.csv')
    return df[(df.split==split)&(df.variant=='after')][['model','MAE','RMSE','R2']].copy()


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    original=prepare()
    sets={};extras={};audits={}
    for key in VARIANTS:
        sets[key],extras[key],audits[key]=build_mix_sets(original,key)
        folder=DEST/key;folder.mkdir(parents=True,exist_ok=True)
        pd.concat([frame.assign(split=split) for split,frame in sets[key].items()],ignore_index=True).to_csv(
            folder/'input_examples.csv',index=False)
    protocol={'조건':{key:{**spec,'features':extras[key]} for key,spec in VARIANTS.items()},
              '신규조합만학습':True,
              'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [RAW,SOLAR_FILE,EXTERNAL_FILE,AIR_FILE]},
              'seed':42,'neural_config':CONFIG,'boosting':dict(n_estimators=200,max_depth=3,learning_rate=.05,random_state=42),
              'rows':{k:len(v) for k,v in original.items()},'neural_train_rows':len(original['train'])-23,
              'solar_time':'target_time KST 정시 순간, 미리 계산 가능한 일출·일몰',
              'air_time':'input_time보다 한 시간 앞서 끝난 관측; 실제 게시 지연 미복원',
              'imputation_train_only':audits,'future_actual_weather_or_operating_status_used':False,
              'selection':'기존 5조건과 신규 3조건의 검증 MAE로만 선택, 신규 시험 예측 전에 기록',
              'scope':'기존 시험을 확인한 뒤의 탐색적 단일 시드 확장. 새 기간 독립 평가 아님. 시험 점수로 재튜닝하지 않음.',
              'cached_conditions':CACHED,'python':platform.python_version(),
              'packages':{p:version(p) for p in ['numpy','pandas','scikit-learn','torch']}}
    (DEST/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    valrows=[];frozen=[];training=[]
    for key,(label,folder) in CACHED.items():
        for r in cached_metrics(folder,'validation').to_dict('records'):
            valrows.append({'condition':key,'label':label,**r,'reused':True})
    for key,spec in VARIANTS.items():
        folder=DEST/key;folder.mkdir(parents=True,exist_ok=True)
        numeric=NUMERIC+extras[key];features=numeric+CATEGORICAL
        batches,mean,std=sequence_data(sets[key],numeric=numeric)
        predictions=sets[key]['validation'][['input_time','target_time','target_count','current_count',*extras[key]]].reset_index(drop=True).copy()
        for name in MODEL_LIST:
            seed_everything()
            if name=='Persistence':
                model=None
                pred=sets[key]['validation'].current_count.to_numpy()
            elif name in ['LinearRegression','GradientBoosting']:
                estimator=LinearRegression() if name=='LinearRegression' else GradientBoostingRegressor(
                    n_estimators=200,max_depth=3,learning_rate=.05,random_state=42)
                model=make_pipeline(estimator,numeric=numeric)
                model.fit(sets[key]['train'][features],sets[key]['train'].target_count)
                pred=np.maximum(0,model.predict(sets[key]['validation'][features]))
            else:
                inputs=batches['train'][0].shape[2]
                model=RNNForecast(inputs,32) if name=='RNN' else TransformerForecast(inputs,32,24)
                detail=train_model(name,model,{k:batches[k] for k in ['train','validation']},mean,std,output_dir=folder)
                detail.update(condition=key,features=features,target_mean=mean,target_std=std,
                              parameters=sum(p.numel() for p in model.parameters()),samples={k:len(b[0]) for k,b in batches.items()})
                training.append(detail)
                pred=infer(model,batches['validation'][0],mean,std)
                pd.testing.assert_series_equal(batches['validation'][2].target_time,predictions.target_time)
            predictions[name]=pred
            result={'condition':key,'label':spec['label'],'model':name,**scores(predictions.target_count,pred),'reused':False}
            valrows.append(result)
            frozen.append((key,name,model,features,batches,mean,std))
            print(key,name,'validation MAE',result['MAE'],flush=True)
        predictions.to_csv(folder/'validation_predictions.csv',index=False)
        pd.DataFrame([r for r in valrows if r['condition']==key]).to_csv(folder/'validation_metrics.csv',index=False)
        (DEST/'training_metadata.json').write_text(json.dumps(training,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(valrows).to_csv(DEST/'validation_metrics.csv',index=False)
    selected=min(valrows,key=lambda r:(r['MAE'],r['condition'],r['model']))
    (DEST/'selection.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf-8')
    # All weights and the choice are now frozen. Only now evaluate new test predictions.
    allrows=[{'split':'validation',**r} for r in valrows]
    timingrows=[];grouprows=[];cache_hashes={}
    solar=sets['solar_calendar']['test']
    for split in ['validation','test']:
        for key,(label,folder) in CACHED.items():
            if split=='test':
                for r in cached_metrics(folder,split).to_dict('records'):
                    allrows.append({'split':split,'condition':key,'label':label,**r,'reused':True})
                path=OUT/folder/('all_test_predictions.csv' if folder=='sequence' else 'after_test_predictions.csv')
                frame=pd.read_csv(path,parse_dates=['input_time','target_time'])
                cache_hashes[str(path.relative_to(ROOT))]=digest(path)
            else:
                # Only new validation time groups are needed; cached validation metrics above are sufficient.
                continue
            pd.testing.assert_frame_equal(original[split][['input_time','target_time','target_count','current_count']].reset_index(drop=True),
                                          frame[['input_time','target_time','target_count','current_count']],check_dtype=False)
            masks=timing_masks(frame.target_time)
            solarframe=solar.reset_index(drop=True)
            masks.update(solar_daylight=solarframe['주간여부'].to_numpy()==1,
                         solar_dark=solarframe['주간여부'].to_numpy()==0,
                         solar_near_boundary=solarframe[SOLAR_COLUMNS[-1]].to_numpy()<=60)
            for name in MODEL_LIST:
                expected=cached_metrics(folder,split).set_index('model').loc[name,'MAE']
                assert abs(scores(frame.target_count,frame[name])['MAE']-expected)<1e-4
                append_groups(timingrows,grouprows,key,split,name,frame,frame[name].to_numpy(),masks)
        for key,spec in VARIANTS.items():
            frame=sets[key][split][['input_time','target_time','target_count','current_count',*extras[key]]].reset_index(drop=True).copy()
            masks=timing_masks(frame.target_time)
            masks.update(solar_daylight=frame['주간여부'].to_numpy()==1,
                         solar_dark=frame['주간여부'].to_numpy()==0,
                         solar_near_boundary=frame[SOLAR_COLUMNS[-1]].to_numpy()<=60)
            for condition,name,model,features,batches,mean,std in frozen:
                if condition!=key:continue
                if name=='Persistence':pred=frame.current_count.to_numpy()
                elif name in ['RNN','Transformer']:
                    pred=infer(model,batches[split][0],mean,std)
                    pd.testing.assert_series_equal(batches[split][2].target_time,frame.target_time)
                else:pred=np.maximum(0,model.predict(sets[key][split][features]))
                frame[name]=pred
                if split=='test':allrows.append({'split':split,'condition':key,'label':spec['label'],'model':name,**scores(frame.target_count,pred),'reused':False})
                append_groups(timingrows,grouprows,key,split,name,frame,pred,masks)
            frame.to_csv(DEST/key/f'{split}_predictions.csv',index=False)
    result=pd.DataFrame(allrows)
    result.to_csv(DEST/'metrics.csv',index=False)
    pd.DataFrame(timingrows).to_csv(DEST/'timing_metrics.csv',index=False)
    pd.DataFrame(grouprows).to_csv(DEST/'group_metrics.csv',index=False)
    (DEST/'cached_prediction_hashes.json').write_text(json.dumps(cache_hashes,ensure_ascii=False,indent=2),encoding='utf-8')
    print(result[result.split=='test'].pivot(index='label',columns='model',values='MAE').to_string(),flush=True)
    print('Validation selected:',selected,flush=True)


def append_groups(timingrows,grouprows,key,split,name,frame,pred,masks):
    for group,mask in masks.items():
        actual,estimated=frame.target_count.to_numpy()[mask],pred[mask]
        timingrows.append({'condition':key,'split':split,'model':name,'group':group,**binary_scores(actual,estimated)})
        grouprows.append({'condition':key,'split':split,'model':name,'group':group,'rows':int(mask.sum()),**scores(actual,estimated)})


if __name__=='__main__':main()
