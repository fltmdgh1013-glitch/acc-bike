"""통제된 외부 달력 정보 추가 전후 실험. 시험 지표로 재튜닝하지 않는다."""
import json
import random
import hashlib
import platform
from importlib.metadata import version
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from forecast_data import OUT, RAW, NUMERIC, CATEGORICAL, prepare
from external_features import EXTERNAL_FILE, EXTERNAL_COLUMNS, enrich_sets
from solution import make_pipeline, scores
from sequence_models import CONFIG, sequence_data, RNNForecast, TransformerForecast, train_model, infer
from evaluate_timing import binary_scores, timing_masks

DEST = OUT / 'external'
MODEL_LIST = ['Persistence','LinearRegression','GradientBoosting','RNN','Transformer']


def seed_everything():
    random.seed(CONFIG['seed']); np.random.seed(CONFIG['seed']); torch.manual_seed(CONFIG['seed'])
    torch.set_num_threads(CONFIG['threads'])
    torch.use_deterministic_algorithms(True)


def predict_frozen(item, split, sets):
    variant, name, model, features, batches, mean, std = item
    if name == 'Persistence':
        return sets[variant][split].current_count.to_numpy()
    if batches is not None:
        return infer(model, batches[split][0], mean, std)
    return np.maximum(0, model.predict(sets[variant][split][features]))


def main(feature_set='calendar'):
    if feature_set=='solar':
        from solar_features import SOLAR_FILE, SOLAR_COLUMNS, enrich_solar_sets
        external_file, extra_columns, enrich = SOLAR_FILE, SOLAR_COLUMNS, enrich_solar_sets
        DEST=OUT/'solar'
        description='four known target-time solar features (no extra calendar or air quality)'
        after_label='기존 정보와 일출·일몰 정보'
    elif feature_set=='rain_forecast':
        from rain_forecast_features import RAIN_FORECAST_FILE,RAIN_FORECAST_COLUMNS,enrich_rain_forecast_sets
        external_file,extra_columns,enrich=RAIN_FORECAST_FILE,RAIN_FORECAST_COLUMNS,enrich_rain_forecast_sets
        DEST=OUT/'rain_forecast'
        description='Seoul unique-grid precipitation forecast fractions available before input time; no extra calendar or air quality'
        after_label='기존 정보와 사전 강수예보'
    elif feature_set=='calendar':
        external_file, extra_columns, enrich = EXTERNAL_FILE, EXTERNAL_COLUMNS, enrich_sets
        DEST=OUT/'external'
        description='eight known target-date calendar features'
        after_label='기존 정보와 외부 달력 정보'
    elif feature_set=='discomfort':
        from discomfort_features import DISCOMFORT_FILE,DISCOMFORT_COLUMNS,enrich_discomfort_sets
        external_file,extra_columns,enrich=DISCOMFORT_FILE,DISCOMFORT_COLUMNS,enrich_discomfort_sets
        DEST=OUT/'discomfort'
        description='one derived discomfort index from current temperature and humidity; no new external observation'
        after_label='기존 정보와 현재 불쾌지수'
    else:
        raise ValueError(feature_set)
    DEST.mkdir(parents=True, exist_ok=True)
    original = prepare()
    if not external_file.exists():
        raise FileNotFoundError(f'실제 외부자료가 필요합니다: {external_file}. 미수집 자료로 실험하지 않습니다.')
    enriched = enrich(original, pd.read_csv(external_file))
    if feature_set in ['rain_forecast','discomfort']:
        pd.concat([frame.assign(split=split) for split,frame in enriched.items()],ignore_index=True).to_csv(
            DEST/'input_examples.csv',index=False)
    sets = {'before':original, 'after':enriched}
    # Written before fitting/evaluation: this run varies external inputs, not model settings.
    protocol = {
        'experiment':'existing features versus '+description,
        'variants':{'before':'기존 정보만', 'after':after_label},
        'extra_features':extra_columns, 'join_key':'date of target_time (KST)',
        'raw_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),
        'external_sha256':hashlib.sha256(external_file.read_bytes()).hexdigest(),
        'seed':42, 'neural_config':CONFIG,
        'linear':'LinearRegression defaults; preprocessing fitted on train only',
        'boosting':dict(n_estimators=200, max_depth=3, learning_rate=.05, random_state=42),
        'selection':'minimum validation MAE among model/variant pairs, before test scoring',
        'classification_threshold':400, 'refit_on_validation':False,
        'target_calendar_only':feature_set=='calendar', 'known_target_time_features_only':feature_set in ['calendar','solar'],
        'future_actual_weather_or_operating_status_used':False,
        'split_rows':{k:len(v) for k,v in original.items()},
        'neural_train_rows':len(original['train'])-23,
        'python':platform.python_version(),
        'packages':{p:version(p) for p in ['numpy','pandas','scikit-learn','torch']},
        'scope':'Exploratory extension after earlier test results were seen. No test-driven setting changes. One seed only.'}
    if feature_set=='rain_forecast':
        from rain_forecast_features import PUBLICATION_LAG_MINUTES,SEOUL_GRID_FILE
        protocol.update(forecast_publication_lag_minutes=PUBLICATION_LAG_MINUTES,
                        forecast_availability_assumed_not_reconstructed=True,
                        forecast_join='same target_time, latest available_time <= input_time, no fill for missing forecasts',
                        forecast_features='fractions of Seoul target unique grids with precipitation and snow forecasts; not probabilities or area shares',
                        seoul_grid_sha256=hashlib.sha256(SEOUL_GRID_FILE.read_bytes()).hexdigest(),
                        spatial_aggregation='equal weights after grid deduplication; require complete fixed grid set at each selected issue/target')
    elif feature_set=='discomfort':
        protocol.update(join_key='input_time (KST), never target_time actual weather',
                        formula='1.8*T - 0.55*(1-H/100)*(1.8*T-26) + 32; T Celsius, H percent',
                        derived_feature_not_independent_external_observation=True,
                        source_metadata='data/external/discomfort/SOURCE.json',
                        groups='fixed descriptive current DI bands <68, [68,75), [75,80), >=80; not tuned on test')
    (DEST/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    samples = enriched['train'].loc[enriched['train'].target_time.between('2017-12-24 23:00','2017-12-25 02:00')]
    samples[['input_time','target_time','target_count',*extra_columns]].to_csv(DEST/'join_examples.csv',index=False)
    validation_rows, frozen, metadata = [], [], []
    for variant, split_sets in sets.items():
        numeric = NUMERIC + (extra_columns if variant == 'after' else [])
        features = numeric + CATEGORICAL
        batches, mean, std = sequence_data(split_sets, numeric=numeric)
        for name in MODEL_LIST:
            seed_everything()
            if name == 'Persistence':
                model, model_batches = None, None
            elif name in ['LinearRegression','GradientBoosting']:
                estimator = (LinearRegression() if name == 'LinearRegression' else GradientBoostingRegressor(
                    n_estimators=200,max_depth=3,learning_rate=.05,random_state=42))
                model = make_pipeline(estimator,numeric=numeric)
                model.fit(split_sets['train'][features],split_sets['train'].target_count)
                model_batches = None
            else:
                inputs = batches['train'][0].shape[2]
                model = (RNNForecast(inputs,32) if name == 'RNN' else TransformerForecast(inputs,32,24))
                model_batches = batches
                details = train_model(name,model,{k:batches[k] for k in ['train','validation']},mean,std,
                                      output_dir=DEST/variant)
                details.update(variant=variant,input_features=features,
                               target_mean=mean,target_std=std,
                               parameters=sum(p.numel() for p in model.parameters()),
                               samples={k:len(v[0]) for k,v in batches.items()})
                metadata.append(details)
            item = (variant,name,model,features,model_batches,mean,std)
            frozen.append(item)
            prediction = predict_frozen(item,'validation',sets)
            validation_rows.append({'variant':variant,'model':name,
                                    **scores(split_sets['validation'].target_count,prediction)})
            print(variant,name,'validation MAE:',validation_rows[-1]['MAE'],flush=True)
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(DEST/'validation_metrics.csv',index=False)
    selected = min(validation_rows,key=lambda r:(r['MAE'],r['variant'],r['model']))
    # Freeze the choice before predicting any test targets in this experiment.
    (DEST/'selection.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    (DEST/'training_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    results, timing_rows, group_rows = [], [], []
    for split in ['validation','test']:
        frame = enriched[split].reset_index(drop=True)
        masks = timing_masks(frame.target_time)
        if feature_set=='calendar':
            masks['calendar_rest_day'] = frame['휴일여부_주말포함'].to_numpy() == 1
            masks['calendar_work_day'] = ~masks['calendar_rest_day']
        elif feature_set=='solar':
            masks['solar_daylight']=frame['주간여부'].to_numpy()==1
            masks['solar_dark']=~masks['solar_daylight']
            masks['solar_near_boundary']=frame['일출일몰_가까운경계까지_분'].to_numpy()<=60
        elif feature_set=='discomfort':
            di=frame['현재_불쾌지수'].to_numpy()
            masks['di_below68']=di<68
            masks['di_68_75']=(di>=68)&(di<75)
            masks['di_75_80']=(di>=75)&(di<80)
            masks['di_80_plus']=di>=80
        else:
            masks['forecast_precipitation']=frame['서울_강수예보격자비율'].to_numpy()>0
            masks['forecast_dry']=~masks['forecast_precipitation']
        predictions = {v:frame[['input_time','target_time','target_count','current_count',*extra_columns]].copy() for v in sets}
        for item in frozen:
            variant,name = item[:2]
            pred = predict_frozen(item,split,sets)
            if item[4] is not None:
                pd.testing.assert_series_equal(item[4][split][2].target_time,frame.target_time)
            predictions[variant][name] = pred
            results.append({'split':split,'variant':variant,'model':name,**scores(frame.target_count,pred)})
            for group,mask in masks.items():
                actual,estimated = frame.target_count.to_numpy()[mask],pred[mask]
                timing_rows.append({'split':split,'variant':variant,'model':name,'group':group,
                                    **binary_scores(actual,estimated)})
                group_rows.append({'split':split,'variant':variant,'model':name,'group':group,
                                   'rows':int(mask.sum()),**(scores(actual,estimated) if mask.any() else {'MAE':float('nan'),'RMSE':float('nan'),'R2':None})})
        for variant,pred in predictions.items():
            pred.to_csv(DEST/f'{variant}_{split}_predictions.csv',index=False)
    results = pd.DataFrame(results)
    results.to_csv(DEST/'metrics.csv',index=False)
    pd.DataFrame(timing_rows).to_csv(DEST/'timing_metrics.csv',index=False)
    pd.DataFrame(group_rows).to_csv(DEST/'group_metrics.csv',index=False)
    # Check the unmodified condition reproduces the previously published results.
    old = pd.read_csv(OUT/'sequence/all_test_metrics.csv').set_index('model')
    for row in results.loc[(results.split=='test') & (results.variant=='before')].itertuples():
        assert abs(row.MAE-float(old.loc[row.model,'MAE'])) < 1e-4, (row.model,row.MAE,old.loc[row.model,'MAE'])
    summary = []
    for name in MODEL_LIST:
        r={'model':name}
        for split in ['validation','test']:
            for variant in ['before','after']:
                r[f'{split}_{variant}_MAE'] = float(results.loc[(results.model==name)&(results.split==split)&(results.variant==variant),'MAE'].iloc[0])
            r[f'{split}_delta_MAE'] = r[f'{split}_after_MAE']-r[f'{split}_before_MAE']
        summary.append(r)
    pd.DataFrame(summary).to_csv(DEST/'comparison.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    print('Validation-selected variant/model:',selected,flush=True)


if __name__ == '__main__':
    main()
