"""One-hour forecast: select by validation MAE, evaluate frozen models on test."""
import json
import platform
from importlib.metadata import version
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from forecast_data import OUT, FEATURES, NUMERIC, CATEGORICAL, prepare


def make_pipeline(estimator, numeric=None, categorical=None):
    numeric = NUMERIC if numeric is None else numeric
    categorical = CATEGORICAL if categorical is None else categorical
    preprocess = ColumnTransformer([
        ('numeric', StandardScaler(), numeric),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
    ])
    return Pipeline([('preprocess', preprocess), ('model', estimator)])


def scores(actual, predicted):
    return {'MAE': float(mean_absolute_error(actual, predicted)),
            'RMSE': float(np.sqrt(mean_squared_error(actual, predicted))),
            'R2': float(r2_score(actual, predicted)) if len(actual) > 1 else None}


def select_model(validation_rows):
    return min(validation_rows, key=lambda row: (row['MAE'], row['model']))['model']


def predict(model, frame):
    if model is None:
        return frame['current_count'].to_numpy()
    return np.maximum(0, model.predict(frame[FEATURES]))


def main():
    sets = prepare()
    train, validation, test = (sets[key] for key in ('train', 'validation', 'test'))
    # Fixed candidates; never tune on the final test set.
    models = {
        'Persistence': None,
        'LinearRegression': make_pipeline(LinearRegression()),
        'GradientBoosting': make_pipeline(GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=.05, random_state=42)),
    }
    validation_rows = []
    for name, model in models.items():
        if model is not None:
            model.fit(train[FEATURES], train['target_count'])
        validation_rows.append({'model': name, **scores(validation.target_count, predict(model, validation))})
    selected = select_model(validation_rows)
    pd.DataFrame(validation_rows).to_csv(OUT / 'validation_metrics.csv', index=False)
    # Save selection before making test predictions. No train+validation refit.
    decision = {'selected_model': selected, 'selection_metric': 'validation MAE',
                'refit_on_validation': False, 'random_seed': 42,
                'protocol': 'rolling one-hour ahead; current observations available each hour',
                'features': FEATURES, 'python': platform.python_version(),
                'packages': {p: version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'matplotlib']}}
    (OUT / 'selection.json').write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding='utf-8')
    predictions = test[['input_time', 'target_time', 'target_count', 'current_count',
                        'Rainfall(mm)', 'current_functioning']].copy()
    test_rows = []
    for name, model in models.items():
        predictions[name] = predict(model, test)
        test_rows.append({'model': name, 'selected_by_validation': name == selected,
                          **scores(test.target_count, predictions[name])})
    pd.DataFrame(test_rows).to_csv(OUT / 'test_metrics.csv', index=False)
    predictions.to_csv(OUT / 'test_predictions.csv', index=False)
    print('VALIDATION (model selection)')
    print(pd.DataFrame(validation_rows).to_string(index=False))
    print(f'Frozen selection: {selected}')
    print('TEST (report only)')
    print(pd.DataFrame(test_rows).to_string(index=False))


if __name__ == '__main__':
    main()
