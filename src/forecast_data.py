"""Create strictly one-hour-ahead examples from the original UCI CSV."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/SeoulBikeData.csv'
OUT = ROOT / 'reports/forecast'
WEATHER = ['Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)',
           'Visibility (10m)', 'Dew point temperature(°C)',
           'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)']
NUMERIC = ['current_count', *WEATHER, 'target_hour_sin', 'target_hour_cos',
           'target_weekday', 'target_month']
CATEGORICAL = ['current_holiday', 'current_season', 'current_functioning']
FEATURES = NUMERIC + CATEGORICAL


def build_examples(raw):
    """Fail on ambiguous data rather than silently inventing labels."""
    df = raw.copy()
    missing = int(df.isna().sum().sum())
    if missing:
        raise ValueError(f'Missing source values: {missing}; inspect before training')
    df['input_time'] = (pd.to_datetime(df['Date'], format='%d/%m/%Y')
                        + pd.to_timedelta(df['Hour'], unit='h'))
    if df['input_time'].duplicated().any():
        raise ValueError('Duplicate hourly timestamps')
    df = df.sort_values('input_time').reset_index(drop=True)
    df['target_time'] = df['input_time'].shift(-1)
    df['target_count'] = df['Rented Bike Count'].shift(-1)
    df['current_count'] = df['Rented Bike Count']
    df['current_holiday'] = df['Holiday']
    df['current_season'] = df['Seasons']
    df['current_functioning'] = df['Functioning Day']
    # Preserve non-operating rows: removing them can make shift(-1) skip hours.
    consecutive = df['target_time'] - df['input_time'] == pd.Timedelta(hours=1)
    examples = df.loc[consecutive].copy()
    hour = examples['target_time'].dt.hour
    examples['target_hour_sin'] = np.sin(2 * np.pi * hour / 24)
    examples['target_hour_cos'] = np.cos(2 * np.pi * hour / 24)
    examples['target_weekday'] = examples['target_time'].dt.day_of_week
    examples['target_month'] = examples['target_time'].dt.month
    examples['target_count'] = examples['target_count'].astype(int)
    cols = ['input_time', 'target_time', *FEATURES, 'target_count']
    return examples[cols].reset_index(drop=True), {
        'source_rows': len(df), 'missing_cells': missing,
        'duplicate_timestamps': 0, 'non_operating_rows': int((df['Functioning Day'] == 'No').sum()),
        'unpaired_rows': int((~consecutive).sum()),
        'gap_pairs_excluded': int((~consecutive).sum()) - 1,
    }


def split_examples(examples):
    n = len(examples)
    a, b = int(n * .70), int(n * .85)
    if not 0 < a < b < n:
        raise ValueError('Not enough examples for three sets')
    return {'train': examples.iloc[:a].copy(),
            'validation': examples.iloc[a:b].copy(),
            'test': examples.iloc[b:].copy()}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    examples, audit = build_examples(pd.read_csv(RAW, encoding='latin1'))
    sets = split_examples(examples)
    audit['sha256'] = hashlib.sha256(RAW.read_bytes()).hexdigest()
    audit['examples'] = len(examples)
    (OUT / 'data_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    summary = []
    for name, part in sets.items():
        summary.append({'split': name, 'rows': len(part),
                        'input_start': part.input_time.min(), 'input_end': part.input_time.max(),
                        'target_start': part.target_time.min(), 'target_end': part.target_time.max()})
    pd.DataFrame(summary).to_csv(OUT / 'split_summary.csv', index=False)
    examples.head(24).to_csv(OUT / 'example_pairs.csv', index=False)
    return sets


if __name__ == '__main__':
    prepare()
    print((OUT / 'split_summary.csv').read_text(encoding='utf-8'))
