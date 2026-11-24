"""24-hour sequence forecasting. Training/checkpoint selection use train/validation only."""
import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from forecast_data import OUT, NUMERIC, CATEGORICAL, FEATURES, prepare
from solution import scores

SEQ_OUT = OUT / 'sequence'
CONFIG = dict(window=24, hidden=32, batch_size=128, max_epochs=40,
              patience=8, learning_rate=0.001, seed=42, threads=2)


def window_indices(times, length=24):
    times = pd.DatetimeIndex(times).as_unit('ns')
    if length < 1 or not times.is_monotonic_increasing or times.has_duplicates:
        raise ValueError('Window length and chronological timestamps must be valid')
    ends = np.arange(length - 1, len(times))
    indices = ends[:, None] - np.arange(length - 1, -1, -1)[None, :]
    if len(indices):
        continuous = np.all(np.diff(times.asi8[indices], axis=1) == pd.Timedelta(hours=1).value, axis=1)
        return indices[continuous]
    return indices


def sequence_data(sets, numeric=None, categorical=None):
    numeric = NUMERIC if numeric is None else numeric
    categorical = CATEGORICAL if categorical is None else categorical
    features = numeric + categorical
    full = pd.concat(sets.values(), ignore_index=True)
    preprocess = ColumnTransformer([
        ('numeric', StandardScaler(), numeric),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)])
    preprocess.fit(sets['train'][features])
    transformed = preprocess.transform(full[features]).astype('float32')
    windows = window_indices(full.input_time, CONFIG['window'])
    ends = windows[:, -1]
    # Fit target scaling on training targets only.
    mean = float(sets['train'].target_count.mean())
    std = float(sets['train'].target_count.std(ddof=0))
    batches = {}
    for key, part in sets.items():
        in_split = full.iloc[ends].target_time.isin(part.target_time).to_numpy()
        w = windows[in_split]
        frame = full.iloc[w[:, -1]].reset_index(drop=True)
        batches[key] = (transformed[w], ((frame.target_count.to_numpy() - mean) / std).astype('float32'), frame)
    return batches, mean, std


class RNNForecast(nn.Module):
    def __init__(self, inputs, hidden=32):
        super().__init__()
        self.rnn = nn.RNN(inputs, hidden, batch_first=True, nonlinearity='tanh')
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        outputs, _ = self.rnn(x)
        return self.head(outputs[:, -1]).squeeze(-1)


class TransformerForecast(nn.Module):
    def __init__(self, inputs, hidden=32, length=24):
        super().__init__()
        self.project = nn.Linear(inputs, hidden)
        position = torch.arange(length).float().unsqueeze(1)
        scale = torch.exp(torch.arange(0, hidden, 2).float() * (-math.log(10000.) / hidden))
        encoding = torch.zeros(1, length, hidden)
        encoding[0, :, 0::2] = torch.sin(position * scale)
        encoding[0, :, 1::2] = torch.cos(position * scale)
        self.register_buffer('position', encoding)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=64,
                                           dropout=.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        # All 24 tokens are already observed. There are no future tokens to mask.
        encoded = self.encoder(self.project(x) + self.position[:, :x.shape[1]])
        return self.head(encoded[:, -1]).squeeze(-1)


def infer(model, x, mean, std):
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(torch.from_numpy(x[i:i+256])) for i in range(0, len(x), 256)])
    return np.maximum(0, pred.numpy() * std + mean)


def train_model(name, model, batches, mean, std, output_dir=SEQ_OUT):
    # This function intentionally has access only to train/validation batches.
    x, y, _ = batches['train']
    val_x, _, val_frame = batches['validation']
    generator = torch.Generator().manual_seed(CONFIG['seed'])
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                        batch_size=CONFIG['batch_size'], shuffle=True, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.MSELoss()
    best_mae, best_state, best_epoch, stale = float('inf'), None, 0, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, CONFIG['max_epochs'] + 1):
        model.train()
        total = 0.
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(bx)
        mae = scores(val_frame.target_count, infer(model, val_x, mean, std))['MAE']
        history.append({'epoch': epoch, 'train_scaled_MSE': total / len(x), 'validation_MAE': mae})
        print(f'{name} epoch {epoch:02d}: val MAE={mae:.3f}', flush=True)
        if mae < best_mae:
            best_mae, best_epoch, stale = mae, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= CONFIG['patience']:
            break
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_dir / f'{name}_weights.pt')
    pd.DataFrame(history).to_csv(output_dir / f'{name}_learning_curve.csv', index=False)
    return {'model': name, 'best_epoch': best_epoch, 'epochs_run': len(history),
            'validation_MAE': best_mae, 'training_seconds': time.perf_counter() - started}


def run(name):
    SEQ_OUT.mkdir(parents=True, exist_ok=True)
    random.seed(CONFIG['seed']); np.random.seed(CONFIG['seed']); torch.manual_seed(CONFIG['seed'])
    torch.set_num_threads(CONFIG['threads'])
    torch.use_deterministic_algorithms(True)
    sets = prepare()
    batches, mean, std = sequence_data(sets)
    inputs = batches['train'][0].shape[2]
    model = (RNNForecast(inputs, CONFIG['hidden']) if name == 'RNN'
             else TransformerForecast(inputs, CONFIG['hidden'], CONFIG['window']))
    # Pass no test data to optimization/checkpoint selection.
    metadata = train_model(name, model, {k: batches[k] for k in ['train', 'validation']}, mean, std)
    metadata.update(config=CONFIG, torch_version=torch.__version__, target_mean=mean,
                    target_std=std, input_features=FEATURES,
                    parameters=sum(p.numel() for p in model.parameters()),
                    samples={k: len(v[0]) for k, v in batches.items()})
    # Freeze metadata and selected weights BEFORE evaluating the test set.
    (SEQ_OUT / f'{name}_training.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    results = []
    for split in ['validation', 'test']:
        x, _, frame = batches[split]
        pred = infer(model, x, mean, std)
        frame = frame[['input_time', 'target_time', 'target_count']].copy()
        frame[name] = pred
        frame.to_csv(SEQ_OUT / f'{name}_{split}_predictions.csv', index=False)
        results.append({'split': split, 'model': name, **scores(frame.target_count, pred)})
    pd.DataFrame(results).to_csv(SEQ_OUT / f'{name}_metrics.csv', index=False)
    print(pd.DataFrame(results).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['RNN', 'Transformer', 'all'], default='all')
    args = parser.parse_args()
    for name in (['RNN', 'Transformer'] if args.model == 'all' else [args.model]):
        run(name)


if __name__ == '__main__':
    main()
