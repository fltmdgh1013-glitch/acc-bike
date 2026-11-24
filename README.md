# 서울 자전거 다음 1시간 대여량 예측 — 실행 안내

현재 시간의 대여량·기상정보로 다음 한 시간의 대여량을 예측하는 수업 프로젝트입니다. 공개 프로젝트를 바탕으로 AI의 도움을 받아 수정·확장했습니다.

## 코드 위치

- `src/`: 학습·실험·평가 코드 34개
- `tests/`: 검사 코드 11개
- `requirements-forecast.txt`: 이번 예측 실험의 설치 목록
- `data/raw/SeoulBikeData.csv`: 기본 실험에 사용하는 원본

## 기본 실험 재현 (Windows / Python 3.12 권장)

저장소를 다운로드하고 압축을 푼 다음, 이 README가 있는 폴더에서 PowerShell을 열어 실행합니다.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-forecast.txt
.\.venv\Scripts\python.exe src/run_forecast.py
```

직전 대여량 기준·선형회귀·그래디언트 부스팅·RNN·Transformer를 비교합니다. 새 결과는 `reports/forecast/sequence/all_test_metrics.csv`에서 확인합니다. 기존 RNN 시험 MAE는 약 97.04대이며 실행 환경에 따라 작은 차이가 날 수 있습니다.

학습·검증·시험은 시간순으로 분리했고, 모델 선택은 검증 MAE로 합니다. 신경망은 최근 24시간을 입력합니다. 최상위의 `solution.py`와 `full_workflow.py`는 업로드 과정에 남은 중복 파일이므로 실행은 위의 `src/run_forecast.py`를 사용합니다. 기존 `reports/` 최상위 파일은 원래 프로젝트의 결과이며, 이번 과제 결과는 실행 후 `reports/forecast/` 아래에 생성됩니다.

## 외부요인 실험

외부 입력자료·근거 원본·기존 학습 가중치·예측·PDF는 별도로 전달한 `재현용.zip`에 있습니다. 전체 실험을 재현하려면 그 ZIP의 `03_실행코드와전체자료` 폴더와 `실행방법.txt`를 사용하세요. 이 GitHub 저장소만으로 모든 외부요인 결과가 준비돼 있는 것은 아닙니다.

- 달력·미세먼지: `src/run_forecast.py --with-air`
- 일출일몰: `src/run_solar_experiment.py`
- 일출일몰 혼합: `src/run_solar_mix_experiment.py`
- 불쾌지수: `src/prepare_discomfort.py` 후 `src/run_discomfort_experiment.py`
- 강수예보는 과거 자료 미확보로 보류했습니다.

## 출처

기존 코드: [Menagasweety/seoul-bike-demand-regression](https://github.com/Menagasweety/seoul-bike-demand-regression). 데이터: UCI Seoul Bike Sharing Demand, DOI 10.24432/C5F62R, CC BY 4.0. 기존 회귀 실험을 다음 한 시간 예측과 검증 선택에 맞게 바꾸고, 신경망·외부요인·출퇴근 분석을 확장했습니다.

아래는 가져온 공개 프로젝트의 원래 설명입니다. 그 내용의 80/20 분리 및 실행 명령은 기존 분석을 설명하며, 이번 과제의 실행은 위 안내를 따릅니다. 로컬 문제별 커밋 전체를 전송한 것은 아닙니다.

---

# Seoul Bike Sharing Demand Regression

A complete, beginner-friendly regression project that predicts the number of public bikes rented in Seoul each hour. It includes a real-world dataset, Excel EDA, a reproducible Python solution, model comparison, and feature-importance analysis.

## Why this dataset?

This is a good portfolio dataset because the target is intuitive, the data combines weather and time variables, and the problem is genuinely a regression task. It is less overused than house-price or salary datasets but remains easy to explain to a reviewer. The dataset contains 8,760 hourly records and has no missing values according to the official UCI documentation.

## Project question

> Given the hour, weather, season, holiday status, and system operating status, how many bikes will be rented?

## Repository structure

- `data/raw/SeoulBikeData.csv` — original UCI file, preserved for traceability.
- `data/processed/seoul_bike_clean.csv` — cleaned data with calendar and cyclic-hour features.
- `data/processed/test_predictions.csv` — held-out predictions from the best model selected by RMSE.
- `reports/seoul_bike_eda.xlsx` — Excel-first EDA workbook with formulas, summaries, charts, and model results.
- `reports/model_metrics.csv` — model comparison results.
- `reports/feature_importance.csv` — permutation importance on the held-out test set.
- `notebooks/seoul_bike_regression.ipynb` — notebook version of the workflow.
- `src/solution.py` — command-line reproducible solution.

## Excel EDA guide

Open `reports/seoul_bike_eda.xlsx` and review the sheets in this order: `README`, `Data_Dictionary`, `Summary`, `Numeric_Summary`, `Category_Summary`, `Hourly_Analysis`, `Monthly_Analysis`, `Model_Results`, and `Feature_Importance`. In Excel, press **Formulas → Calculate Now** if formula values do not refresh automatically.

Recommended EDA questions:

1. Which hours have the highest average demand?
2. Does demand differ by season, holiday, and functioning day?
3. Which weather variables have the strongest relationship with demand?
4. Are there unusual outliers during extreme rainfall or snowfall?

## Modeling choices

The solution uses an 80/20 chronological split. The final 20% is treated as future-like test data, which is more realistic than randomly shuffling hourly observations. It compares Linear Regression, Gradient Boosting, and Random Forest. The primary selection metric is RMSE; MAE and R² are reported alongside it.

## Run locally

```bash
pip install -r requirements.txt
python src/solution.py
```

To run the notebook:

```bash
jupyter notebook notebooks/seoul_bike_regression.ipynb
```

## Dataset attribution

Seoul Bike Sharing Demand. (2020). UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5F62R. Licensed under CC BY 4.0.

The raw file is included for educational and reproducibility purposes. Keep this attribution when publishing the repository.

## Deployment app

Run the Streamlit app from the project root:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app provides an interactive prediction form, model comparison, dataset audit indicators, and simple EDA charts. It trains the selected regularized pipeline on all available cleaned data for prediction while displaying evaluation metrics from the chronological holdout.

## Full workflow

Run `python src/full_workflow.py` to regenerate the audit file, EDA CSV summaries, and the baseline/Ridge/Lasso/ElasticNet comparison.
