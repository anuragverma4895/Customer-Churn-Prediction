# Customer Churn Prediction & Analytics Dashboard

A Flask and scikit-learn project that predicts telecom customer churn using the real IBM Telco Customer Churn dataset. The web dashboard shows dataset KPIs, model evaluation, feature importance, ROC analysis, confusion matrix, and a live customer-risk prediction form.

## Real Dataset

This project uses the public IBM Telco Customer Churn dataset, not generated or synthetic data.

- Source: https://github.com/IBM/telco-customer-churn-on-icp4d
- Kaggle mirror: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Local file: `data/telco_churn.csv`
- Rows: 7,043 customers
- Columns: 21 fields including demographics, services, billing, and churn label
- Churned customers: 1,869
- Churn rate: 26.5%

The download script validates row count, column count, required schema, and churn-rate range before accepting the dataset.

## What The Dashboard Does

- Loads real customer examples directly from `data/telco_churn.csv`.
- Uses trained Logistic Regression and Random Forest models from `models/`.
- Defaults predictions to the best exported model from `outputs/metrics.json`.
- Shows business KPIs from `outputs/dataset_stats.json`.
- Displays model metrics, ROC curve, feature importance, confusion matrix, and churn distribution.
- Returns churn probability, predicted churn label, risk level, and retention guidance for a customer profile.

## Project Structure

```text
Customer-Churn-Prediction/
|-- app/
|   |-- server.py
|   |-- static/
|   |   |-- css/styles.css
|   |   `-- js/app.js, charts.js, icons.js, theme.js
|   `-- templates/index.html
|-- data/telco_churn.csv
|-- models/
|   |-- logistic_regression.pkl
|   |-- random_forest.pkl
|   |-- scaler.pkl
|   `-- feature_names.pkl
|-- outputs/
|   |-- metrics.json
|   |-- dataset_stats.json
|   |-- feature_importance.json
|   |-- confusion_matrix.json
|   `-- roc_data.json
|-- scripts/
|   |-- download_data.py
|   `-- train_and_export.py
|-- src/
|   |-- data_preprocessing.py
|   |-- feature_engineering.py
|   |-- model_evaluation.py
|   |-- model_training.py
|   |-- predict.py
|   `-- utils.py
|-- requirements.txt
`-- render.yaml
```

## Setup

```bash
pip install -r requirements.txt
```

## Train Models

```bash
python scripts/train_and_export.py
```

This downloads and validates the IBM dataset if needed, cleans and encodes the data, trains both models, saves model artifacts, and exports dashboard JSON files.

## Start Dashboard

```bash
python app/server.py
```

Open:

```text
http://localhost:5000
```

## API Endpoints

- `GET /health`
- `GET /api/metrics`
- `GET /api/dataset-stats`
- `GET /api/customer-examples`
- `GET /api/feature-importance`
- `GET /api/confusion-matrix`
- `GET /api/roc-curve`
- `POST /api/predict`

## Data Attribution

The dataset was originally provided by IBM as part of IBM sample data assets for Telco Customer Churn analysis and is publicly mirrored on Kaggle.

## License

MIT License
