# Customer Churn Prediction & Analytics Dashboard

![Customer Churn Dashboard](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)

A production-grade machine learning project that predicts customer churn using the **IBM Telco Customer Churn** dataset. It features a complete data pipeline, model training (Logistic Regression & Random Forest), and a premium interactive web dashboard to visualize metrics and make live predictions.

## Dataset

This project uses the **IBM Telco Customer Churn** dataset — a real-world dataset from IBM Sample Data Sets containing 7,043 customers and 21 features.

- **Source**: [IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d) (GitHub) / [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows**: 7,043 customers
- **Features**: 21 (demographics, services, billing, and churn label)
- **Churn Rate**: ~26.5% (1,869 out of 7,043 customers churned)

The dataset is automatically downloaded from IBM's GitHub repository when you run the training pipeline for the first time.

## Features

- **Real Data Pipeline**: Automatically downloads and processes the IBM Telco Churn dataset (7,043 rows, 21 features).
- **ML Pipeline**: 
  - Automated data cleaning, scaling (StandardScaler), and encoding (Label & One-Hot).
  - Feature engineering (tenure buckets, service counts).
  - Model training with `GridSearchCV` hyperparameter tuning.
  - Cross-validation and full evaluation metrics (Accuracy, Precision, Recall, F1, AUC, ROC).
- **Interactive Dashboard**:
  - Premium dark/light theme (glassmorphism UI, micro-animations).
  - Live prediction engine (form inputs → Flask API → real-time churn probability).
  - Chart.js visualizations (Confusion Matrix, ROC Curve, Feature Importance).
- **REST API**: Built with Flask to serve models and metrics.

## Project Structure

```text
Customer-Churn-Prediction/
├── app/
│   ├── server.py              # Flask API server
│   ├── static/                # CSS, JS, Fonts
│   │   ├── css/styles.css
│   │   └── js/ (app.js, charts.js, icons.js, theme.js)
│   └── templates/
│       └── index.html         # Main dashboard HTML
├── data/
│   └── telco_churn.csv        # IBM Telco Customer Churn dataset
├── models/                    # Trained model artifacts (.pkl)
├── outputs/                   # Exported metrics JSON for dashboard
├── scripts/
│   ├── download_data.py       # Dataset downloader (from IBM GitHub)
│   └── train_and_export.py    # Master pipeline script
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_evaluation.py
│   ├── model_training.py
│   ├── predict.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/anuragverma4895/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Train the Models
Run the master pipeline script. This will download the real IBM Telco dataset (if not already present), preprocess the data, train both models (Logistic Regression & Random Forest) with GridSearch, and export all metrics/artifacts.

```bash
python scripts/train_and_export.py
```

### 2. Start the Dashboard
Launch the Flask API server to serve the models and the web interface.

```bash
python app/server.py
```

### 3. View the Dashboard
Open your browser and navigate to:
[http://localhost:5000](http://localhost:5000)

## Data Attribution

The dataset used in this project is the **Telco Customer Churn** dataset, originally provided by IBM as part of their sample data sets for IBM Cognos Analytics. It is publicly available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and [IBM's GitHub](https://github.com/IBM/telco-customer-churn-on-icp4d).

## License
MIT License
