# Customer Churn Prediction & Analytics Dashboard

![Customer Churn Dashboard](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)

A production-grade machine learning project that predicts customer churn using the Telco Customer Churn dataset. It features a complete data pipeline, model training (Logistic Regression & Random Forest), and a premium interactive web dashboard to visualize metrics and make live predictions.

## Features

- **Synthetic Data Generator**: Recreates the classic Kaggle Telco Churn dataset with realistic distributions (7,043 rows, 21 features).
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
│   └── telco_churn.csv        # Generated dataset
├── models/                    # Trained model artifacts (.pkl)
├── outputs/                   # Exported metrics JSON for dashboard
├── scripts/
│   ├── generate_data.py       # Data generator
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
Run the master pipeline script. This will generate the dataset (if it doesn't exist), preprocess the data, train both models (Logistic Regression & Random Forest) with GridSearch, and export all metrics/artifacts.

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

## License
MIT License
