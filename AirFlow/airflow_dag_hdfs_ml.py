"""
Apache Airflow DAG for HDFS Incident Prediction using scikit-learn
This DAG orchestrates the complete ML pipeline from data loading to model deployment
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import joblib
import os
import json

# Default arguments
default_args = {
    'owner': 'data_engineer',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'hdfs_incident_prediction_pipeline',
    default_args=default_args,
    description='HDFS Incident Prediction ML Pipeline with scikit-learn',
    schedule_interval=timedelta(days=1),  # Run daily
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'hdfs', 'incident_prediction', 'scikit-learn'],
)

# Directory paths
DATA_DIR = '/tmp/hdfs_ml_data'
MODEL_DIR = '/tmp/hdfs_ml_models'
RESULTS_DIR = '/tmp/hdfs_ml_results'

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_and_validate_data(**context):
    """Load and validate the HDFS features CSV file"""
    print("Loading HDFS features data...")
    
    csv_path = "hdfs_features.csv"
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")
    
    # Basic validation
    assert 'Incident' in df.columns, "Target column 'Incident' not found"
    assert df.shape[0] > 0, "Dataset is empty"
    
    # Save processed data
    output_path = os.path.join(DATA_DIR, 'raw_data.csv')
    df.to_csv(output_path, index=False)
    
    # Log statistics
    stats = {
        'total_records': int(df.shape[0]),
        'total_features': int(df.shape[1]),
        'incident_count': int(df['Incident'].sum()),
        'incident_rate': float(df['Incident'].mean()),
        'missing_values': int(df.isnull().sum().sum())
    }
    
    print(f"Data Statistics: {stats}")
    
    # Push statistics to XCom for downstream tasks
    context['ti'].xcom_push(key='data_stats', value=stats)
    
    return output_path


def preprocess_data(**context):
    """Preprocess data for machine learning"""
    print("Preprocessing data...")
    
    input_path = os.path.join(DATA_DIR, 'raw_data.csv')
    df = pd.read_csv(input_path)
    
    # Feature engineering
    print("Creating engineered features...")
    
    # Interaction features
    df['Warn_Log_Interaction'] = df['Warn_Count'] * df['Log_Volume']
    df['Log_Per_Event_Ratio'] = df['Log_Volume'] / (df['Unique_Event_Count'] + 1)
    
    # Binary flags
    df['High_Warn_Flag'] = (df['Warn_Count'] > 5).astype(int)
    df['High_Log_Volume_Flag'] = (df['Log_Volume'] > 20).astype(int)
    
    # Extract hour from Time_Window
    df['Hour'] = df['Time_Window'].str[7:9].astype(int)
    
    # Encode Component (categorical)
    le_component = LabelEncoder()
    df['Component_Encoded'] = le_component.fit_transform(df['Component'])
    
    # Select features for modeling
    feature_columns = [
        'Warn_Count',
        'Log_Volume',
        'Unique_Event_Count',
        'Warn_Log_Interaction',
        'Log_Per_Event_Ratio',
        'High_Warn_Flag',
        'High_Log_Volume_Flag',
        'Hour',
        'Component_Encoded'
    ]
    
    X = df[feature_columns]
    y = df['Incident']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save preprocessed data
    train_path = os.path.join(DATA_DIR, 'train_data.csv')
    test_path = os.path.join(DATA_DIR, 'test_data.csv')
    
    pd.DataFrame(X_train_scaled, columns=feature_columns).to_csv(train_path, index=False)
    pd.DataFrame(X_test_scaled, columns=feature_columns).to_csv(test_path, index=False)
    
    y_train.to_csv(os.path.join(DATA_DIR, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(DATA_DIR, 'y_test.csv'), index=False)
    
    # Save preprocessors
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(le_component, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Class distribution - Train: {y_train.value_counts().to_dict()}")
    print(f"Class distribution - Test: {y_test.value_counts().to_dict()}")
    
    # Push info to XCom
    context['ti'].xcom_push(key='feature_columns', value=feature_columns)
    context['ti'].xcom_push(key='train_shape', value=X_train.shape)
    context['ti'].xcom_push(key='test_shape', value=X_test.shape)
    
    return train_path, test_path


def train_model(**context):
    """Train scikit-learn ML model"""
    print("Training machine learning model...")
    
    # Load preprocessed data
    train_path = os.path.join(DATA_DIR, 'train_data.csv')
    test_path = os.path.join(DATA_DIR, 'test_data.csv')
    y_train_path = os.path.join(DATA_DIR, 'y_train.csv')
    y_test_path = os.path.join(DATA_DIR, 'y_test.csv')
    
    X_train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    y_test = pd.read_csv(y_test_path).squeeze()
    
    # Model selection - using Random Forest as default
    model_type = context['dag'].default_args.get('model_type', 'random_forest')
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            learning_rate=0.1
        )
    elif model_type == 'logistic_regression':
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            solver='lbfgs'
        )
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    print(f"Training {model_type} model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_auc = roc_auc_score(y_test, y_test_proba) if len(np.unique(y_test)) > 1 else 0.0
    
    metrics = {
        'model_type': model_type,
        'train_accuracy': float(train_accuracy),
        'test_accuracy': float(test_accuracy),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1': float(test_f1),
        'test_auc': float(test_auc),
        'training_date': datetime.now().isoformat()
    }
    
    print("\nModel Performance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Save model
    model_path = os.path.join(MODEL_DIR, 'hdfs_incident_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_test_pred,
        'y_proba': y_test_proba
    })
    predictions_df.to_csv(os.path.join(RESULTS_DIR, 'predictions.csv'), index=False)
    
    # Save confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    cm_df = pd.DataFrame(cm, index=['No Incident', 'Incident'], 
                        columns=['No Incident', 'Incident'])
    cm_df.to_csv(os.path.join(RESULTS_DIR, 'confusion_matrix.csv'))
    
    # Save classification report
    report = classification_report(y_test, y_test_pred, output_dict=True)
    with open(os.path.join(RESULTS_DIR, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Push metrics to XCom
    context['ti'].xcom_push(key='model_metrics', value=metrics)
    context['ti'].xcom_push(key='model_path', value=model_path)
    
    return metrics


def evaluate_model(**context):
    """Evaluate model and generate detailed reports"""
    print("Evaluating model...")
    
    # Get metrics from previous task
    metrics = context['ti'].xcom_pull(task_ids='train_model', key='model_metrics')
    
    # Load predictions
    predictions_path = os.path.join(RESULTS_DIR, 'predictions.csv')
    predictions_df = pd.read_csv(predictions_path)
    
    # Generate evaluation report
    report = f"""
    ============================================
    HDFS INCIDENT PREDICTION - MODEL EVALUATION
    ============================================
    
    Model Type: {metrics['model_type']}
    Training Date: {metrics['training_date']}
    
    Performance Metrics:
    --------------------
    Training Accuracy: {metrics['train_accuracy']:.4f}
    Test Accuracy: {metrics['test_accuracy']:.4f}
    Test Precision: {metrics['test_precision']:.4f}
    Test Recall: {metrics['test_recall']:.4f}
    Test F1 Score: {metrics['test_f1']:.4f}
    Test AUC-ROC: {metrics['test_auc']:.4f}
    
    Dataset Information:
    --------------------
    Total Predictions: {len(predictions_df)}
    Actual Incidents: {int(predictions_df['y_true'].sum())}
    Predicted Incidents: {int(predictions_df['y_pred'].sum())}
    
    ============================================
    """
    
    print(report)
    
    # Save evaluation report
    report_path = os.path.join(RESULTS_DIR, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Evaluation report saved to {report_path}")
    
    return report_path


def model_validation_check(**context):
    """Validate model meets minimum performance thresholds"""
    print("Validating model performance...")
    
    metrics = context['ti'].xcom_pull(task_ids='train_model', key='model_metrics')
    
    # Define thresholds
    min_accuracy = 0.70
    min_f1 = 0.60
    min_auc = 0.70
    
    validation_results = {
        'accuracy_check': metrics['test_accuracy'] >= min_accuracy,
        'f1_check': metrics['test_f1'] >= min_f1,
        'auc_check': metrics['test_auc'] >= min_auc,
        'all_checks_passed': False
    }
    
    validation_results['all_checks_passed'] = all([
        validation_results['accuracy_check'],
        validation_results['f1_check'],
        validation_results['auc_check']
    ])
    
    print("\nValidation Results:")
    print(f"Accuracy >= {min_accuracy}: {validation_results['accuracy_check']} ({metrics['test_accuracy']:.4f})")
    print(f"F1 Score >= {min_f1}: {validation_results['f1_check']} ({metrics['test_f1']:.4f})")
    print(f"AUC >= {min_auc}: {validation_results['auc_check']} ({metrics['test_auc']:.4f})")
    print(f"All Checks Passed: {validation_results['all_checks_passed']}")
    
    if not validation_results['all_checks_passed']:
        print("WARNING: Model does not meet all performance thresholds!")
    
    context['ti'].xcom_push(key='validation_results', value=validation_results)
    
    return validation_results


# Define tasks
task_load_data = PythonOperator(
    task_id='load_and_validate_data',
    python_callable=load_and_validate_data,
    dag=dag,
)

task_preprocess = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

task_train = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

task_evaluate = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

task_validate = PythonOperator(
    task_id='model_validation_check',
    python_callable=model_validation_check,
    dag=dag,
)

task_save_results = BashOperator(
    task_id='save_results_summary',
    bash_command=f"""
    echo "Pipeline completed at $(date)" >> {RESULTS_DIR}/pipeline_summary.txt
    echo "Results saved in {RESULTS_DIR}" >> {RESULTS_DIR}/pipeline_summary.txt
    """,
    dag=dag,
)

# Define task dependencies
task_load_data >> task_preprocess >> task_train >> [task_evaluate, task_validate] >> task_save_results
