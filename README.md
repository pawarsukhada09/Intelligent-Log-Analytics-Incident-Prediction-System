# HDFS Incident Prediction Pipeline

A comprehensive machine learning pipeline for predicting HDFS (Hadoop Distributed File System) incidents using log features. This project includes HIVE analysis, Exploratory Data Analysis (EDA), PySpark ML models, and an Apache Airflow orchestration pipeline with scikit-learn.

-------------------------------------------------------------------------------

## Project Structure

Pipeline project/
├── hdfs_features.csv              # Input dataset
├── hive_analysis.hql              # HIVE SQL analysis scripts
├── eda_analysis.py                # Exploratory Data Analysis script
├── pyspark_ml_model.py            # PySpark ML model implementation
├── airflow_dag_hdfs_ml.py         # Apache Airflow DAG for ML pipeline
└── README.md                      # This file

-------------------------------------------------------------------------------

## Dataset

The `hdfs_features.csv` file contains HDFS log features with the following columns:
- **Component**: HDFS component name (e.g. dfs.DataNode, dfs.DataBlockScanner)
- **Time_Window**: Time window identifier (format: YYMMDD_HH)
- **Warn_Count**: Number of warnings in the time window
- **Log_Volume**: Volume of logs generated
- **Unique_Event_Count**: Count of unique events
- **Incident**: Target variable (0 = No Incident, 1 = Incident)

-------------------------------------------------------------------------------

### 1. HIVE Analysis (`hive_analysis.hql`)

Comprehensive SQL analysis scripts for HIVE/Hadoop ecosystem:
- Basic statistics and aggregations
- Component-wise incident rate analysis
- Time window analysis
- Feature engineering for ML
- Summary table creation

**Command:**
/home/Hadoop/apache-hive-2.4.2-bin/bin -f hive_analysis.hql

-------------------------------------------------------------------------------

### 2. Exploratory Data Analysis (`eda_analysis.py`)

Comprehensive EDA with visualizations:
- Dataset overview and statistics
- Target variable analysis
- Component distribution analysis
- Feature distributions and correlations
- Time-based pattern analysis
- Incident vs features relationship analysis

**Command:**
python3 eda_analysis.py


Output: Results saved in `eda_output/` directory and files are:-
- `incident_distribution.png`
- `component_analysis.png`
- `feature_distributions.png`
- `correlation_matrix.png`
- `incident_vs_features.png`
- `time_analysis.png`
- `eda_report.txt`

-------------------------------------------------------------------------------

### 3. PySpark ML Model (`pyspark_ml_model.py`)

Machine learning models using PySpark MLlib:
- Data preprocessing and feature engineering
- Multiple model options: Random Forest, Gradient Boosting, Logistic Regression
- Model evaluation with comprehensive metrics
- Feature importance analysis
- Model persistence

**Command:**
python3 pyspark_ml_model.py

Model Options:
- `random_forest`: Random Forest Classifier 
- `gbt`: Gradient Boosting Classifier

-------------------------------------------------------------------------------

### 4. Apache Airflow Pipeline (`airflow_dag_hdfs_ml.py`)

End-to-end ML pipeline orchestration with scikit-learn:
- Automated data loading and validation
- Data preprocessing and feature engineering
- Model training with scikit-learn
- Model evaluation and validation
- Results persistence

A. Copy DAG to Airflow DAGs folder:
**Command:**
cp airflow_dag_hdfs_ml.py ~/airflow/dags/

B. Access Airflow UI:
   - Open browser: `http://localhost:8080`
   - Login with credentials 
   - Find DAG: `hdfs_incident_prediction_pipeline`
   - Trigger the DAG manually or wait for scheduled execution

**Pipeline Tasks:**
1. `load_and_validate_data`: Load and validate CSV data
2. `preprocess_data`: Feature engineering and data splitting
3. `train_model`: Train scikit-learn model (Random Forest by default)
4. `evaluate_model`: Generate evaluation metrics and reports
5. `model_validation_check`: Validate model meets performance thresholds
6. `save_results_summary`: Save pipeline summary

-------------------------------------------------------------------------------

## Model Performance

The models are evaluated using:
- Accuracy: Overall prediction accuracy
- Precision: Precision for incident class
- Recall: Recall for incident class
- F1 Score: Harmonic mean of precision and recall
- AUC-ROC: Area under the ROC curve

## Output Locations

- EDA Output: `eda_output/` directory
- PySpark Models: `models/` directory
- Airflow Results: `/tmp/hdfs_ml_results/` 
- Airflow Models: `/tmp/hdfs_ml_models/`