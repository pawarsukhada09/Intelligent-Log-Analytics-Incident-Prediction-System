"""
PySpark Machine Learning Model for HDFS Incident Prediction
This script creates and trains ML models using PySpark MLlib
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, isnull, count, avg
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import pandas as pd

class HDFSMLModel:
    def __init__(self, csv_path="hdfs_features.csv", app_name="HDFS_Incident_Prediction"):
        """Initialize Spark session and load data"""
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
        
        self.csv_path = csv_path
        self.df = None
        self.model = None
        self.predictions = None
        
    def load_data(self):
        """Load CSV data into Spark DataFrame"""
        print("Loading data into Spark...")
        self.df = self.spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .csv(self.csv_path)
        
        print(f"Data loaded: {self.df.count()} rows")
        self.df.printSchema()
        self.df.show(10, truncate=False)
        return self.df
    
    def data_preprocessing(self):
        """Preprocess data for ML"""
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Check for missing values
        print("\nMissing values:")
        for col_name in self.df.columns:
            null_count = self.df.filter(col(col_name).isNull() | isnan(col_name)).count()
            if null_count > 0:
                print(f"{col_name}: {null_count}")
        
        # Convert Incident to integer if needed
        self.df = self.df.withColumn("Incident", col("Incident").cast("integer"))
        
        # Feature engineering
        print("\nCreating features...")
        
        # Create interaction features
        self.df = self.df.withColumn(
            "Warn_Log_Interaction",
            col("Warn_Count") * col("Log_Volume")
        )
        
        self.df = self.df.withColumn(
            "Log_Per_Event_Ratio",
            when(col("Unique_Event_Count") != 0, 
                 col("Log_Volume") / col("Unique_Event_Count")).otherwise(0)
        )
        
        # Create binary flags
        self.df = self.df.withColumn(
            "High_Warn_Flag",
            when(col("Warn_Count") > 5, 1).otherwise(0)
        )
        
        self.df = self.df.withColumn(
            "High_Log_Volume_Flag",
            when(col("Log_Volume") > 20, 1).otherwise(0)
        )
        
        # Extract hour from Time_Window
        self.df = self.df.withColumn(
            "Hour",
            col("Time_Window").substr(8, 2).cast("integer")
        )
        
        print("\nSample of processed data:")
        self.df.select("Component", "Warn_Count", "Log_Volume", 
                      "Warn_Log_Interaction", "Incident").show(10)
        
        # Check class distribution
        print("\nClass distribution:")
        self.df.groupBy("Incident").count().show()
        
        return self.df
    
    def create_feature_pipeline(self):
        """Create feature engineering pipeline"""
        print("\n" + "="*50)
        print("CREATING FEATURE PIPELINE")
        print("="*50)
        
        # Index Component (categorical feature)
        component_indexer = StringIndexer(
            inputCol="Component",
            outputCol="ComponentIndex"
        )
        
        # One-hot encode Component
        component_encoder = OneHotEncoder(
            inputCols=["ComponentIndex"],
            outputCols=["ComponentVec"]
        )
        
        # Assemble all features
        feature_columns = [
            "Warn_Count",
            "Log_Volume",
            "Unique_Event_Count",
            "Warn_Log_Interaction",
            "Log_Per_Event_Ratio",
            "High_Warn_Flag",
            "High_Log_Volume_Flag",
            "Hour",
            "ComponentVec"
        ]
        
        assembler = VectorAssembler(
            inputCols=feature_columns,
            outputCol="features"
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=[
            component_indexer,
            component_encoder,
            assembler
        ])
        
        return pipeline
    
    def train_model(self, model_type="random_forest"):
        """Train ML model"""
        print("\n" + "="*50)
        print(f"TRAINING {model_type.upper()} MODEL")
        print("="*50)
        
        # Create feature pipeline
        feature_pipeline = self.create_feature_pipeline()
        
        # Select model
        if model_type.lower() == "random_forest":
            classifier = RandomForestClassifier(
                labelCol="Incident",
                featuresCol="features",
                numTrees=100,
                maxDepth=10,
                seed=42
            )
        elif model_type.lower() == "gbt":
            classifier = GBTClassifier(
                labelCol="Incident",
                featuresCol="features",
                maxIter=50,
                maxDepth=5,
                seed=42
            )
        elif model_type.lower() == "logistic_regression":
            classifier = LogisticRegression(
                labelCol="Incident",
                featuresCol="features",
                maxIter=100,
                regParam=0.01
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create full pipeline
        full_pipeline = Pipeline(stages=[
            feature_pipeline.getStages()[0],  # component_indexer
            feature_pipeline.getStages()[1],  # component_encoder
            feature_pipeline.getStages()[2],  # assembler
            classifier
        ])
        
        # Split data
        train_data, test_data = self.df.randomSplit([0.8, 0.2], seed=42)
        
        print(f"\nTraining set size: {train_data.count()}")
        print(f"Test set size: {test_data.count()}")
        
        # Train model
        print("\nTraining model...")
        self.model = full_pipeline.fit(train_data)
        
        # Make predictions
        print("\nMaking predictions...")
        self.predictions = self.model.transform(test_data)
        
        # Show sample predictions
        print("\nSample predictions:")
        self.predictions.select(
            "Component", "Warn_Count", "Log_Volume", 
            "Incident", "prediction", "probability"
        ).show(20, truncate=False)
        
        return self.model, self.predictions
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        if self.predictions is None:
            raise ValueError("Model must be trained first. Call train_model()")
        
        # Binary classification evaluator (AUC)
        binary_evaluator = BinaryClassificationEvaluator(
            labelCol="Incident",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
        
        auc = binary_evaluator.evaluate(self.predictions)
        print(f"\nArea Under ROC (AUC): {auc:.4f}")
        
        # Multiclass evaluator
        multi_evaluator = MulticlassClassificationEvaluator(
            labelCol="Incident",
            predictionCol="prediction",
            metricName="f1"
        )
        
        f1_score = multi_evaluator.evaluate(self.predictions)
        print(f"F1 Score: {f1_score:.4f}")
        
        accuracy = multi_evaluator.evaluate(
            self.predictions,
            {multi_evaluator.metricName: "accuracy"}
        )
        print(f"Accuracy: {accuracy:.4f}")
        
        precision = multi_evaluator.evaluate(
            self.predictions,
            {multi_evaluator.metricName: "weightedPrecision"}
        )
        print(f"Precision: {precision:.4f}")
        
        recall = multi_evaluator.evaluate(
            self.predictions,
            {multi_evaluator.metricName: "weightedRecall"}
        )
        print(f"Recall: {recall:.4f}")
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        confusion_matrix = self.predictions.groupBy("Incident", "prediction").count().orderBy("Incident", "prediction")
        confusion_matrix.show()
        
        # Feature importance (for tree-based models)
        try:
            rf_model = self.model.stages[-1]
            if hasattr(rf_model, 'featureImportances'):
                print("\nTop 10 Feature Importances:")
                importances = rf_model.featureImportances.toArray()
                feature_names = [
                    "Warn_Count", "Log_Volume", "Unique_Event_Count",
                    "Warn_Log_Interaction", "Log_Per_Event_Ratio",
                    "High_Warn_Flag", "High_Log_Volume_Flag", "Hour"
                ] + [f"Component_{i}" for i in range(len(importances) - 8)]
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names[:len(importances)],
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                print(importance_df.head(10).to_string(index=False))
        except:
            print("\nFeature importance not available for this model type")
        
        return {
            'AUC': auc,
            'F1': f1_score,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall
        }
    
    def save_model(self, model_path="models/hdfs_ml_model"):
        """Save trained model"""
        print(f"\nSaving model to {model_path}...")
        self.model.write().overwrite().save(model_path)
        print("Model saved successfully!")
    
    def load_model(self, model_path="models/hdfs_ml_model"):
        """Load saved model"""
        from pyspark.ml import PipelineModel
        print(f"Loading model from {model_path}...")
        self.model = PipelineModel.load(model_path)
        print("Model loaded successfully!")
        return self.model
    
    def predict_new_data(self, new_data_df):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model must be loaded or trained first")
        
        predictions = self.model.transform(new_data_df)
        return predictions
    
    def run_full_pipeline(self, model_type="random_forest", save_model=True):
        """Run complete ML pipeline"""
        print("\n" + "="*60)
        print("STARTING PYSPARK ML PIPELINE")
        print("="*60)
        
        self.load_data()
        self.data_preprocessing()
        self.train_model(model_type=model_type)
        metrics = self.evaluate_model()
        
        if save_model:
            self.save_model()
        
        print("\n" + "="*60)
        print("ML PIPELINE COMPLETE!")
        print("="*60)
        
        return metrics


if __name__ == "__main__":
    # Run ML pipeline
    ml_model = HDFSMLModel("hdfs_features.csv")
    
    # Train Random Forest model
    print("\nTraining Random Forest Model...")
    ml_model.run_full_pipeline(model_type="random_forest")
    
    # Train GBT models
    print("\nTraining GBT Model...")
    ml_model_gbt = HDFSMLModel("hdfs_features.csv")
    ml_model_gbt.run_full_pipeline(model_type="gbt")

    