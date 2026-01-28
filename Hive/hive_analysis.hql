-- HIVE Analysis Script for HDFS Features
-- This script performs comprehensive analysis on the HDFS features dataset

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS hdfs_analysis;
USE hdfs_analysis;

-- Create external table pointing to CSV file
DROP TABLE IF EXISTS hdfs_features;
CREATE EXTERNAL TABLE hdfs_features (
    Component STRING,
    Time_Window STRING,
    Warn_Count INT,
    Log_Volume INT,
    Unique_Event_Count INT,
    Incident INT
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/usr/hive/warehouse/hdfs_analysis/hdfs_features'
TBLPROPERTIES ('skip.header.line.count'='1');

-- Alternative: Load from local CSV (if using local mode)
-- LOAD DATA LOCAL INPATH '/home/hadoop/hdfs_features.csv' OVERWRITE INTO TABLE hdfs_features;

-- 1. Basic Statistics
SELECT 
    COUNT(*) as total_records,
    COUNT(DISTINCT Component) as unique_components,
    SUM(Incident) as total_incidents,
    AVG(Warn_Count) as avg_warn_count,
    AVG(Log_Volume) as avg_log_volume,
    AVG(Unique_Event_Count) as avg_unique_events
FROM hdfs_features;

-- 2. Incident Rate Analysis
SELECT 
    Component,
    COUNT(*) as total_records,
    SUM(Incident) as incident_count,
    ROUND(SUM(Incident) * 100.0 / COUNT(*), 2) as incident_rate_percent,
    AVG(Warn_Count) as avg_warn_count,
    AVG(Log_Volume) as avg_log_volume
FROM hdfs_features
GROUP BY Component
ORDER BY incident_rate_percent DESC;

-- 3. Time Window Analysis
SELECT 
    SUBSTR(Time_Window, 1, 6) as date_part,
    COUNT(*) as records_per_date,
    SUM(Incident) as incidents_per_date,
    AVG(Warn_Count) as avg_warn_count
FROM hdfs_features
GROUP BY SUBSTR(Time_Window, 1, 6)
ORDER BY date_part;

-- 4. Component-wise Feature Analysis
SELECT 
    Component,
    MIN(Warn_Count) as min_warn,
    MAX(Warn_Count) as max_warn,
    AVG(Warn_Count) as avg_warn,
    MIN(Log_Volume) as min_log_volume,
    MAX(Log_Volume) as max_log_volume,
    AVG(Log_Volume) as avg_log_volume,
    MIN(Unique_Event_Count) as min_unique_events,
    MAX(Unique_Event_Count) as max_unique_events,
    AVG(Unique_Event_Count) as avg_unique_events
FROM hdfs_features
GROUP BY Component;

-- 5. High Risk Time Windows (where incidents occurred)
SELECT 
    Time_Window,
    Component,
    Warn_Count,
    Log_Volume,
    Unique_Event_Count,
    Incident
FROM hdfs_features
WHERE Incident = 1
ORDER BY Time_Window;

-- 6. Correlation Analysis (using window functions)
SELECT 
    Component,
    AVG(Warn_Count) as avg_warn,
    AVG(Log_Volume) as avg_log_volume,
    AVG(Unique_Event_Count) as avg_unique_events,
    SUM(Incident) as total_incidents,
    CASE 
        WHEN AVG(Warn_Count) > 5 THEN 'High Warning'
        WHEN AVG(Warn_Count) > 2 THEN 'Medium Warning'
        ELSE 'Low Warning'
    END as warning_category
FROM hdfs_features
GROUP BY Component;

-- 7. Create summary table for ML features
DROP TABLE IF EXISTS hdfs_features_summary;
CREATE TABLE hdfs_features_summary AS
SELECT 
    Component,
    Time_Window,
    Warn_Count,
    Log_Volume,
    Unique_Event_Count,
    Incident,
    -- Feature engineering
    Warn_Count * Log_Volume as warn_log_interaction,
    Log_Volume / NULLIF(Unique_Event_Count, 0) as log_per_event_ratio,
    CASE WHEN Warn_Count > 5 THEN 1 ELSE 0 END as high_warn_flag,
    CASE WHEN Log_Volume > 20 THEN 1 ELSE 0 END as high_log_volume_flag
FROM hdfs_features;

-- 8. Export summary for ML
INSERT OVERWRITE LOCAL DIRECTORY '/airflow_project/tmp/hdfs_ml_data'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
SELECT * FROM hdfs_features_summary;
