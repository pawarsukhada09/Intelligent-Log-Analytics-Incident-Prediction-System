"""
Exploratory Data Analysis (EDA) for HDFS Features Dataset
This script performs comprehensive EDA including visualizations and statistical analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class HDFSEdaAnalyzer:
    def __init__(self, csv_path):
        """Initialize the EDA analyzer with CSV file path"""
        self.csv_path = csv_path
        self.df = None
        self.output_dir = Path("eda_output")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """Load the CSV data"""
        print("Loading data...")
        self.df = pd.read_csv(self.csv_path)
        print(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df
    
    def basic_info(self):
        """Display basic information about the dataset"""
        print("\n" + "="*50)
        print("BASIC DATASET INFORMATION")
        print("="*50)
        
        print("\nFirst few rows:")
        print(self.df.head(10))
        
        print("\nDataset Info:")
        print(self.df.info())
        
        print("\nData Types:")
        print(self.df.dtypes)
        
        print("\nMissing Values:")
        missing = self.df.isnull().sum()
        print(missing[missing > 0] if missing.sum() > 0 else "No missing values")
        
        print("\nBasic Statistics:")
        print(self.df.describe())
        
        return self.df.describe()
    
    def target_analysis(self):
        """Analyze the target variable (Incident)"""
        print("\n" + "="*50)
        print("TARGET VARIABLE ANALYSIS")
        print("="*50)
        
        incident_counts = self.df['Incident'].value_counts()
        print("\nIncident Distribution:")
        print(incident_counts)
        print(f"\nIncident Rate: {self.df['Incident'].mean()*100:.2f}%")
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        sns.countplot(data=self.df, x='Incident', ax=axes[0])
        axes[0].set_title('Incident Distribution')
        axes[0].set_xlabel('Incident (0=No, 1=Yes)')
        
        # Pie chart
        incident_counts.plot(kind='pie', autopct='%1.1f%%', ax=axes[1])
        axes[1].set_title('Incident Proportion')
        axes[1].set_ylabel('')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'incident_distribution.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {self.output_dir / 'incident_distribution.png'}")
        plt.close()
        
        return incident_counts
    
    def component_analysis(self):
        """Analyze component distribution and incident rates"""
        print("\n" + "="*50)
        print("COMPONENT ANALYSIS")
        print("="*50)
        
        component_stats = self.df.groupby('Component').agg({
            'Incident': ['count', 'sum', 'mean'],
            'Warn_Count': ['mean', 'max'],
            'Log_Volume': ['mean', 'max'],
            'Unique_Event_Count': ['mean', 'max']
        }).round(2)
        
        component_stats.columns = ['Total_Records', 'Incident_Count', 'Incident_Rate',
                                   'Avg_Warn_Count', 'Max_Warn_Count',
                                   'Avg_Log_Volume', 'Max_Log_Volume',
                                   'Avg_Unique_Events', 'Max_Unique_Events']
        
        component_stats = component_stats.sort_values('Incident_Rate', ascending=False)
        print("\nComponent Statistics:")
        print(component_stats)
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Component distribution
        component_counts = self.df['Component'].value_counts()
        component_counts.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Component Distribution')
        axes[0, 0].set_xlabel('Component')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Incident rate by component
        incident_rate = self.df.groupby('Component')['Incident'].mean().sort_values(ascending=False)
        incident_rate.plot(kind='bar', ax=axes[0, 1], color='coral')
        axes[0, 1].set_title('Incident Rate by Component')
        axes[0, 1].set_xlabel('Component')
        axes[0, 1].set_ylabel('Incident Rate')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Average warn count by component
        avg_warn = self.df.groupby('Component')['Warn_Count'].mean().sort_values(ascending=False)
        avg_warn.plot(kind='bar', ax=axes[1, 0], color='skyblue')
        axes[1, 0].set_title('Average Warn Count by Component')
        axes[1, 0].set_xlabel('Component')
        axes[1, 0].set_ylabel('Average Warn Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Average log volume by component
        avg_log = self.df.groupby('Component')['Log_Volume'].mean().sort_values(ascending=False)
        avg_log.plot(kind='bar', ax=axes[1, 1], color='lightgreen')
        axes[1, 1].set_title('Average Log Volume by Component')
        axes[1, 1].set_xlabel('Component')
        axes[1, 1].set_ylabel('Average Log Volume')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'component_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {self.output_dir / 'component_analysis.png'}")
        plt.close()
        
        return component_stats
    
    def feature_distributions(self):
        """Analyze distributions of numerical features"""
        print("\n" + "="*50)
        print("FEATURE DISTRIBUTIONS")
        print("="*50)
        
        numerical_features = ['Warn_Count', 'Log_Volume', 'Unique_Event_Count']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        for idx, feature in enumerate(numerical_features):
            # Histogram
            self.df[feature].hist(bins=30, ax=axes[0, idx], edgecolor='black')
            axes[0, idx].set_title(f'{feature} Distribution')
            axes[0, idx].set_xlabel(feature)
            axes[0, idx].set_ylabel('Frequency')
            
            # Box plot
            self.df.boxplot(column=feature, ax=axes[1, idx])
            axes[1, idx].set_title(f'{feature} Box Plot')
            axes[1, idx].set_ylabel(feature)
            
            # Statistics
            print(f"\n{feature} Statistics:")
            print(self.df[feature].describe())
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {self.output_dir / 'feature_distributions.png'}")
        plt.close()
    
    def correlation_analysis(self):
        """Analyze correlations between features"""
        print("\n" + "="*50)
        print("CORRELATION ANALYSIS")
        print("="*50)
        
        numerical_df = self.df.select_dtypes(include=[np.number])
        correlation_matrix = numerical_df.corr()
        
        print("\nCorrelation Matrix:")
        print(correlation_matrix)
        
        # Visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {self.output_dir / 'correlation_matrix.png'}")
        plt.close()
        
        return correlation_matrix
    
    def incident_vs_features(self):
        """Analyze relationship between incidents and features"""
        print("\n" + "="*50)
        print("INCIDENT vs FEATURES ANALYSIS")
        print("="*50)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Warn Count vs Incident
        sns.boxplot(data=self.df, x='Incident', y='Warn_Count', ax=axes[0, 0])
        axes[0, 0].set_title('Warn Count Distribution by Incident')
        
        # Log Volume vs Incident
        sns.boxplot(data=self.df, x='Incident', y='Log_Volume', ax=axes[0, 1])
        axes[0, 1].set_title('Log Volume Distribution by Incident')
        
        # Unique Event Count vs Incident
        sns.boxplot(data=self.df, x='Incident', y='Unique_Event_Count', ax=axes[1, 0])
        axes[1, 0].set_title('Unique Event Count Distribution by Incident')
        
        # Scatter plot: Warn Count vs Log Volume (colored by Incident)
        scatter = axes[1, 1].scatter(self.df['Warn_Count'], self.df['Log_Volume'], 
                                   c=self.df['Incident'], cmap='RdYlGn', alpha=0.6)
        axes[1, 1].set_xlabel('Warn Count')
        axes[1, 1].set_ylabel('Log Volume')
        axes[1, 1].set_title('Warn Count vs Log Volume (Incident colored)')
        plt.colorbar(scatter, ax=axes[1, 1], label='Incident')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'incident_vs_features.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {self.output_dir / 'incident_vs_features.png'}")
        plt.close()
        
        # Statistical comparison
        print("\nFeature Statistics by Incident Status:")
        incident_stats = self.df.groupby('Incident')[['Warn_Count', 'Log_Volume', 'Unique_Event_Count']].agg(['mean', 'std', 'median'])
        print(incident_stats)
        
        return incident_stats
    
    def time_analysis(self):
        """Analyze time-based patterns"""
        print("\n" + "="*50)
        print("TIME WINDOW ANALYSIS")
        print("="*50)
        
        # Extract date and hour from Time_Window
        self.df['Date'] = self.df['Time_Window'].str[:6]
        self.df['Hour'] = self.df['Time_Window'].str[7:9].astype(int)
        
        # Date analysis
        date_incidents = self.df.groupby('Date').agg({
            'Incident': ['count', 'sum', 'mean'],
            'Warn_Count': 'mean'
        })
        date_incidents.columns = ['Total_Records', 'Incident_Count', 'Incident_Rate', 'Avg_Warn_Count']
        print("\nIncidents by Date:")
        print(date_incidents)
        
        # Hour analysis
        hour_incidents = self.df.groupby('Hour').agg({
            'Incident': ['count', 'sum', 'mean'],
            'Warn_Count': 'mean'
        })
        hour_incidents.columns = ['Total_Records', 'Incident_Count', 'Incident_Rate', 'Avg_Warn_Count']
        print("\nIncidents by Hour:")
        print(hour_incidents)
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Incidents by date
        date_incidents['Incident_Count'].plot(kind='bar', ax=axes[0, 0], color='steelblue')
        axes[0, 0].set_title('Incident Count by Date')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Incident Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Incident rate by hour
        hour_incidents['Incident_Rate'].plot(kind='line', ax=axes[0, 1], marker='o', color='coral')
        axes[0, 1].set_title('Incident Rate by Hour of Day')
        axes[0, 1].set_xlabel('Hour')
        axes[0, 1].set_ylabel('Incident Rate')
        axes[0, 1].grid(True)
        
        # Average warn count by hour
        hour_incidents['Avg_Warn_Count'].plot(kind='bar', ax=axes[1, 0], color='lightgreen')
        axes[1, 0].set_title('Average Warn Count by Hour')
        axes[1, 0].set_xlabel('Hour')
        axes[1, 0].set_ylabel('Average Warn Count')
        
        # Records by hour
        hour_incidents['Total_Records'].plot(kind='bar', ax=axes[1, 1], color='skyblue')
        axes[1, 1].set_title('Total Records by Hour')
        axes[1, 1].set_xlabel('Hour')
        axes[1, 1].set_ylabel('Total Records')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'time_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {self.output_dir / 'time_analysis.png'}")
        plt.close()
        
        return date_incidents, hour_incidents
    
    def generate_report(self):
        """Generate comprehensive EDA report"""
        print("\n" + "="*50)
        print("GENERATING EDA REPORT")
        print("="*50)
        
        report_path = self.output_dir / 'eda_report.txt'
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("HDFS FEATURES - EXPLORATORY DATA ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Dataset Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns\n\n")
            
            f.write("Target Variable Distribution:\n")
            f.write(str(self.df['Incident'].value_counts()) + "\n\n")
            
            f.write("Component Distribution:\n")
            f.write(str(self.df['Component'].value_counts()) + "\n\n")
            
            f.write("Numerical Features Summary:\n")
            f.write(str(self.df.describe()) + "\n\n")
            
            f.write("Correlation with Incident:\n")
            numerical_df = self.df.select_dtypes(include=[np.number])
            correlations = numerical_df.corr()['Incident'].sort_values(ascending=False)
            f.write(str(correlations) + "\n")
        
        print(f"Report saved: {report_path}")
        return report_path
    
    def run_full_eda(self):
        """Run complete EDA pipeline"""
        print("\n" + "="*60)
        print("STARTING EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        self.load_data()
        self.basic_info()
        self.target_analysis()
        self.component_analysis()
        self.feature_distributions()
        self.correlation_analysis()
        self.incident_vs_features()
        self.time_analysis()
        self.generate_report()
        
        print("\n" + "="*60)
        print("EDA COMPLETE! Check 'eda_output' directory for results.")
        print("="*60)


if __name__ == "__main__":
    # Run EDA
    analyzer = HDFSEdaAnalyzer("hdfs_features.csv")
    analyzer.run_full_eda()
