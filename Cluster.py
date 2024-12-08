import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def clean_and_convert(series):
    """
    Clean and convert series to numeric, handling potential string or mixed data
    """
    # Remove any non-numeric characters
    return pd.to_numeric(series.astype(str).str.replace('[^0-9.]', '', regex=True), errors='coerce')

def analyze_calorie_burn_clustering(csv_path):
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Print column types for debugging
    print("Original Column Types:")
    print(df.dtypes)
    
    # Clean and convert potentially problematic columns
    numeric_columns = [
        'Session_Duration (hours)', 
        'Max_BPM', 
        'Avg_BPM', 
        'Fat_Percentage', 
        'Workout_Frequency (days/week)',
        'Calories_Burned'
    ]
    
    # Clean each numeric column
    for col in numeric_columns:
        df[col] = clean_and_convert(df[col])
    
    # Remove rows with NaN values
    df_cleaned = df.dropna(subset=numeric_columns)
    
    # Encode Workout Type
    le = LabelEncoder()
    df_cleaned['Workout_Type_Encoded'] = le.fit_transform(df_cleaned['Workout_Type'])
    
    # Select features for clustering
    clustering_features = [
        'Session_Duration (hours)', 
        'Max_BPM', 
        'Avg_BPM', 
        'Fat_Percentage', 
        'Workout_Frequency (days/week)',
        'Workout_Type_Encoded'
    ]
    
    # Prepare feature matrix
    X = df_cleaned[clustering_features]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_cleaned['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Analyze calories burned by cluster
    cluster_summary = df_cleaned.groupby('Cluster').agg({
        'Calories_Burned': ['mean', 'max', 'min', 'count'],
        'Workout_Type': lambda x: x.value_counts().index[0],
        'Session_Duration (hours)': 'mean',
        'Max_BPM': 'mean',
        'Workout_Frequency (days/week)': 'mean'
    }).reset_index()
    
    # Workout type breakdown
    workout_type_breakdown = df_cleaned.groupby(['Cluster', 'Workout_Type'])['Calories_Burned'].agg(['mean', 'count']).reset_index()
    
    # Visualization: Boxplot of Calories Burned by Cluster
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y='Calories_Burned', data=df_cleaned)
    plt.title('Calories Burned Across Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Calories Burned')
    plt.tight_layout()
    plt.savefig('calories_burned_clusters.png')
    plt.close()
    
    # Correlation of features with Calories Burned
    correlation_features = [
        'Session_Duration (hours)', 
        'Max_BPM', 
        'Avg_BPM', 
        'Fat_Percentage', 
        'Workout_Frequency (days/week)',
        'Calories_Burned'
    ]
    correlation_with_calories = df_cleaned[correlation_features].corr()['Calories_Burned'].drop('Calories_Burned')
    
    # Print and save results
    print("\nCluster Summary:")
    print(cluster_summary)
    print("\nWorkout Type Breakdown:")
    print(workout_type_breakdown)
    print("\nFeature Correlation with Calories Burned:")
    print(correlation_with_calories)
    
    # Save results to CSV
    cluster_summary.to_csv('cluster_summary.csv', index=False)
    workout_type_breakdown.to_csv('workout_type_breakdown.csv', index=False)
    correlation_with_calories.to_frame('Correlation').to_csv('feature_correlation.csv')
    
    return {
        'cluster_summary': cluster_summary,
        'workout_breakdown': workout_type_breakdown,
        'feature_correlation': correlation_with_calories
    }

# Main execution
if __name__ == '__main__':
    results = analyze_calorie_burn_clustering('tracking_data.csv')