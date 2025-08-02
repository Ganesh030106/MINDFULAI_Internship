import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')


REPORT_FOLDER = "reports"


def ensure_reports_dir():
    """Ensures the reports directory exists."""
    if not os.path.exists(REPORT_FOLDER):
        os.makedirs(REPORT_FOLDER)
    print(f"Report directory '{REPORT_FOLDER}' ensured.")

def load_fresh_data(filepath):
    """Loads data from a CSV file."""
    print("Loading data from:", filepath)
    return pd.read_csv(filepath)

def preprocess_data(df):
    """
    Preprocesses the dataset by converting dates, handling missing values,
    creating new date-based features, encoding categorical columns,
    and creating a 'High_Sales' target variable.
    """
    print("[INFO] Preprocessing Dataset...")
    # Convert 'Order Date' to datetime, coercing errors to NaT
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    # Drop 'Ship Date' as it's not used in current analysis
    df = df.drop('Ship Date', axis=1)
    # Drop rows where 'Order Date' could not be parsed
    df = df.dropna(subset=['Order Date'])
    # Fill missing 'Postal Code' with a sentinel value -1
    df['Postal Code'] = df['Postal Code'].fillna(-1)
    # Remove duplicate rows
    df = df.drop_duplicates()

    # Feature Engineering: Extract month, quarter, and weekday from 'Order Date'
    df['Order_Month'] = df['Order Date'].dt.month
    df['Order_Quarter'] = df['Order Date'].dt.quarter
    df['Order_Weekday'] = df['Order Date'].dt.weekday

    # Identify categorical columns for encoding
    cat_cols = ['Category', 'Sub-Category', 'Region', 'Segment', 'Ship Mode']
    # Apply Label Encoding to categorical columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    # Create a binary target variable 'High_Sales' based on median sales
    df['High_Sales'] = (df['Sales'] > df['Sales'].median()).astype(int)
    print("[INFO] Preprocessing complete.")
    return df, cat_cols


def detect_anomalies(df):
    """
    Detects anomalies in the 'Sales' column using the IQR method,
    saves anomalous records, plots a boxplot, and returns a DataFrame
    with anomalies removed.
    """
    print("\n🚨 Anomaly Detection Report...")
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    q1 = df['Sales'].quantile(0.25)
    q3 = df['Sales'].quantile(0.75)
    # Calculate IQR (Interquartile Range)
    iqr = q3 - q1
    # Define lower and upper bounds for anomaly detection
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Identify anomalous records
    anomalies = df[(df['Sales'] < lower_bound) | (df['Sales'] > upper_bound)]
    print(f"Found {len(anomalies)} anomalous records based on IQR method.")
    # Save anomalous records to a CSV file
    anomalies.to_csv(os.path.join(REPORT_FOLDER, 'anomaly_records.csv'), index=False)
    print(f"Anomalous records saved to '{os.path.join(REPORT_FOLDER, 'anomaly_records.csv')}'")

    # Save anomaly boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df["Sales"])
    plt.title("Sales Anomaly Detection (Boxplot)")
    plt.xlabel("Sales")
    plt.tight_layout()
    plot_path = os.path.join(REPORT_FOLDER, "1. sales_anomalies_boxplot.png")
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    print(f"Anomaly boxplot saved to '{plot_path}'")

    # Filter out anomalous records and return the cleaned DataFrame
    df_cleaned = df[~((df['Sales'] < lower_bound) | (df['Sales'] > upper_bound))].copy()
    print(f"Removed {len(anomalies)} anomalies. Remaining records: {len(df_cleaned)}")
    return df_cleaned


def plot_scatter(df, x_col, y_col, hue_col, title, filename):
    """
    Generates and saves a scatter plot.
    """
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=df[x_col], y=df[y_col], hue=df[hue_col], palette="coolwarm")
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    path = os.path.join(REPORT_FOLDER, filename)
    plt.savefig(path)
    plt.show()
    plt.close()
    print(f"Scatterplot saved: {path}")


def run_classification(df, features, target='High_Sales'):
    """
    Performs XGBoost classification with hyperparameter tuning,
    and saves confusion matrix and ROC curve plots.
    """
    print("\n🔍 Classification: High Sales Prediction...")
    X = df[features]
    y = df[target]
    
    # Split data into training and testing sets, ensuring stratification for target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=42)

    # Define hyperparameter grid for GridSearchCV
    param_grid = {
        'max_depth': [3, 5],
        'n_estimators': [100, 150],
        'learning_rate': [0.05, 0.1],
    }
    
    # Initialize GridSearchCV with XGBClassifier
    # Removed 'use_label_encoder=False' as it's deprecated and no longer needed
    best_model_finder = GridSearchCV(
        XGBClassifier(eval_metric='logloss', random_state=42),
        param_grid, cv=3, verbose=0, n_jobs=-1 # Use all available cores
    )
    # Fit GridSearchCV to find the best model
    best_model_finder.fit(X_train, y_train)
    best_model = best_model_finder.best_estimator_ # Get the best model
    print("Best Classification Model Params:", best_model_finder.best_params_)

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)
    print(f"✅ Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")

    # Plot and save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = os.path.join(REPORT_FOLDER, "4. classification_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.show()
    plt.close()
    print(f"Confusion Matrix saved to '{cm_path}'")

    # Plot and save ROC Curve
    y_probs = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, color='navy') # Dashed diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(REPORT_FOLDER, "5. classification_roc_curve.png")
    plt.savefig(roc_path)
    plt.show()
    plt.close()
    print(f"ROC Curve saved to '{roc_path}'")


def run_regression(df, features, target='Sales'):
    """
    Performs Gradient Boosting Regression for sales prediction,
    and saves a scatter plot of actual vs. predicted sales.
    """
    print("\n🔧 Regression: Sales Prediction...")
    # Ensure target column and features are present and not NaN for regression
    reg_df = df[features + [target]].dropna()
    X = reg_df[features]
    y = reg_df[target]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    # Initialize and train GradientBoostingRegressor model
    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate and print regression metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Regression Metrics: RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")

    # Plot and save Actual vs Predicted Sales scatter plot
    plt.figure(figsize=(6,5))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) # Diagonal line
    plt.title("Actual vs Predicted Sales")
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    reg_plot_path = os.path.join(REPORT_FOLDER, "6. regression_actual_vs_pred.png")
    plt.savefig(reg_plot_path)
    plt.show()
    plt.close()
    print(f"Actual vs Predicted Sales plot saved to '{reg_plot_path}'")


def run_forecasting(df):
    """
    Performs time series forecasting using Facebook Prophet,
    and saves the forecast plot and component plots.
    """
    print("\n📈 Forecasting with Prophet...")
    # Group by 'Order Date' and sum 'Sales' for time series data
    df_ts = df.groupby('Order Date')['Sales'].sum().reset_index().dropna()
    # Rename columns to 'ds' (datestamp) and 'y' (value) as required by Prophet
    df_ts.columns = ['ds', 'y']
    
    # Initialize and fit Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False # Assuming data is not daily granular enough for this
    )
    model.fit(df_ts)
    
    # Make future dataframe for 30 periods (days)
    future = model.make_future_dataframe(periods=30)
    # Predict future sales
    forecast = model.predict(future)
    
    # Plot and save the main forecast
    fig1 = model.plot(forecast)
    plt.title("Prophet Sales Forecast")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.tight_layout()
    forecast_path = os.path.join(REPORT_FOLDER, "7. forecast_prophet.png")
    fig1.savefig(forecast_path)
    plt.show()
    plt.close(fig1)
    print(f"Prophet Sales Forecast plot saved to '{forecast_path}'")

    # Plot and save Prophet components (trend, seasonality)
    fig2 = model.plot_components(forecast)
    plt.tight_layout()
    components_path = os.path.join(REPORT_FOLDER, "8. forecast_components.png")
    fig2.savefig(components_path)
    plt.show()
    plt.close(fig2)
    print(f"Prophet Forecast Components plot saved to '{components_path}'")


def main():
    """
    Main function to orchestrate the data analysis pipeline.
    """
    ensure_reports_dir()
    file_path = "E:/Project Internship/Mindfluai/data/train.csv" # Ensure this path is correct
    
    # Load and preprocess data
    df = load_fresh_data(file_path)
    df, cat_features = preprocess_data(df)

    # Plot initial scatter plots (using the original df for broad overview)
    plot_scatter(df, 'Category', 'Sales', 'High_Sales', 'Sales by Category (Before Anomaly Removal)', "2. scatter_sales_by_category_original.png")
    plot_scatter(df, 'Region', 'Sales', 'High_Sales', 'Sales by Region (Before Anomaly Removal)', "3. scatter_sales_by_region_original.png")

    # Detect and REMOVE anomalies
    df_no_anomalies = detect_anomalies(df.copy()) # Pass a copy to detect_anomalies if you want the original df for initial plots

    # Define features for machine learning models
    ml_features = cat_features + ['Order_Month', 'Order_Quarter', 'Order_Weekday']

    # Run classification, regression, and forecasting on the anomaly-removed dataset
    run_classification(df_no_anomalies, ml_features)
    run_regression(df_no_anomalies, ml_features)
    run_forecasting(df_no_anomalies)
    
    # Save the final cleaned dataset (without anomalies)
    cleaned_output_path = "Cleaned_dataset_no_anomalies.csv"
    df_no_anomalies.to_csv(cleaned_output_path, index=False)
    print(f"\nFinal cleaned dataset (without anomalies) saved to '{cleaned_output_path}'")

if __name__ == "__main__":
    main()
