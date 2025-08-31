"""
Data Preparation Script for Customer Churn Prediction
Downloads dataset and prepares it for model training
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pickle
import os

def download_sample_data():
    """Create a sample customer churn dataset"""
    print("Creating sample customer churn dataset...")
    
    np.random.seed(42)
    n_samples = 5000
    
    # Generate synthetic customer churn data
    data = {
        'CustomerID': range(1, n_samples + 1),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'tenure': np.random.randint(1, 73, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.4, 0.5, 0.1]),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.35, 0.45, 0.2]),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.35, 0.45, 0.2]),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.25, 0.55, 0.2]),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.4, 0.4, 0.2]),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.4, 0.4, 0.2]),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.25, 0.2]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], 
                                        n_samples, p=[0.35, 0.2, 0.25, 0.2]),
        'MonthlyCharges': np.round(np.random.uniform(18.25, 118.75, n_samples), 2),
        'TotalCharges': np.round(np.random.uniform(18.8, 8684.8, n_samples), 2),
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic churn based on features
    churn_prob = np.zeros(n_samples)
    
    # Higher churn for month-to-month contracts
    churn_prob += (df['Contract'] == 'Month-to-month') * 0.3
    # Higher churn for senior citizens
    churn_prob += df['SeniorCitizen'] * 0.15
    # Higher churn for electronic check payment
    churn_prob += (df['PaymentMethod'] == 'Electronic check') * 0.2
    # Higher churn for no dependents
    churn_prob += (df['Dependents'] == 'No') * 0.1
    # Higher churn for fiber optic
    churn_prob += (df['InternetService'] == 'Fiber optic') * 0.15
    # Lower churn for longer tenure
    churn_prob -= (df['tenure'] > 24) * 0.2
    # Add some randomness
    churn_prob += np.random.uniform(0, 0.3, n_samples)
    
    # Create churn labels
    df['Churn'] = (churn_prob > 0.5).astype(int)
    df['Churn'] = df['Churn'].map({0: 'No', 1: 'Yes'})
    
    return df

def preprocess_data(df):
    """Preprocess the dataset for model training"""
    print("Preprocessing data...")
    
    # Handle missing values
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Encode categorical variables
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    if 'Churn' in categorical_columns:
        categorical_columns.remove('Churn')
    if 'CustomerID' in categorical_columns:
        categorical_columns.remove('CustomerID')
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    # Drop CustomerID if exists
    if 'CustomerID' in df_encoded.columns:
        df_encoded = df_encoded.drop('CustomerID', axis=1)
    
    return df_encoded

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and save them"""
    print("Training models...")
    
    models = {}
    results = {}
    
    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    models['random_forest'] = rf
    results['random_forest'] = {
        'accuracy': rf.score(X_test, y_test),
        'roc_auc': roc_auc_score(y_test, rf_proba),
        'classification_report': classification_report(y_test, rf_pred)
    }
    
    # Logistic Regression
    print("Training Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
    models['logistic_regression'] = {'model': lr, 'scaler': scaler}
    results['logistic_regression'] = {
        'accuracy': lr.score(X_test_scaled, y_test),
        'roc_auc': roc_auc_score(y_test, lr_proba),
        'classification_report': classification_report(y_test, lr_pred)
    }
    
    # XGBoost (if available)
    try:
        from xgboost import XGBClassifier
        print("Training XGBoost...")
        xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        xgb_proba = xgb.predict_proba(X_test)[:, 1]
        models['xgboost'] = xgb
        results['xgboost'] = {
            'accuracy': xgb.score(X_test, y_test),
            'roc_auc': roc_auc_score(y_test, xgb_proba),
            'classification_report': classification_report(y_test, xgb_pred)
        }
    except ImportError:
        print("XGBoost not available, skipping...")
    
    return models, results

def save_models_and_data(models, X_train, feature_names, dataset_path):
    """Save trained models and feature information"""
    print("Saving models...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save models
    for name, model in models.items():
        with open(f'models/{name}_model.pkl', 'wb') as f:
            pickle.dump(model, f)
    
    # Save feature names for consistency
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    # Save sample data for reference
    with open('models/sample_data.pkl', 'wb') as f:
        pickle.dump(X_train.iloc[:5], f)
    
    print(f"Dataset saved to: {dataset_path}")
    print("Models saved to: models/ directory")

def main():
    """Main function to prepare data and train models"""
    print("Starting data preparation and model training...")
    
    # Create or download dataset
    df = download_sample_data()
    
    # Save raw dataset
    dataset_path = 'data/customer_churn_dataset.csv'
    os.makedirs('data', exist_ok=True)
    df.to_csv(dataset_path, index=False)
    
    # Preprocess data
    df_processed = preprocess_data(df.copy())
    
    # Prepare features and target
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models
    models, results = train_models(X_train, X_test, y_train, y_test)
    
    # Print results
    print("\n" + "="*50)
    print("MODEL TRAINING RESULTS")
    print("="*50)
    for name, result in results.items():
        print(f"\n{name.upper()} RESULTS:")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"ROC-AUC: {result['roc_auc']:.4f}")
        print("Classification Report:")
        print(result['classification_report'])
    
    # Save everything
    save_models_and_data(models, X_train, X.columns.tolist(), dataset_path)
    
    print("\n" + "="*50)
    print("SETUP COMPLETE!")
    print("="*50)
    print("✅ Dataset created and saved")
    print("✅ Models trained and saved")
    print("✅ Ready to run Streamlit app!")
    print("\nNext steps:")
    print("1. Run: streamlit run app/main.py")
    print("2. Upload the dataset: data/customer_churn_dataset.csv")
    print("3. Explore the trained models and predictions!")

if __name__ == "__main__":
    main()
