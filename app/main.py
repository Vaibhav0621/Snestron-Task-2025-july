import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="Customer Churn Prediction", layout="wide", page_icon="📊")

# Custom CSS for modern styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    .stApp > div:first-child {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .content-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .prediction-result {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .churn-yes {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
    }
    .churn-no {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
    }
    h1 {
        text-align: center;
        color: white;
        font-size: 3rem;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

def load_pre_trained_models():
    """Load pre-trained models if they exist"""
    models = {}
    if os.path.exists('models'):
        try:
            # Load Random Forest
            if os.path.exists('models/random_forest_model.pkl'):
                with open('models/random_forest_model.pkl', 'rb') as f:
                    models['Random Forest'] = pickle.load(f)
            
            # Load Logistic Regression
            if os.path.exists('models/logistic_regression_model.pkl'):
                with open('models/logistic_regression_model.pkl', 'rb') as f:
                    models['Logistic Regression'] = pickle.load(f)
            
            # Load XGBoost if available
            if os.path.exists('models/xgboost_model.pkl'):
                with open('models/xgboost_model.pkl', 'rb') as f:
                    models['XGBoost'] = pickle.load(f)
            
            # Load feature names
            if os.path.exists('models/feature_names.pkl'):
                with open('models/feature_names.pkl', 'rb') as f:
                    feature_names = pickle.load(f)
                return models, feature_names
                    
        except Exception as e:
            st.error(f"Error loading models: {e}")
    
    return models, None

def load_sample_dataset():
    """Load the sample dataset if it exists"""
    if os.path.exists('data/customer_churn_dataset.csv'):
        return pd.read_csv('data/customer_churn_dataset.csv')
    return None

def preprocess_data_for_prediction(df, target_column='Churn'):
    """Preprocess data for model prediction"""
    # Handle missing values
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Encode categorical variables
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    if target_column in categorical_columns:
        categorical_columns.remove(target_column)
    if 'CustomerID' in categorical_columns:
        categorical_columns.remove('CustomerID')
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    # Drop CustomerID if exists
    if 'CustomerID' in df_encoded.columns:
        df_encoded = df_encoded.drop('CustomerID', axis=1)
    
    return df_encoded

def create_confusion_matrix_plot(y_true, y_pred):
    """Create a confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, 
                    text_auto=True, 
                    aspect="auto",
                    title="Confusion Matrix",
                    labels=dict(x="Predicted", y="Actual"))
    return fig

def create_feature_importance_plot(model, feature_names, model_name):
    """Create feature importance plot"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features
        
        fig = go.Figure(data=[
            go.Bar(x=[feature_names[i] for i in indices][::-1], 
                   y=importances[indices][::-1],
                   orientation='h',
                   marker_color='rgba(102, 126, 234, 0.8)')
        ])
        fig.update_layout(
            title=f"Top 15 Feature Importances - {model_name}",
            xaxis_title="Importance",
            yaxis_title="Features"
        )
        return fig
    elif hasattr(model, 'coef_'):
        coef = abs(model.coef_[0])
        indices = np.argsort(coef)[::-1][:15]
        
        fig = go.Figure(data=[
            go.Bar(x=[feature_names[i] for i in indices][::-1], 
                   y=coef[indices][::-1],
                   orientation='h',
                   marker_color='rgba(102, 126, 234, 0.8)')
        ])
        fig.update_layout(
            title=f"Top 15 Feature Coefficients - {model_name}",
            xaxis_title="Coefficient Magnitude",
            yaxis_title="Features"
        )
        return fig
    return None

# Main App
st.markdown('<h1>🚀 Customer Churn Prediction Platform</h1>', unsafe_allow_html=True)

# Load pre-trained models
pre_trained_models, saved_features = load_pre_trained_models()

# Sidebar for navigation
st.sidebar.title("📋 Navigation")
page = st.sidebar.selectbox("Choose a page", 
                           ["📊 Dataset Analysis", "🤖 Model Training", "🔮 Make Predictions", "📈 Model Comparison"])

if page == "📊 Dataset Analysis":
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.header("📊 Dataset Analysis")
    
    # Option to use sample dataset or upload new one
    data_option = st.radio("Choose data source:", 
                          ["📂 Use Sample Dataset", "📤 Upload New Dataset"])
    
    if data_option == "📂 Use Sample Dataset":
        sample_data = load_sample_dataset()
        if sample_data is not None:
            data = sample_data
            st.success("✅ Sample dataset loaded successfully!")
        else:
            st.error("❌ Sample dataset not found. Please run data_preparation.py first.")
            st.stop()
    else:
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
        else:
            st.info("Please upload a CSV file to proceed.")
            st.stop()
    
    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><h3>{data.shape[0]}</h3><p>Total Customers</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h3>{data.shape[1]}</h3><p>Features</p></div>', unsafe_allow_html=True)
    with col3:
        if 'Churn' in data.columns:
            churn_rate = (data['Churn'] == 'Yes').mean() * 100
            st.markdown(f'<div class="metric-card"><h3>{churn_rate:.1f}%</h3><p>Churn Rate</p></div>', unsafe_allow_html=True)
    with col4:
        missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
        st.markdown(f'<div class="metric-card"><h3>{missing_pct:.1f}%</h3><p>Missing Data</p></div>', unsafe_allow_html=True)
    
    # Data preview
    st.subheader("📋 Data Preview")
    st.dataframe(data.head(10))
    
    # Visualizations
    if 'Churn' in data.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🥧 Churn Distribution")
            churn_counts = data['Churn'].value_counts()
            fig = px.pie(values=churn_counts.values, names=churn_counts.index, 
                        title="Customer Churn Distribution",
                        color_discrete_sequence=['#51cf66', '#ff6b6b'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Churn by Contract Type")
            if 'Contract' in data.columns:
                fig = px.histogram(data, x='Contract', color='Churn', 
                                 title="Churn Distribution by Contract Type",
                                 color_discrete_sequence=['#51cf66', '#ff6b6b'])
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "🤖 Model Training":
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.header("🤖 Model Training & Evaluation")
    
    # Load data
    data_option = st.radio("Choose data source:", 
                          ["📂 Use Sample Dataset", "📤 Upload New Dataset"])
    
    if data_option == "📂 Use Sample Dataset":
        data = load_sample_dataset()
        if data is None:
            st.error("❌ Sample dataset not found. Please run data_preparation.py first.")
            st.stop()
    else:
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
        else:
            st.info("Please upload a CSV file to proceed.")
            st.stop()
    
    if st.button("🚀 Train Models"):
        with st.spinner("Training models... Please wait!"):
            # Preprocess data
            target_col = st.selectbox("Select Target Column", data.columns, 
                                    index=list(data.columns).index('Churn') if 'Churn' in data.columns else 0)
            
            df_processed = preprocess_data_for_prediction(data.copy(), target_col)
            X = df_processed.drop(target_col, axis=1)
            y = df_processed[target_col]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Train models
            models = {}
            results = {}
            
            # Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            rf_proba = rf.predict_proba(X_test)[:, 1]
            models['Random Forest'] = rf
            results['Random Forest'] = {
                'accuracy': rf.score(X_test, y_test),
                'roc_auc': roc_auc_score(y_test, rf_proba),
                'predictions': rf_pred,
                'probabilities': rf_proba
            }
            
            # Logistic Regression
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            lr = LogisticRegression(random_state=42, max_iter=1000)
            lr.fit(X_train_scaled, y_train)
            lr_pred = lr.predict(X_test_scaled)
            lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
            models['Logistic Regression'] = {'model': lr, 'scaler': scaler}
            results['Logistic Regression'] = {
                'accuracy': lr.score(X_test_scaled, y_test),
                'roc_auc': roc_auc_score(y_test, lr_proba),
                'predictions': lr_pred,
                'probabilities': lr_proba
            }
            
            # Store in session state
            st.session_state.models = models
            st.session_state.results = results
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.feature_names = X.columns.tolist()
        
        st.success("✅ Models trained successfully!")
    
    # Display results if available
    if hasattr(st.session_state, 'results'):
        st.subheader("📊 Model Performance")
        
        col1, col2 = st.columns(2)
        for i, (name, result) in enumerate(st.session_state.results.items()):
            with col1 if i % 2 == 0 else col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{name}</h3>
                    <p>Accuracy: {result['accuracy']:.3f}</p>
                    <p>ROC-AUC: {result['roc_auc']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Feature importance
        selected_model = st.selectbox("Select model for feature importance:", 
                                    list(st.session_state.models.keys()))
        if selected_model:
            model = st.session_state.models[selected_model]
            if isinstance(model, dict):
                model = model['model']
            
            fig = create_feature_importance_plot(model, st.session_state.feature_names, selected_model)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrices
        st.subheader("🎯 Confusion Matrices")
        col1, col2 = st.columns(2)
        for i, (name, result) in enumerate(st.session_state.results.items()):
            with col1 if i % 2 == 0 else col2:
                fig = create_confusion_matrix_plot(st.session_state.y_test, result['predictions'])
                fig.update_layout(title=f"Confusion Matrix - {name}")
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "🔮 Make Predictions":
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.header("🔮 Make Churn Predictions")
    
    # Use pre-trained models if available
    if pre_trained_models:
        st.success(f"✅ Found {len(pre_trained_models)} pre-trained models!")
        
        # Model selection
        selected_model_name = st.selectbox("Choose a model:", list(pre_trained_models.keys()))
        selected_model = pre_trained_models[selected_model_name]
        
        st.subheader("📝 Enter Customer Information")
        
        # Create input form based on common features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
        
        with col2:
            tenure = st.slider("Tenure (months)", 1, 72, 12)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        
        with col3:
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", 
                                        ["Electronic check", "Mailed check", 
                                         "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly_charges = st.number_input("Monthly Charges", 18.0, 120.0, 65.0)
            total_charges = st.number_input("Total Charges", 18.0, 9000.0, 1000.0)
        
        if st.button("🔮 Predict Churn"):
            # Create input dataframe
            input_data = {
                'Gender': gender,
                'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'InternetService': internet_service,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            
            # Add dummy values for other features that might be needed
            for feature in ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
                input_data[feature] = "No"
            
            input_df = pd.DataFrame([input_data])
            
            # Preprocess
            processed_df = preprocess_data_for_prediction(input_df)
            
            # Align features with saved model features
            if saved_features:
                # Add missing columns with default values
                for feature in saved_features:
                    if feature not in processed_df.columns:
                        processed_df[feature] = 0
                
                # Reorder columns to match training data
                processed_df = processed_df[saved_features]
            
            # Make prediction
            try:
                if isinstance(selected_model, dict) and 'model' in selected_model:
                    # Logistic regression with scaler
                    scaled_input = selected_model['scaler'].transform(processed_df)
                    prediction = selected_model['model'].predict(scaled_input)[0]
                    probability = selected_model['model'].predict_proba(scaled_input)[0][1]
                else:
                    # Other models
                    prediction = selected_model.predict(processed_df)[0]
                    probability = selected_model.predict_proba(processed_df)[0][1]
                
                # Display result
                if prediction == 1 or prediction == "Yes":
                    st.markdown(f"""
                    <div class="prediction-result churn-yes">
                        ⚠️ HIGH CHURN RISK<br>
                        Probability: {probability:.2%}
                    </div>
                    """, unsafe_allow_html=True)
                    st.error("This customer is likely to churn. Consider retention strategies!")
                else:
                    st.markdown(f"""
                    <div class="prediction-result churn-no">
                        ✅ LOW CHURN RISK<br>
                        Probability: {probability:.2%}
                    </div>
                    """, unsafe_allow_html=True)
                    st.success("This customer is likely to stay. Keep up the good service!")
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info("This might be due to feature mismatch. Try retraining the model with your data.")
    
    else:
        st.warning("⚠️ No pre-trained models found. Please train models first or run data_preparation.py")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "📈 Model Comparison":
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.header("📈 Model Comparison Dashboard")
    
    if pre_trained_models:
        st.success(f"✅ Comparing {len(pre_trained_models)} pre-trained models")
        
        # Load sample data for evaluation
        sample_data = load_sample_dataset()
        if sample_data is not None:
            # Preprocess data
            df_processed = preprocess_data_for_prediction(sample_data.copy())
            X = df_processed.drop('Churn', axis=1)
            y = df_processed['Churn']
            
            # Align features
            if saved_features:
                for feature in saved_features:
                    if feature not in X.columns:
                        X[feature] = 0
                X = X[saved_features]
            
            # Split for evaluation
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Evaluate all models
            comparison_results = {}
            for name, model in pre_trained_models.items():
                try:
                    if isinstance(model, dict) and 'model' in model:
                        # Logistic regression with scaler
                        X_test_scaled = model['scaler'].transform(X_test)
                        pred = model['model'].predict(X_test_scaled)
                        proba = model['model'].predict_proba(X_test_scaled)[:, 1]
                        accuracy = model['model'].score(X_test_scaled, y_test)
                    else:
                        # Other models
                        pred = model.predict(X_test)
                        proba = model.predict_proba(X_test)[:, 1]
                        accuracy = model.score(X_test, y_test)
                    
                    comparison_results[name] = {
                        'accuracy': accuracy,
                        'roc_auc': roc_auc_score(y_test, proba),
                        'predictions': pred,
                        'probabilities': proba
                    }
                except Exception as e:
                    st.error(f"Error evaluating {name}: {e}")
            
            # Display comparison
            if comparison_results:
                # Performance metrics
                metrics_df = pd.DataFrame(comparison_results).T
                st.subheader("📊 Performance Metrics")
                st.dataframe(metrics_df[['accuracy', 'roc_auc']].round(4))
                
                # Visual comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(x=list(comparison_results.keys()), 
                               y=[r['accuracy'] for r in comparison_results.values()],
                               title="Model Accuracy Comparison",
                               color=[r['accuracy'] for r in comparison_results.values()],
                               color_continuous_scale='viridis')
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(x=list(comparison_results.keys()), 
                               y=[r['roc_auc'] for r in comparison_results.values()],
                               title="Model ROC-AUC Comparison",
                               color=[r['roc_auc'] for r in comparison_results.values()],
                               color_continuous_scale='plasma')
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Best model recommendation
                best_model = max(comparison_results.keys(), 
                               key=lambda x: comparison_results[x]['roc_auc'])
                st.success(f"🏆 Best performing model: **{best_model}** "
                          f"(ROC-AUC: {comparison_results[best_model]['roc_auc']:.4f})")
        
        else:
            st.error("❌ Sample dataset not found for comparison.")
    
    else:
        st.warning("⚠️ No pre-trained models found for comparison.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white;'>
    <p>🚀 <strong>Snestron Internship Project</strong> | Modern Customer Churn Prediction Platform</p>
    <p>Built with ❤️ using Streamlit, Scikit-learn, and Plotly</p>
</div>
""", unsafe_allow_html=True)
