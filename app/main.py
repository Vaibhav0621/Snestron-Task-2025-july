import streamlit as st
import plotly.express as px
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
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        color: white;
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        border-radius: 8px;
        font-size: 18px;
        padding: 0.5em 2em;
    }
    .stTextInput>div>input {
        border-radius: 8px;
        border: 1px solid #0072ff;
    }
    </style>
""", unsafe_allow_html=True)
st.title("Customer Churn Prediction")
st.markdown("""
#### Upload your customer churn dataset to get started.
""")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(data.head())
    st.subheader("Data Summary")
    st.write(data.describe())
    st.write("Shape:", data.shape)
    st.write("Missing Values:")
    st.write(data.isnull().sum())
    st.write("Class Distribution:")
    if 'Churn' in data.columns:
        churn_counts = data['Churn'].value_counts()
        fig = px.pie(names=churn_counts.index, values=churn_counts.values, title="Churn Distribution")
        st.plotly_chart(fig)
    else:
        st.warning("No 'Churn' column found. Please ensure your dataset has a 'Churn' column.")

    st.subheader("Feature Correlation")
    corr = data.corr(numeric_only=True)
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax, cmap='Blues')
    st.pyplot(fig)

    st.subheader("Preprocessing & Model Training")
    target_col = st.selectbox("Select Target Column (Churn)", options=data.columns, index=list(data.columns).index('Churn') if 'Churn' in data.columns else 0)
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
    random_state = st.number_input("Random State", value=42)
    model_type = st.selectbox("Select Model", options=["Random Forest", "Logistic Regression"] + (["XGBoost"] if xgb_available else []))

    # Simple preprocessing: drop NA, encode categoricals
    df = data.dropna()
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    # Encode categoricals
    X = pd.get_dummies(X)
    if y.dtype == 'O':
        y = y.map({"Yes": 1, "No": 0}) if set(y.unique()) == {"Yes", "No"} else pd.factorize(y)[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))
    if model_type == "Random Forest":
        model = RandomForestClassifier(random_state=int(random_state))
    elif model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=int(random_state))
    elif model_type == "XGBoost" and xgb_available:
        model = XGBClassifier(random_state=int(random_state), use_label_encoder=False, eval_metric='logloss')
    else:
        st.error("Selected model is not available.")
        st.stop()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else None

    st.subheader("Model Evaluation")
    st.text(classification_report(y_test, y_pred))
    if y_proba is not None:
        st.write("ROC-AUC:", roc_auc_score(y_test, y_proba))
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc = px.area(x=fpr, y=tpr, title='ROC Curve', labels={'x':'False Positive Rate', 'y':'True Positive Rate'})
        st.plotly_chart(fig_roc)
    # Feature Importance
    st.subheader("Feature Importance")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        fig_imp = px.bar(x=X.columns, y=importances, title="Feature Importances")
        st.plotly_chart(fig_imp)
    elif hasattr(model, 'coef_'):
        fig_imp = px.bar(x=X.columns, y=model.coef_[0], title="Feature Coefficients")
        st.plotly_chart(fig_imp)
    st.success("Model trained! You can now use it for predictions.")

    st.subheader("Predict Churn for New Customer")
    input_dict = {}
    cols = st.columns(len(X.columns))
    for idx, col in enumerate(X.columns):
        val = cols[idx].text_input(f"{col}", value="0")
        input_dict[col] = val
    if st.button("Predict Churn"):
        input_df = pd.DataFrame([input_dict])
        input_df = input_df.astype(X.dtypes.to_dict())
        pred = model.predict(input_df)[0]
        st.write(f"Prediction: {'Churn' if pred == 1 else 'No Churn'}")
else:
    st.info("Please upload a CSV file to proceed.")

st.markdown("---")
st.caption("Snestron Internship Project | Modern, Responsive UI")
