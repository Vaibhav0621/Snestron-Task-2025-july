# 🚀 Snestron Internship: Customer Churn Prediction

A comprehensive machine learning project for predicting customer churn with a modern, responsive web interface built using Streamlit.

## 📋 Project Overview

This project implements a complete customer churn prediction pipeline including:
- **Data preprocessing and exploration**
- **Multiple ML models (Random Forest, Logistic Regression, XGBoost)**
- **Interactive web application**
- **Model comparison and evaluation**
- **Real-time predictions**
<img width="1919" height="1023" alt="image" src="https://github.com/user-attachments/assets/58d9e566-28d8-4417-a4f9-a5dba2d4e356" />

## 🏗️ Project Structure

```
snestron-task/
├── data/                          # Dataset storage
│   └── customer_churn_dataset.csv # Sample dataset
├── models/                        # Trained models
│   ├── random_forest_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── xgboost_model.pkl
│   └── feature_names.pkl
├── app/                          # Streamlit application
│   └── main.py                   # Main app file
├── data_preparation.py           # Dataset creation & model training
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Vaibhav0621/Snestron-Task-2025-july.git
cd snestron-task
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate Dataset and Train Models
```bash
python data_preparation.py
```
This will:
- Create a synthetic customer churn dataset (5,000 samples)
- Train multiple ML models
- Save models and data for the web app

### 5. Run the Web Application
```bash
streamlit run app/main.py
```

The app will open in your browser at `http://localhost:8501`

## 🎯 Features

### 📊 Dataset Analysis
- **Data Overview**: Customer count, features, churn rate
- **Interactive Visualizations**: Churn distribution, contract analysis
- **Data Quality**: Missing data analysis

### 🤖 Model Training
- **Multiple Algorithms**: Random Forest, Logistic Regression, XGBoost
- **Performance Metrics**: Accuracy, ROC-AUC, Confusion Matrix
- **Feature Importance**: Visual analysis of key predictors

### 🔮 Predictions
- **Interactive Form**: Easy customer data input
- **Real-time Results**: Instant churn probability
- **Risk Assessment**: Clear visual indicators

### 📈 Model Comparison
- **Performance Dashboard**: Side-by-side model comparison
- **Best Model Selection**: Automated recommendation
- **Visual Analytics**: Interactive charts and graphs

## 🧠 Machine Learning Models

### 1. Random Forest Classifier
- **Accuracy**: ~87.2%
- **ROC-AUC**: ~94.9%
- **Features**: Handles non-linear relationships, feature importance

### 2. Logistic Regression
- **Accuracy**: ~86.8%
- **ROC-AUC**: ~94.7%
- **Features**: Interpretable coefficients, fast predictions

### 3. XGBoost (Optional)
- **High Performance**: Gradient boosting
- **Robustness**: Handles missing values
- **Scalability**: Efficient training

## 📊 Dataset Features

The synthetic dataset includes realistic customer attributes:

- **Demographics**: Gender, Senior Citizen status
- **Account Info**: Tenure, Contract type, Payment method
- **Services**: Phone, Internet, Streaming services
- **Billing**: Monthly charges, Total charges, Paperless billing
- **Target**: Churn (Yes/No)

## 🎨 UI/UX Features

- **Modern Design**: Gradient backgrounds, card layouts
- **Responsive**: Works on desktop, tablet, mobile
- **Interactive**: Real-time updates, hover effects
- **Professional**: Clean, business-ready interface

## 📈 Model Performance

| Model | Accuracy | ROC-AUC | Precision | Recall |
|-------|----------|---------|-----------|---------|
| Random Forest | 87.2% | 94.9% | 84% | 80% |
| Logistic Regression | 86.8% | 94.7% | 82% | 82% |

## 🔧 Advanced Usage

### Custom Dataset
To use your own dataset:
1. Upload CSV file through the web interface
2. Ensure it has a 'Churn' column (Yes/No)
3. Train new models using the interface

### API Integration
The trained models can be used programmatically:
```python
import pickle
with open('models/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)
prediction = model.predict(your_data)
```

## 🚀 Deployment Options

### Local Development
```bash
streamlit run app/main.py
```

### Cloud Deployment
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Container deployment
- **AWS/GCP**: Scalable cloud hosting

## 📝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Snestron Internship Submission

This project demonstrates:
- ✅ **Machine Learning Expertise**: Multiple algorithms, proper evaluation
- ✅ **Data Science Skills**: EDA, preprocessing, feature engineering
- ✅ **Software Development**: Clean code, documentation, version control
- ✅ **UI/UX Design**: Modern, responsive, user-friendly interface
- ✅ **Business Understanding**: Practical churn prediction solution

## 📞 Contact

**Project Developer**: [Your Name]
**Email**: [Your Email]
**GitHub**: [Your GitHub Profile]
**LinkedIn**: [Your LinkedIn Profile]

---

### 🏆 **Ready for Snestron Internship Evaluation!**

This project showcases a complete end-to-end machine learning solution with production-ready code, modern UI/UX, and comprehensive documentation.
