import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import datetime

st.set_page_config(page_title="YouTube Monetization Modeler", layout="wide")

@st.cache_data
def load_data():
    # Using the direct raw link to avoid 404 errors
    url = 'https://raw.githubusercontent.com/vinodg-dotcom/Content_Monetization_Modeler/main/cleaned_featured_youtube_data.csv'
    return pd.read_csv(url)

df = load_data()

@st.cache_resource
def load_models():
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    poly_transformer = pickle.load(open('poly_transformer.pkl', 'rb'))
    
    model_files = {
        "Polynomial": 'poly_model.pkl',
        "Ridge": 'ridge_model.pkl',
        "Lasso": 'lasso_model.pkl',
        "SGD": 'sgd_model.pkl',
        "ElasticNet": 'en_model.pkl'
    }
    
    loaded_models = {name: pickle.load(open(path, 'rb')) for name, path in model_files.items()}
    return scaler, poly_transformer, loaded_models

scaler, poly_transformer, models = load_models()

# ---FEATURE ENGINEERING & SELECTION ---
num_features = ['views', 'likes', 'comments', 'watch_time_minutes', 'video_length_minutes', 'subscribers']
date_features = ['year', 'month', 'day']
cols_encoded = ['category_', 'device_', 'country_']


encoded_features = []
for col in df.columns:
    for keyword in cols_encoded:
        if keyword in col:
            encoded_features.append(col)
            break

all_features = num_features + date_features + encoded_features
X = df[all_features]
y = df['ad_revenue_usd']


st.sidebar.header("🕹️ Control Panel")
model_choice = st.sidebar.selectbox("Select Model", list(models.keys()))
current_model = models[model_choice]

st.sidebar.subheader("Video Engagement")
v = st.sidebar.number_input("Views", value=10000, step=1000)
l = st.sidebar.number_input("Likes", value=500, step=50)
c = st.sidebar.number_input("Comments", value=50, step=10)
wt = st.sidebar.number_input("Watch Time (min)", value=20000.0)
len_min = st.sidebar.number_input("Length (min)", value=10.0)
s = st.sidebar.number_input("Subscribers", value=1000)

st.sidebar.subheader("Contextual Data")

countries = sorted([c.replace('country_', '') for c in df.columns if 'country_' in c])
categories = sorted([cat.replace('category_', '') for cat in df.columns if 'category_' in cat])
sel_country = st.sidebar.selectbox("Country", countries)
sel_cat = st.sidebar.selectbox("Category", categories)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_test_scaled = scaler.transform(X_test)
X_test_poly = poly_transformer.transform(X_test_scaled)
test_predictions = current_model.predict(X_test_poly)


st.title("📺 YouTube Revenue Predictor")
st.markdown(f"Currently evaluating performance using the **{model_choice}** model.")

# Metrics Row
m1, m2, m3 = st.columns(3)
m1.metric("R² Accuracy", f"{r2_score(y_test, test_predictions):.4f}")
m2.metric("Mean Absolute Error", f"${mean_absolute_error(y_test, test_predictions):.2f}")
m3.metric("RMSE", f"${np.sqrt(mean_squared_error(y_test, test_predictions)):.2f}")

st.divider()


st.subheader("📊 Visual Analytics")
col_a, col_b = st.columns(2)

with col_a:
    st.write("**Model Accuracy (Actual vs Predicted)**")
    fig1, ax1 = plt.subplots()
    ax1.scatter(y_test, test_predictions, alpha=0.4, color='peru')
    ideal = [y_test.min(), y_test.max()]
    ax1.plot(ideal, ideal, 'r--', lw=2, label="Perfect Fit")
    ax1.set_xlabel("Actual Revenue ($)")
    ax1.set_ylabel("Predicted Revenue ($)")
    ax1.grid(True, linestyle=':', alpha=0.7)
    st.pyplot(fig1)

with col_b:
    st.write("**Top Revenue Drivers (Correlations)**")
    # Finding top 10 numeric correlations
    corrs = df.corr(numeric_only=True)['ad_revenue_usd'].abs().sort_values(ascending=False).head(11)[1:]
    fig2, ax2 = plt.subplots()
    sns.barplot(x=corrs.values, y=corrs.index, palette='viridis', ax=ax2)
    st.pyplot(fig2)

st.divider()


st.subheader("Revenue Estimation")
if st.button("Generate Revenue Prediction"):
    # Prepare base row of zeros
    input_row = pd.DataFrame(0, index=[0], columns=all_features)
    
    # Fill engagement
    input_row[num_features] = [v, l, c, wt, len_min, s]
    
    # Fill current date
    now = datetime.date.today()
    input_row['year'], input_row['month'], input_row['day'] = now.year, now.month, now.day
    
    #  Set the flags for selected country and category
    if f"country_{sel_country}" in input_row.columns:
        input_row[f"country_{sel_country}"] = 1
    if f"category_{sel_cat}" in input_row.columns:
        input_row[f"category_{sel_cat}"] = 1
        
    # Pipeline: Scale -> Poly -> Predict
    user_scaled = scaler.transform(input_row)
    user_poly = poly_transformer.transform(user_scaled)
    final_val = current_model.predict(user_poly)
    
    st.success(f"### Estimated Ad Revenue: ${final_val[0]:,.2f}")
   #st.info("Note: This estimate is based on historical data and the chosen regression model.")