"""
YouTube Ad Revenue Prediction - Streamlit App
==============================================
Predict YouTube ad revenue using machine learning
(Updated: Compatible with One-Hot Encoding)
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="YouTube Revenue Predictor",
    page_icon="🎥",
    layout="wide"
)

# ============================================================
# LOAD MODELS & DATA (WITH CACHING FOR SPEED)
# ============================================================

@st.cache_resource
def load_all_models():
    """
    Load trained machine learning models and preprocessing tools
    This function runs only once and stores results in memory
    """
    try:
        with open('models/all_models.pkl', 'rb') as f:
            all_models = pickle.load(f)
        
        with open('models/results.pkl', 'rb') as f:
            results = pickle.load(f)
        
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('models/encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        
        with open('models/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        
        return all_models, results, scaler, encoders, model_info
    
    except FileNotFoundError:
        st.error("❌ Model files not found!")
        st.info("💡 Please run 'train1.py' first to create the models.")
        st.stop()

@st.cache_data
def load_dataset():
    """Load the cleaned dataset for showing statistics"""
    try:
        return pd.read_csv('data/cleaned_data.csv')
    except FileNotFoundError:
        return None

# Load everything
all_models, results, scaler, encoders, model_info = load_all_models()
df = load_dataset()

# ============================================================
# CREATE FEATURES FUNCTION (UPDATED FOR ONE-HOT ENCODING)
# ============================================================

def create_all_features(views, likes, comments, video_length, subscribers,
                        category, device, country):
    """
    Create all features with one-hot encoding to match training data
    
    The training script created binary columns for each category/device/country.
    We need to recreate the exact same column structure here.
    
    Total features: 31
    - 5 basic metrics (views, likes, comments, video_length, subscribers)
    - 6 category binary columns
    - 4 device binary columns
    - 6 country binary columns
    - 4 basic engagement features
    - 3 advanced engineered features
    - 3 log transformations
    """
    
    # Basic engagement features
    engagement_rate = (likes + comments) / (views + 1)
    like_rate = likes / (views + 1)
    comment_rate = comments / (views + 1)
    views_per_subscriber = views / (subscribers + 1)
    
    # Advanced features
    engagement_quality = (likes * 1 + comments * 3) / (views + 1)
    video_popularity = views / (video_length + 1)
    views_likes_interaction = views * like_rate
    
    # Log transformations
    log_views = np.log1p(views)
    log_subscribers = np.log1p(subscribers)
    log_likes = np.log1p(likes)
    
    # Start building feature dictionary
    feature_dict = {
        'views': views,
        'likes': likes,
        'comments': comments,
        'video_length_minutes': video_length,
        'subscribers': subscribers
    }
    
    # One-hot encode category (6 categories)
    for cat_name in encoders['category']:
        feature_dict[f'category_{cat_name}'] = 1 if cat_name == category else 0
    
    # One-hot encode device (4 devices)
    for dev_name in encoders['device']:
        feature_dict[f'device_{dev_name}'] = 1 if dev_name == device else 0
    
    # One-hot encode country (6 countries)
    for country_name in encoders['country']:
        feature_dict[f'country_{country_name}'] = 1 if country_name == country else 0
    
    # Add engineered features
    feature_dict['engagement_rate'] = engagement_rate
    feature_dict['like_rate'] = like_rate
    feature_dict['comment_rate'] = comment_rate
    feature_dict['views_per_subscriber'] = views_per_subscriber
    feature_dict['engagement_quality'] = engagement_quality
    feature_dict['video_popularity'] = video_popularity
    feature_dict['views_likes_interaction'] = views_likes_interaction
    feature_dict['log_views'] = log_views
    feature_dict['log_subscribers'] = log_subscribers
    feature_dict['log_likes'] = log_likes
    
    # Convert to DataFrame (models expect DataFrame input)
    df_features = pd.DataFrame([feature_dict])
    
    return df_features

# ============================================================
# PREDICTION FUNCTION (UPDATED FOR ONE-HOT ENCODING)
# ============================================================

def predict_revenue(model_name, features_df):
    """
    Make prediction using selected model
    
    features_df: DataFrame with one-hot encoded columns (31 features)
    """
    
    needs_scaling = [
        'Linear Regression',
        'L2 (Ridge) Regression',
        'L1 (Lasso) Regression',
        'SGD Regressor'
    ]
    
    try:
        if model_name in needs_scaling:
            # Scale the features
            scaled_features = scaler.transform(features_df)
            prediction = all_models[model_name].predict(scaled_features)[0]
        else:
            # Polynomial model has its own preprocessing
            prediction = all_models[model_name].predict(features_df)[0]
        
        # Ensure non-negative
        prediction = max(0, prediction)
        
        return prediction
    
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        st.error(f"Expected features: {len(features_df.columns)}")
        st.error(f"Feature names (first 10): {list(features_df.columns)[:10]}...")
        return None

# ============================================================
# HELPER FUNCTION: CONVERT DATAFRAME TO CSV
# ============================================================

def convert_df_to_csv(df):
    """Convert DataFrame to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')

# ============================================================
# HEADER
# ============================================================
st.title("🎥 YouTube Ad Revenue Predictor")
st.markdown("### Predict how much money a YouTube video might earn from ads")
st.markdown("---")

st.success(f"✅ Successfully loaded {len(all_models)} machine learning models!")

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.header("📋 Menu")
page = st.sidebar.radio(
    "Choose a page:",
    ["Make Prediction", "View Analytics"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏆 Best Performing Model")
best_model_name = model_info.get('model_name', 'Linear Regression')
best_r2 = model_info.get('r2_score', 0)
st.sidebar.info(f"**{best_model_name}**  \nR² Score: {best_r2:.4f}")

# ============================================================
# PAGE 1: MAKE PREDICTION
# ============================================================
if page == "Make Prediction":
    
    st.header("🎯 Make a Revenue Prediction")
    
    # MODEL SELECTION
    st.subheader("1️⃣ Choose a Model")
    
    selected_model = st.selectbox(
        "Select which machine learning model to use:",
        list(all_models.keys()),
        help="All models are trained on the same data but use different algorithms"
    )
    
    # Show selected model's performance
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{results[selected_model]['R2']:.4f}")
    col2.metric("RMSE", f"${results[selected_model]['RMSE']:.2f}")
    col3.metric("MAE", f"${results[selected_model]['MAE']:.2f}")
    
    st.markdown("---")
    
    # INPUT FORM
    st.subheader("2️⃣ Enter Video Details")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("**📊 Video Performance Metrics**")
        
        views = st.number_input(
            "👁️ Views",
            min_value=0,
            value=10000,
            step=100,
            help="How many times the video was watched"
        )
        
        likes = st.number_input(
            "👍 Likes",
            min_value=0,
            value=500,
            step=10,
            help="Number of likes (typically 2-10% of views)"
        )
        
        comments = st.number_input(
            "💬 Comments",
            min_value=0,
            value=100,
            step=5,
            help="Number of comments (typically 0.5-2% of views)"
        )
    
    with col_right:
        st.markdown("**📺 Video & Channel Information**")
        
        video_length = st.number_input(
            "⏱️ Video Length (minutes)",
            min_value=0.1,
            value=10.0,
            step=0.5,
            help="How long is the video?"
        )
        
        subscribers = st.number_input(
            "👥 Channel Subscribers",
            min_value=0,
            value=50000,
            step=1000,
            help="Total subscribers on the channel"
        )
    
    st.info("""
    💡 **Note:** This model predicts revenue based on engagement metrics and video 
    characteristics. Predictions are estimates that help guide content strategy decisions.
    """)
    
    st.markdown("---")
    
    # CATEGORICAL INPUTS (UPDATED FOR ONE-HOT ENCODING)
    st.markdown("**🏷️ Categories**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        category = st.selectbox(
            "📂 Video Category",
            options=encoders['category'],  # Now it's a list, not a LabelEncoder object
            help="What type of content is the video?"
        )
    
    with col2:
        device = st.selectbox(
            "📱 Primary Device Type",
            options=encoders['device'],
            help="Most common device viewers use"
        )
    
    with col3:
        country = st.selectbox(
            "🌍 Primary Country",
            options=encoders['country'],
            help="Country with most viewers (affects ad rates)"
        )
    
    st.markdown("---")
    
    # VALIDATION
    show_warning = False
    warning_messages = []
    
    if likes > views:
        st.error("❌ Error: Likes cannot be more than views!")
        show_warning = True
    
    if comments > views:
        st.error("❌ Error: Comments cannot be more than views!")
        show_warning = True
    
    if views > 0:
        like_rate_pct = (likes / views) * 100
        if like_rate_pct > 20:
            warning_messages.append(f"⚠️ Like rate is {like_rate_pct:.1f}% (typically 2-10%)")
    
    if warning_messages:
        with st.expander("⚠️ Warnings (click to see)"):
            for msg in warning_messages:
                st.warning(msg)
    
    # PREDICT BUTTON
    st.subheader("3️⃣ Get Prediction")
    
    predict_btn = st.button(
        "💰 Predict Ad Revenue",
        type="primary",
        disabled=show_warning,
        use_container_width=True
    )
    
    if predict_btn:
        with st.spinner("🔮 Calculating prediction..."):
            
            # Create features with one-hot encoding
            # (No need to manually encode - function handles it)
            all_features = create_all_features(
                views, likes, comments, video_length, subscribers,
                category, device, country  # Pass string values directly
            )
            
            # Make prediction
            prediction = predict_revenue(selected_model, all_features)
            
            if prediction is not None:
                # Show the result!
                st.success(f"## 💰 Predicted Revenue: ${prediction:.2f}")
                
                # Show engagement metrics
                st.markdown("---")
                st.markdown("**📊 Your Video's Engagement Metrics:**")
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                engagement_rate = (likes + comments) / (views + 1) if views > 0 else 0
                like_rate = (likes / views) if views > 0 else 0
                comment_rate = (comments / views) if views > 0 else 0
                views_per_sub = (views / subscribers) if subscribers > 0 else 0
                
                metric_col1.metric("Engagement Rate", f"{engagement_rate*100:.2f}%")
                metric_col2.metric("Like Rate", f"{like_rate*100:.2f}%")
                metric_col3.metric("Comment Rate", f"{comment_rate*100:.2f}%")
                metric_col4.metric("Views per Subscriber", f"{views_per_sub:.2f}")
                
                # Compare with dataset average
                if df is not None:
                    avg_revenue = df['ad_revenue_usd'].mean()
                    difference = prediction - avg_revenue
                    
                    st.markdown("---")
                    st.markdown("**📈 Comparison with Dataset:**")
                    
                    comp_col1, comp_col2 = st.columns(2)
                    comp_col1.metric("Dataset Average Revenue", f"${avg_revenue:.2f}")
                    comp_col2.metric("Your Prediction vs Average", f"${difference:+.2f}")
                
                # Download prediction report
                st.markdown("---")
                st.markdown("**📥 Download Prediction Report:**")
                
                # Create prediction report DataFrame
                prediction_report = pd.DataFrame({
                    'Metric': [
                        'Prediction Date',
                        'Model Used',
                        'Views',
                        'Likes',
                        'Comments',
                        'Video Length (min)',
                        'Subscribers',
                        'Category',
                        'Device',
                        'Country',
                        'Predicted Revenue',
                        'Engagement Rate',
                        'Like Rate',
                        'Comment Rate',
                        'Views per Subscriber'
                    ],
                    'Value': [
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        selected_model,
                        f"{views:,}",
                        f"{likes:,}",
                        f"{comments:,}",
                        f"{video_length:.1f}",
                        f"{subscribers:,}",
                        category,
                        device,
                        country,
                        f"${prediction:.2f}",
                        f"{engagement_rate*100:.2f}%",
                        f"{like_rate*100:.2f}%",
                        f"{comment_rate*100:.2f}%",
                        f"{views_per_sub:.3f}"
                    ]
                })
                
                # Convert to CSV
                csv = convert_df_to_csv(prediction_report)
                
                # Download button
                st.download_button(
                    label="📥 Download Prediction as CSV",
                    data=csv,
                    file_name=f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download your prediction results as a CSV file"
                )
                
                # Option to compare all models
                st.markdown("---")
                if st.checkbox("🔄 Show predictions from all 5 models"):
                    st.markdown("**Predictions from all models:**")
                    
                    comparison_results = []
                    for model_name in all_models.keys():
                        pred = predict_revenue(model_name, all_features)
                        if pred is not None:
                            comparison_results.append({
                                'Model': model_name,
                                'Prediction': f"${pred:.2f}",
                                'R² Score': f"{results[model_name]['R2']:.4f}"
                            })
                    
                    comparison_df = pd.DataFrame(comparison_results)
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    # Download all model comparisons
                    csv_comparison = convert_df_to_csv(comparison_df)
                    st.download_button(
                        label="📥 Download All Model Predictions",
                        data=csv_comparison,
                        file_name=f"all_models_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Download predictions from all 5 models"
                    )

# ============================================================
# PAGE 2: VIEW ANALYTICS
# ============================================================
elif page == "View Analytics":
    
    st.header("📊 Dataset Analytics")
    
    if df is None:
        st.error("❌ Dataset not found!")
        st.stop()
    
    # OVERVIEW STATISTICS
    st.subheader("📈 Dataset Overview")
    
    overview_col1, overview_col2, overview_col3 = st.columns(3)
    overview_col1.metric("Total Videos", f"{len(df):,}")
    overview_col2.metric("Average Revenue", f"${df['ad_revenue_usd'].mean():.2f}")
    overview_col3.metric("Average Views", f"{df['views'].mean():,.0f}")
    
    st.markdown("---")
    
    # ============================================================
    # CORRELATION ANALYSIS
    # ============================================================
    st.subheader("🔗 Correlation Analysis")
    st.markdown("**See which features are most related to revenue:**")
    
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate correlation matrix
    correlation_matrix = df[numeric_cols].corr()
    
    # Create correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix, 
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title('Correlation Heatmap - How Features Relate to Each Other', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show top correlations with revenue
    if 'ad_revenue_usd' in correlation_matrix.columns:
        st.markdown("**📊 Features Most Correlated with Revenue:**")
        
        revenue_corr = correlation_matrix['ad_revenue_usd'].sort_values(ascending=False)
        revenue_corr = revenue_corr[revenue_corr.index != 'ad_revenue_usd']
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['green' if x > 0 else 'red' for x in revenue_corr.values]
        bars = ax.barh(revenue_corr.index, revenue_corr.values, color=colors, edgecolor='black')
        ax.set_xlabel('Correlation Coefficient', fontsize=12)
        ax.set_title('Feature Correlations with Revenue', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, revenue_corr.values)):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f' {value:.3f}', 
                   ha='left' if value > 0 else 'right', 
                   va='center', 
                   fontsize=9,
                   fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Insights
        strongest_positive = revenue_corr.idxmax()
        strongest_negative = revenue_corr.idxmin()
        
        st.success(f"""
        **Key Insights:**
        - 🟢 Strongest positive correlation: **{strongest_positive}** ({revenue_corr.max():.3f})
        - 🔴 Strongest negative correlation: **{strongest_negative}** ({revenue_corr.min():.3f})
        
        💡 Features with higher correlation (closer to 1 or -1) have stronger relationships with revenue.
        """)
    
    st.markdown("---")
    
    # ============================================================
    # SCATTER PLOTS (SIMPLE VERSION - NO BUBBLES)
    # ============================================================
    st.subheader("📈 Relationship Analysis (Scatter Plots)")
    st.markdown("**See how different metrics relate to revenue:**")
    
    # Sample data for clearer visualization (1000 points instead of 5000)
    if len(df) > 1000:
        df_sample = df.sample(n=1000, random_state=42)
        st.info(f"ℹ️ Showing 1,000 randomly sampled videos out of {len(df):,} total for clearer visualization")
    else:
        df_sample = df
    
    scatter_col1, scatter_col2 = st.columns(2)
    
    with scatter_col1:
        # Scatter plot: Revenue vs Views
        st.markdown("**Revenue vs Views:**")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(df_sample['views'], df_sample['ad_revenue_usd'], 
                  alpha=0.4,
                  s=30,
                  color='steelblue', 
                  edgecolors='white',
                  linewidth=0.5)
        ax.set_xlabel('Views', fontsize=12)
        ax.set_ylabel('Revenue ($)', fontsize=12)
        ax.set_title('Revenue vs Views', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df_sample['views'], df_sample['ad_revenue_usd'], 1)
        p = np.poly1d(z)
        ax.plot(df_sample['views'].sort_values(), 
               p(df_sample['views'].sort_values()), 
               "r--", linewidth=2, label='Trend line')
        ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show correlation
        corr_views = df['views'].corr(df['ad_revenue_usd'])
        st.info(f"📊 Correlation: {corr_views:.3f}")
    
    with scatter_col2:
        # Scatter plot: Revenue vs Likes
        st.markdown("**Revenue vs Likes:**")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(df_sample['likes'], df_sample['ad_revenue_usd'], 
                  alpha=0.4,
                  s=30,
                  color='coral', 
                  edgecolors='white',
                  linewidth=0.5)
        ax.set_xlabel('Likes', fontsize=12)
        ax.set_ylabel('Revenue ($)', fontsize=12)
        ax.set_title('Revenue vs Likes', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df_sample['likes'], df_sample['ad_revenue_usd'], 1)
        p = np.poly1d(z)
        ax.plot(df_sample['likes'].sort_values(), 
               p(df_sample['likes'].sort_values()), 
               "r--", linewidth=2, label='Trend line')
        ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show correlation
        corr_likes = df['likes'].corr(df['ad_revenue_usd'])
        st.info(f"📊 Correlation: {corr_likes:.3f}")
    
    st.markdown("---")
    
    # ============================================================
    # DISTRIBUTION ANALYSIS (UNIVARIATE)
    # ============================================================
    st.subheader("📊 Distribution Analysis")
    st.markdown("**Understanding the distribution of key metrics:**")
    
    dist_col1, dist_col2 = st.columns(2)
    
    with dist_col1:
        # Views distribution
        st.markdown("**Views Distribution:**")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(df['views'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(df['views'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["views"].mean():,.0f}')
        ax.axvline(df['views'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["views"].median():,.0f}')
        ax.set_xlabel('Views', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Views', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Stats
        st.info(f"""
        **Views Statistics:**
        - Mean: {df['views'].mean():,.0f}
        - Median: {df['views'].median():,.0f}
        - Std Dev: {df['views'].std():,.0f}
        """)
    
    with dist_col2:
        # Likes distribution
        st.markdown("**Likes Distribution:**")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(df['likes'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
        ax.axvline(df['likes'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["likes"].mean():,.0f}')
        ax.axvline(df['likes'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["likes"].median():,.0f}')
        ax.set_xlabel('Likes', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Likes', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Stats
        st.info(f"""
        **Likes Statistics:**
        - Mean: {df['likes'].mean():,.0f}
        - Median: {df['likes'].median():,.0f}
        - Std Dev: {df['likes'].std():,.0f}
        """)
    
    # Comments distribution (full width)
    st.markdown("**Comments Distribution:**")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(df['comments'], bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    ax.axvline(df['comments'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["comments"].mean():,.0f}')
    ax.axvline(df['comments'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["comments"].median():,.0f}')
    ax.set_xlabel('Comments', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Comments', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    st.pyplot(fig)
    
    comment_col1, comment_col2, comment_col3 = st.columns(3)
    comment_col1.metric("Mean Comments", f"{df['comments'].mean():,.0f}")
    comment_col2.metric("Median Comments", f"{df['comments'].median():,.0f}")
    comment_col3.metric("Std Dev", f"{df['comments'].std():,.0f}")
    
    st.markdown("---")
    
    # ============================================================
    # TOP & BOTTOM PERFORMERS
    # ============================================================
    st.subheader("🏆 Top & Bottom Performing Videos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🥇 Top 10 Highest Revenue Videos:**")
        top_10 = df.nlargest(10, 'ad_revenue_usd')[['views', 'likes', 'comments', 'subscribers', 'ad_revenue_usd']].copy()
        
        top_10.reset_index(drop=True, inplace=True)
        top_10.index = top_10.index + 1
        
        st.dataframe(
            top_10.style.format({
                'views': '{:,.0f}',
                'likes': '{:,.0f}',
                'comments': '{:,.0f}',
                'subscribers': '{:,.0f}',
                'ad_revenue_usd': '${:.2f}'
            }).background_gradient(subset=['ad_revenue_usd'], cmap='Greens'),
            use_container_width=True
        )
        
        st.info(f"""
        **Top 10 Insights:**
        - Average Revenue: ${top_10['ad_revenue_usd'].mean():.2f}
        - Average Views: {top_10['views'].mean():,.0f}
        - Average Engagement: {((top_10['likes'].mean() + top_10['comments'].mean()) / top_10['views'].mean() * 100):.2f}%
        """)
        
        csv_top10 = convert_df_to_csv(top_10)
        st.download_button(
            label="📥 Download Top 10",
            data=csv_top10,
            file_name=f"top_10_performers_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="download_top10"
        )
    
    with col2:
        st.markdown("**📉 Bottom 10 Lowest Revenue Videos:**")
        bottom_10 = df.nsmallest(10, 'ad_revenue_usd')[['views', 'likes', 'comments', 'subscribers', 'ad_revenue_usd']].copy()
        
        bottom_10.reset_index(drop=True, inplace=True)
        bottom_10.index = bottom_10.index + 1
        
        st.dataframe(
            bottom_10.style.format({
                'views': '{:,.0f}',
                'likes': '{:,.0f}',
                'comments': '{:,.0f}',
                'subscribers': '{:,.0f}',
                'ad_revenue_usd': '${:.2f}'
            }).background_gradient(subset=['ad_revenue_usd'], cmap='Reds_r'),
            use_container_width=True
        )
        
        st.info(f"""
        **Bottom 10 Insights:**
        - Average Revenue: ${bottom_10['ad_revenue_usd'].mean():.2f}
        - Average Views: {bottom_10['views'].mean():,.0f}
        - Average Engagement: {((bottom_10['likes'].mean() + bottom_10['comments'].mean()) / bottom_10['views'].mean() * 100):.2f}%
        """)
        
        csv_bottom10 = convert_df_to_csv(bottom_10)
        st.download_button(
            label="📥 Download Bottom 10",
            data=csv_bottom10,
            file_name=f"bottom_10_performers_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="download_bottom10"
        )
    
    revenue_gap = top_10['ad_revenue_usd'].mean() - bottom_10['ad_revenue_usd'].mean()
    st.success(f"""
    💡 **Key Insight:** Top performers earn **${revenue_gap:.2f}** more on average than bottom performers.
    This represents a **{(revenue_gap / bottom_10['ad_revenue_usd'].mean() * 100):.1f}%** difference!
    """)
    
    st.markdown("---")
    
    # ============================================================
    # PERCENTILE ANALYSIS
    # ============================================================
    st.subheader("📊 Revenue Percentile Analysis")
    
    st.markdown("""
    **What are percentiles?**  
    Percentiles show how your video compares to others. For example:
    - **50th percentile** = Median (half earn more, half earn less)
    - **90th percentile** = Top 10% (only 10% earn more than this)
    - **10th percentile** = Bottom 10% (90% earn more than this)
    """)
    
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    percentile_values = [np.percentile(df['ad_revenue_usd'], p) for p in percentiles]
    
    percentile_df = pd.DataFrame({
        'Percentile': [f"{p}th" for p in percentiles],
        'Revenue': percentile_values,
        'Interpretation': [
            "Bottom 10% earn less than this",
            "Bottom 25% earn less than this (Q1)",
            "Median - 50% earn less than this",
            "Top 25% earn more than this (Q3)",
            "Top 10% earn more than this",
            "Top 5% earn more than this",
            "Top 1% earn more than this (Elite)"
        ]
    })
    
    st.dataframe(
        percentile_df.style.format({'Revenue': '${:.2f}'}).background_gradient(
            subset=['Revenue'], cmap='RdYlGn'
        ),
        use_container_width=True,
        hide_index=True
    )
    
    csv_percentiles = convert_df_to_csv(percentile_df)
    st.download_button(
        label="📥 Download Percentile Analysis",
        data=csv_percentiles,
        file_name=f"percentile_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key="download_percentiles"
    )
    
    st.markdown("**Revenue by Percentile (Visual):**")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850', '#006837']
    bars = ax.barh(percentile_df['Percentile'], percentile_df['Revenue'], color=colors, edgecolor='black')
    
    ax.set_xlabel('Revenue ($)', fontsize=12)
    ax.set_ylabel('Percentile', fontsize=12)
    ax.set_title('Revenue Distribution Across Percentiles', fontsize=14, fontweight='bold')
    
    for i, (bar, value) in enumerate(zip(bars, percentile_values)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'  ${value:.2f}', 
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.success(f"""
    **📈 Percentile Insights:**
    
    - **Median Revenue:** ${percentile_values[2]:.2f} (half of videos earn more, half earn less)
    - **To be in top 25%:** Need to earn more than ${percentile_values[3]:.2f}
    - **To be in top 10%:** Need to earn more than ${percentile_values[4]:.2f}
    - **Elite 1%:** Earning more than ${percentile_values[6]:.2f}
    
    **Gap Analysis:**
    - Difference between top 10% and median: ${(percentile_values[4] - percentile_values[2]):.2f}
    - Difference between top 1% and median: ${(percentile_values[6] - percentile_values[2]):.2f}
    """)
    
    # INTERACTIVE PERCENTILE CALCULATOR
    st.markdown("---")
    st.markdown("**🔍 Find Your Percentile:**")
    
    user_revenue = st.number_input(
        "Enter a revenue amount to see which percentile it falls into:",
        min_value=0.0,
        value=250.0,
        step=10.0
    )
    
    percentile_rank = (df['ad_revenue_usd'] <= user_revenue).sum() / len(df) * 100
    
    if percentile_rank > 100:
        percentile_rank = 100
    if percentile_rank < 0:
        percentile_rank = 0
    
    st.metric(
        "Your Percentile Rank",
        f"{percentile_rank:.1f}th percentile",
        delta=f"Better than {percentile_rank:.1f}% of videos"
    )
    
    if percentile_rank >= 99:
        st.success("🏆 **Elite!** You're in the top 1%!")
    elif percentile_rank >= 90:
        st.success("⭐ **Excellent!** You're in the top 10%!")
    elif percentile_rank >= 75:
        st.info("✅ **Good!** You're in the top 25%!")
    elif percentile_rank >= 50:
        st.info("📊 **Average** - Above the median!")
    else:
        st.warning("📉 Below median - Room for improvement!")
    
    st.markdown("---")
    
    # DOWNLOAD COMPLETE ANALYTICS SUMMARY
    st.subheader("📥 Download Complete Analytics Report")
    
    st.markdown("**Generate a comprehensive summary report with all key statistics:**")
    
    summary_report = pd.DataFrame({
        'Metric': [
            'Report Generated',
            'Total Videos Analyzed',
            'Average Revenue',
            'Median Revenue',
            'Revenue Std Dev',
            'Min Revenue',
            'Max Revenue',
            'Total Revenue',
            'Average Views',
            'Average Likes',
            'Average Comments',
            'Average Subscribers',
            'Top 10 Avg Revenue',
            'Bottom 10 Avg Revenue',
            'Revenue Gap (Top vs Bottom)',
            '10th Percentile',
            '25th Percentile (Q1)',
            '50th Percentile (Median)',
            '75th Percentile (Q3)',
            '90th Percentile',
            '95th Percentile',
            '99th Percentile'
        ],
        'Value': [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            f"{len(df):,}",
            f"${df['ad_revenue_usd'].mean():.2f}",
            f"${df['ad_revenue_usd'].median():.2f}",
            f"${df['ad_revenue_usd'].std():.2f}",
            f"${df['ad_revenue_usd'].min():.2f}",
            f"${df['ad_revenue_usd'].max():.2f}",
            f"${df['ad_revenue_usd'].sum():.2f}",
            f"{df['views'].mean():,.0f}",
            f"{df['likes'].mean():,.0f}",
            f"{df['comments'].mean():,.0f}",
            f"{df['subscribers'].mean():,.0f}",
            f"${top_10['ad_revenue_usd'].mean():.2f}",
            f"${bottom_10['ad_revenue_usd'].mean():.2f}",
            f"${revenue_gap:.2f}",
            f"${percentile_values[0]:.2f}",
            f"${percentile_values[1]:.2f}",
            f"${percentile_values[2]:.2f}",
            f"${percentile_values[3]:.2f}",
            f"${percentile_values[4]:.2f}",
            f"${percentile_values[5]:.2f}",
            f"${percentile_values[6]:.2f}"
        ]
    })
    
    with st.expander("📋 Preview Report"):
        st.dataframe(summary_report, use_container_width=True, hide_index=True)
    
    csv_summary = convert_df_to_csv(summary_report)
    st.download_button(
        label="📥 Download Complete Analytics Summary",
        data=csv_summary,
        file_name=f"analytics_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key="download_summary",
        help="Download a comprehensive report with all analytics statistics"
    )
    
    st.markdown("---")
    
    # REVENUE DISTRIBUTION
    st.subheader("💰 Revenue Distribution")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['ad_revenue_usd'], bins=30, color='skyblue', edgecolor='black')
        ax.set_xlabel('Revenue ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('How Revenue is Distributed')
        st.pyplot(fig)
    
    with chart_col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(df['ad_revenue_usd'])
        ax.set_ylabel('Revenue ($)')
        ax.set_title('Revenue Range (Box Plot)')
        st.pyplot(fig)
    
    st.markdown("---")
    
    # REVENUE BY CATEGORY (UPDATED FOR ONE-HOT ENCODING)
    st.subheader("📂 Revenue by Category")
    
    # Check if category columns exist (one-hot encoded)
    category_cols = [col for col in df.columns if col.startswith('category_')]
    
    if category_cols:
        # One-hot encoded - reconstruct category column
        df_temp = df.copy()
        
        # Find which category column is 1 for each row
        def get_category(row):
            for col in category_cols:
                if row[col] == 1:
                    return col.replace('category_', '')
            return 'Unknown'
        
        df_temp['category_name'] = df_temp.apply(get_category, axis=1)
        category_avg = df_temp.groupby('category_name')['ad_revenue_usd'].mean().sort_values(ascending=False)
        
        category_report = pd.DataFrame({
            'Category': category_avg.index,
            'Average Revenue': category_avg.values
        })
        
        fig, ax = plt.subplots(figsize=(10, 5))
        category_avg.plot(kind='bar', ax=ax, color='coral', edgecolor='black')
        ax.set_ylabel('Average Revenue ($)')
        ax.set_xlabel('Category')
        ax.set_title('Which Categories Earn Most?')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        csv_category = convert_df_to_csv(category_report)
        st.download_button(
            label="📥 Download Category Analysis",
            data=csv_category,
            file_name=f"category_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="download_category"
        )
    else:
        st.info("ℹ️ Category data is one-hot encoded in the cleaned dataset")
    
    st.markdown("---")
    
    # MODEL PERFORMANCE COMPARISON
    st.subheader("🏆 Model Performance Comparison")
    
    model_perf = []
    for model_name, metrics in results.items():
        model_perf.append({
            'Model': model_name,
            'R² Score': f"{metrics['R2']:.4f}",
            'RMSE': f"${metrics['RMSE']:.2f}",
            'MAE': f"${metrics['MAE']:.2f}"
        })
    
    perf_df = pd.DataFrame(model_perf)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)
    
    best_model = max(results, key=lambda x: results[x]['R2'])
    st.success(f"🏆 **Best Model:** {best_model} (R² = {results[best_model]['R2']:.4f})")
    
    csv_models = convert_df_to_csv(perf_df)
    st.download_button(
        label="📥 Download Model Performance Report",
        data=csv_models,
        file_name=f"model_performance_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key="download_models"
    )
    
    st.markdown("---")
    
    # SAMPLE DATA
    st.subheader("📋 Sample Data (First 10 Rows)")
    st.dataframe(df.head(10), use_container_width=True)
    
    csv_full = convert_df_to_csv(df)
    st.download_button(
        label="📥 Download Full Dataset",
        data=csv_full,
        file_name=f"full_dataset_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key="download_full",
        help="Download the complete cleaned dataset"
    )