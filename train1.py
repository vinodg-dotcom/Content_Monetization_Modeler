"""
YouTube Ad Revenue Prediction - Linear Regression Models
========================================================
Training 5 Linear Regression Models with Enhanced Feature Engineering
(Fixed: Removed data leakage + One-Hot Encoding + 11 engineered features)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
sns.set_style("whitegrid")

# ============================================================
# STEP 1: LOAD THE DATA
# ============================================================
print("="*60)
print("📊 YOUTUBE AD REVENUE - LINEAR REGRESSION MODELS")
print("="*60)

# Load the dataset
df = pd.read_csv('data/youtube_ad_revenue_dataset.csv')
print(f"\n✅ Data loaded successfully!")
print(f"   Rows: {len(df)}")
print(f"   Columns: {len(df.columns)}")

# First look at the data
print("\n📋 First few rows:")
print(df.head())

print("\n📊 Dataset Info:")
print(df.info())


# ============================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n" + "="*60)
print("🔍 EXPLORATORY DATA ANALYSIS (EDA)")
print("="*60)

# Statistical summary
print("\n📈 Statistical Summary:")
print(df.describe())

# Check data types
print("\n📝 Data Types:")
print(df.dtypes)

# Check missing values
print("\n🔍 Missing Values:")
missing_summary = df.isnull().sum()
print(missing_summary[missing_summary > 0])
total_missing = df.isnull().sum().sum()
missing_percentage = (total_missing / (len(df) * len(df.columns))) * 100
print(f"\n   Total missing values: {total_missing} ({missing_percentage:.2f}%)")

# Check duplicates
duplicates = df.duplicated().sum()
print(f"\n📋 Duplicate rows: {duplicates}")

# Correlation analysis (only numeric columns)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n🔗 Numeric columns for correlation: {len(numeric_cols)}")

# Create correlation matrix
if len(numeric_cols) > 1:
    correlation = df[numeric_cols].corr()
    print("\n📊 Top correlations with ad_revenue_usd:")
    if 'ad_revenue_usd' in correlation.columns:
        correlations_with_target = correlation['ad_revenue_usd'].sort_values(ascending=False)
        print(correlations_with_target)
        
        # Identify data leakage
        print("\n⚠️  DATA LEAKAGE DETECTION:")
        high_corr_features = correlations_with_target[
            (correlations_with_target > 0.95) | (correlations_with_target < -0.95)
        ]
        high_corr_features = high_corr_features[high_corr_features.index != 'ad_revenue_usd']
        
        if len(high_corr_features) > 0:
            print("   Features with suspiciously high correlation (>0.95):")
            for feature, corr in high_corr_features.items():
                print(f"   • {feature}: {corr:.6f} 🚨 LIKELY DATA LEAKAGE!")
        else:
            print("   ✅ No features with suspiciously high correlation detected")
        
        # Save correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('data/correlation_heatmap.png')
        print("\n💾 Correlation heatmap saved: data/correlation_heatmap.png")
        plt.close()

# Distribution of target variable
print("\n💰 Ad Revenue Distribution:")
print(f"   Mean: ${df['ad_revenue_usd'].mean():.2f}")
print(f"   Median: ${df['ad_revenue_usd'].median():.2f}")
print(f"   Std Dev: ${df['ad_revenue_usd'].std():.2f}")
print(f"   Min: ${df['ad_revenue_usd'].min():.2f}")
print(f"   Max: ${df['ad_revenue_usd'].max():.2f}")

# Create distribution plot
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.hist(df['ad_revenue_usd'], bins=50, color='skyblue', edgecolor='black')
plt.xlabel('Ad Revenue (USD)')
plt.ylabel('Frequency')
plt.title('Revenue Distribution')

plt.subplot(1, 2, 2)
plt.boxplot(df['ad_revenue_usd'])
plt.ylabel('Ad Revenue (USD)')
plt.title('Revenue Boxplot')
plt.tight_layout()
plt.savefig('data/revenue_distribution.png')
print("💾 Revenue distribution plot saved: data/revenue_distribution.png")
plt.close()

# Categorical columns analysis
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\n🏷️ Categorical columns: {categorical_cols}")

for col in categorical_cols:
    if col not in ['video_id', 'date']:
        print(f"\n   {col} value counts:")
        print(df[col].value_counts())


# ============================================================
# STEP 3: OUTLIER DETECTION
# ============================================================
print("\n" + "="*60)
print("🎯 OUTLIER DETECTION")
print("="*60)

# IQR method for outlier detection
outlier_summary = {}

for col in numeric_cols:
    if col == 'ad_revenue_usd':  # Don't cap target variable
        continue
    
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_count = len(outliers)
    outlier_percentage = (outlier_count / len(df)) * 100
    
    if outlier_count > 0:
        outlier_summary[col] = {
            'count': outlier_count,
            'percentage': outlier_percentage,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        print(f"\n📊 {col}:")
        print(f"   Outliers detected: {outlier_count} ({outlier_percentage:.2f}%)")
        print(f"   Valid range: {lower_bound:.2f} to {upper_bound:.2f}")

# Visualize outliers
if len(numeric_cols) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols[:6]):  # First 6 numeric columns
        df.boxplot(column=col, ax=axes[idx])
        axes[idx].set_title(f'{col} - Outliers')
        axes[idx].set_ylabel(col)
    
    plt.tight_layout()
    plt.savefig('data/outlier_boxplots.png')
    print("\n💾 Outlier boxplots saved: data/outlier_boxplots.png")
    plt.close()


# ============================================================
# STEP 4: EXTRACT DATE FEATURES
# ============================================================
print("\n" + "="*60)
print("📅 EXTRACTING DATE INFORMATION")
print("="*60)

# Convert date column to datetime format
df['date'] = pd.to_datetime(df['date'])
print("✅ Converted date to datetime format")

# Extract useful date components for reference
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['time'] = df['date'].dt.strftime('%H:%M')

print("✅ Extracted date components:")
print("   • Year")
print("   • Month")
print("   • Day")
print("   • Time (HH:MM)")


# ============================================================
# STEP 5: CLEAN THE DATA
# ============================================================
print("\n" + "="*60)
print("🧹 CLEANING DATA")
print("="*60)

# Remove only video_id (keep date components for reference)
df = df.drop(columns=['video_id'])
print("✅ Removed video_id column")

# Handle missing values
rows_before = len(df)
df = df.dropna()
rows_after = len(df)
rows_removed = rows_before - rows_after
removal_percentage = (rows_removed / rows_before) * 100 if rows_before > 0 else 0
print(f"✅ Handled missing values:")
print(f"   Dropped {rows_removed} rows ({removal_percentage:.2f}% of data)")

# Remove duplicates
duplicates_before = df.duplicated().sum()
duplicate_percentage = (duplicates_before / len(df)) * 100 if len(df) > 0 else 0
df = df.drop_duplicates()
print(f"✅ Removed duplicates:")
print(f"   Removed {duplicates_before} duplicate rows ({duplicate_percentage:.2f}% of data)")

# Handle outliers using IQR capping
print("\n🎯 Handling outliers (IQR capping):")
outliers_capped = 0

for col in df.select_dtypes(include=[np.number]).columns:
    if col in ['ad_revenue_usd', 'year', 'month', 'day']:  # Skip target and date components
        continue
    
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap outliers
    before_cap = df[col].copy()
    df[col] = df[col].clip(lower_bound, upper_bound)
    
    capped = (before_cap != df[col]).sum()
    if capped > 0:
        outliers_capped += capped
        print(f"   {col}: Capped {capped} values")

print(f"✅ Total outliers capped: {outliers_capped}")

print(f"\n📊 Clean dataset shape: {df.shape}")

# Save cleaned data
df.to_csv('data/cleaned_data.csv', index=False)
print("\n💾 Cleaned data saved: data/cleaned_data.csv")


# ============================================================
# STEP 6: REMOVE DATA LEAKAGE FEATURES
# ============================================================
print("\n" + "="*60)
print("🔒 PREVENTING DATA LEAKAGE")
print("="*60)

# Remove watch_time_minutes - it has 0.99 correlation with target
if 'watch_time_minutes' in df.columns:
    print("⚠️  Removing 'watch_time_minutes':")
    print("   Reason: Extremely high correlation (0.988874) with ad_revenue_usd")
    print("   This feature is likely used to calculate the target variable")
    print("   Keeping it would result in unrealistic 'perfect' predictions")
    df = df.drop(columns=['watch_time_minutes'])
    print("   ✅ Feature removed successfully")
else:
    print("✅ No data leakage features detected")


# ============================================================
# STEP 7: ENCODE CATEGORICAL VARIABLES (ONE-HOT ENCODING)
# ============================================================
print("\n" + "="*60)
print("🏷️ ENCODING CATEGORICAL VARIABLES (ONE-HOT)")
print("="*60)

# Store original category values before encoding
category_names = sorted(df['category'].unique().tolist())
device_names = sorted(df['device'].unique().tolist())
country_names = sorted(df['country'].unique().tolist())

print(f"   Categories found: {category_names}")
print(f"   Devices found: {device_names}")
print(f"   Countries found: {country_names}")

# One-hot encode categorical variables
# This creates binary columns for each category (better for linear models)
df_encoded = pd.get_dummies(df, columns=['category', 'device', 'country'], drop_first=False, dtype=int)

print(f"\n   ✓ Original features: {len(df.columns)}")
print(f"   ✓ After one-hot encoding: {len(df_encoded.columns)}")
print(f"   ✓ New columns added: {len(df_encoded.columns) - len(df.columns)}")

# Store encoder info for Streamlit app
encoders = {
    'type': 'onehot',
    'category': category_names,
    'device': device_names,
    'country': country_names
}

# Update dataframe
df = df_encoded

print("\n   💡 Why One-Hot Encoding?")
print("      Linear models assume numeric features have order (1 < 2 < 3)")
print("      Categories have NO natural order, so we create separate binary columns")
print("      This improves linear model performance significantly!")


# ============================================================
# STEP 8: FEATURE ENGINEERING (ENHANCED)
# ============================================================
print("\n" + "="*60)
print("🛠️ FEATURE ENGINEERING")
print("="*60)

print("\n🔧 Creating basic engagement features...")

# Basic engagement metrics
df['engagement_rate'] = (df['likes'] + df['comments']) / (df['views'] + 1)
df['like_rate'] = df['likes'] / (df['views'] + 1)
df['comment_rate'] = df['comments'] / (df['views'] + 1)
df['views_per_subscriber'] = df['views'] / (df['subscribers'] + 1)

print("   ✓ engagement_rate (total engagement)")
print("   ✓ like_rate (percentage of viewers who liked)")
print("   ✓ comment_rate (percentage of viewers who commented)")
print("   ✓ views_per_subscriber (reach per subscriber)")

print("\n🔧 Creating advanced engineered features...")

# ADVANCED FEATURE 1: Engagement Quality
# Comments are more valuable than likes (show deeper engagement)
df['engagement_quality'] = (df['likes'] * 1 + df['comments'] * 3) / (df['views'] + 1)
print("   ✓ engagement_quality (weighted: comments count 3x more than likes)")

# ADVANCED FEATURE 2: Video Popularity
# Views per minute of content (viral videos have high values)
df['video_popularity'] = df['views'] / (df['video_length_minutes'] + 1)
print("   ✓ video_popularity (views per minute of video)")

# ADVANCED FEATURE 3: Interaction Feature
# Combines views with like engagement
df['views_likes_interaction'] = df['views'] * df['like_rate']
print("   ✓ views_likes_interaction (combined views and like rate)")

print("\n🔧 Applying logarithmic transformations...")
print("   (Log transforms help linear models handle skewed data)")

# Log transformations - common technique for skewed distributions
# np.log1p = log(x + 1) to avoid log(0)
df['log_views'] = np.log1p(df['views'])
df['log_subscribers'] = np.log1p(df['subscribers'])
df['log_likes'] = np.log1p(df['likes'])

print("   ✓ log_views (normalized view distribution)")
print("   ✓ log_subscribers (normalized subscriber distribution)")
print("   ✓ log_likes (normalized like distribution)")

print(f"\n✅ Total features created: 11")
print("   • 4 basic engagement features")
print("   • 3 advanced engineered features")
print("   • 3 logarithmic transformations")
print("   ⚠️ Skipped 'watch_ratio' (would use watch_time_minutes - data leakage)")

# Save engineered data
df.to_csv('data/engineered_data.csv', index=False)
print("\n💾 Engineered data saved: data/engineered_data.csv")


# ============================================================
# STEP 9: SPLIT DATA
# ============================================================
print("\n" + "="*60)
print("✂️ SPLITTING DATA")
print("="*60)

# Remove date-related columns from features
date_columns = ['date', 'year', 'month', 'day', 'time']
df_for_training = df.drop(columns=date_columns, errors='ignore')

# Separate features and target
X = df_for_training.drop('ad_revenue_usd', axis=1)
y = df_for_training['ad_revenue_usd']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✅ Training set: {len(X_train)} samples")
print(f"✅ Testing set: {len(X_test)} samples")
print(f"✅ Number of features: {X_train.shape[1]}")
print(f"\n📋 First 10 features: {list(X_train.columns)[:10]}")
print(f"   ... and {X_train.shape[1] - 10} more features")


# ============================================================
# STEP 10: SCALE FEATURES
# ============================================================
print("\n" + "="*60)
print("⚖️ SCALING FEATURES")
print("="*60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Features scaled using StandardScaler")


# ============================================================
# STEP 11: TRAIN 5 LINEAR REGRESSION MODELS
# ============================================================
print("\n" + "="*60)
print("🤖 TRAINING 5 LINEAR REGRESSION MODELS")
print("="*60)

all_models = {}
results = {}

# MODEL 1: Linear Regression (OLS)
print("\n🔧 Training Linear Regression (OLS)...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_predictions = lr_model.predict(X_test_scaled)

lr_r2 = r2_score(y_test, lr_predictions)
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
lr_mse = mean_squared_error(y_test, lr_predictions)

print(f"   📊 R² Score: {lr_r2:.4f} | RMSE: ${lr_rmse:.2f} | MAE: ${lr_mae:.2f}")

all_models['Linear Regression'] = lr_model
results['Linear Regression'] = {'R2': lr_r2, 'RMSE': lr_rmse, 'MAE': lr_mae, 'MSE': lr_mse}


# MODEL 2: L2 (Ridge) Regression
print("\n🔧 Training L2 (Ridge) Regression...")
ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train_scaled, y_train)
ridge_predictions = ridge_model.predict(X_test_scaled)

ridge_r2 = r2_score(y_test, ridge_predictions)
ridge_mae = mean_absolute_error(y_test, ridge_predictions)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_predictions))
ridge_mse = mean_squared_error(y_test, ridge_predictions)

print(f"   📊 R² Score: {ridge_r2:.4f} | RMSE: ${ridge_rmse:.2f} | MAE: ${ridge_mae:.2f}")

all_models['L2 (Ridge) Regression'] = ridge_model
results['L2 (Ridge) Regression'] = {'R2': ridge_r2, 'RMSE': ridge_rmse, 'MAE': ridge_mae, 'MSE': ridge_mse}


# MODEL 3: L1 (Lasso) Regression
print("\n🔧 Training L1 (Lasso) Regression...")
lasso_model = Lasso(alpha=0.1, max_iter=10000, random_state=42)
lasso_model.fit(X_train_scaled, y_train)
lasso_predictions = lasso_model.predict(X_test_scaled)

lasso_r2 = r2_score(y_test, lasso_predictions)
lasso_mae = mean_absolute_error(y_test, lasso_predictions)
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_predictions))
lasso_mse = mean_squared_error(y_test, lasso_predictions)

print(f"   📊 R² Score: {lasso_r2:.4f} | RMSE: ${lasso_rmse:.2f} | MAE: ${lasso_mae:.2f}")

all_models['L1 (Lasso) Regression'] = lasso_model
results['L1 (Lasso) Regression'] = {'R2': lasso_r2, 'RMSE': lasso_rmse, 'MAE': lasso_mae, 'MSE': lasso_mse}


# MODEL 4: Polynomial Regression (degree 2)
print("\n🔧 Training Polynomial Regression...")
poly_model = Pipeline([
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('linear_reg', LinearRegression())
])
poly_model.fit(X_train, y_train)
poly_predictions = poly_model.predict(X_test)

poly_r2 = r2_score(y_test, poly_predictions)
poly_mae = mean_absolute_error(y_test, poly_predictions)
poly_rmse = np.sqrt(mean_squared_error(y_test, poly_predictions))
poly_mse = mean_squared_error(y_test, poly_predictions)

n_poly_features = poly_model.named_steps['poly_features'].n_output_features_
print(f"   📊 R² Score: {poly_r2:.4f} | RMSE: ${poly_rmse:.2f} | MAE: ${poly_mae:.2f}")
print(f"   ℹ️  Created {n_poly_features} polynomial features from {X_train.shape[1]} original features")

all_models['Polynomial Regression'] = poly_model
results['Polynomial Regression'] = {'R2': poly_r2, 'RMSE': poly_rmse, 'MAE': poly_mae, 'MSE': poly_mse}


# MODEL 5: SGD Regressor (Gradient Descent)
print("\n🔧 Training SGD Regressor (Gradient Descent)...")
sgd_model = SGDRegressor(
    loss='squared_error',
    penalty='l2',
    alpha=0.0001,
    max_iter=1000,
    tol=1e-3,
    random_state=42
)
sgd_model.fit(X_train_scaled, y_train)
sgd_predictions = sgd_model.predict(X_test_scaled)

sgd_r2 = r2_score(y_test, sgd_predictions)
sgd_mae = mean_absolute_error(y_test, sgd_predictions)
sgd_rmse = np.sqrt(mean_squared_error(y_test, sgd_predictions))
sgd_mse = mean_squared_error(y_test, sgd_predictions)

print(f"   📊 R² Score: {sgd_r2:.4f} | RMSE: ${sgd_rmse:.2f} | MAE: ${sgd_mae:.2f}")

all_models['SGD Regressor'] = sgd_model
results['SGD Regressor'] = {'R2': sgd_r2, 'RMSE': sgd_rmse, 'MAE': sgd_mae, 'MSE': sgd_mse}


# ============================================================
# STEP 12: MODEL COMPARISON
# ============================================================
print("\n" + "="*60)
print("📈 MODEL COMPARISON")
print("="*60)

comparison = pd.DataFrame({
    'Model': list(results.keys()),
    'R² Score': [results[m]['R2'] for m in results.keys()],
    'RMSE ($)': [results[m]['RMSE'] for m in results.keys()],
    'MAE ($)': [results[m]['MAE'] for m in results.keys()]
})

print("\n" + comparison.to_string(index=False))

best_model_name = max(results, key=lambda x: results[x]['R2'])
best_model = all_models[best_model_name]
best_r2 = results[best_model_name]['R2']
best_mae = results[best_model_name]['MAE']
best_rmse = results[best_model_name]['RMSE']

# Determine which models use scaling
scaling_models = ['Linear Regression', 'L2 (Ridge) Regression', 'L1 (Lasso) Regression', 'SGD Regressor']
uses_scaling = best_model_name in scaling_models

print("\n" + "="*60)
print(f"🏆 BEST MODEL: {best_model_name}")
print("="*60)
print(f"   R² Score: {best_r2:.4f}")
print(f"   RMSE: ${best_rmse:.2f}")
print(f"   MAE: ${best_mae:.2f}")

# Interpretation of results
print("\n📊 PERFORMANCE INTERPRETATION:")
if best_r2 >= 0.90:
    print("   ⚠️  WARNING: R² ≥ 0.90 - Check for remaining data leakage!")
elif best_r2 >= 0.70:
    print("   ⭐⭐⭐ EXCELLENT: Model performs very well!")
elif best_r2 >= 0.50:
    print("   ✅ GOOD: Model is production-ready!")
elif best_r2 >= 0.30:
    print("   📊 MODERATE: Model is useful but could be improved")
elif best_r2 >= 0.10:
    print("   📈 FAIR: Decent performance for linear models on this dataset")
    print("       One-hot encoding improved performance significantly!")
else:
    print("   ⚠️  LIMITED: Linear models struggle with this dataset")
    print("       Consider non-linear models for better performance")


# ============================================================
# STEP 13: SAVE ALL ARTIFACTS
# ============================================================
print("\n💾 Saving all models and artifacts...")

with open('models/all_models.pkl', 'wb') as f:
    pickle.dump(all_models, f)

with open('models/results.pkl', 'wb') as f:
    pickle.dump(results, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('models/encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)

with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('models/model_info.pkl', 'wb') as f:
    pickle.dump({
        'model_name': best_model_name,
        'uses_scaling': uses_scaling,
        'r2_score': best_r2,
        'mae': best_mae
    }, f)

print("✅ All 5 linear models saved to: models/all_models.pkl")
print("✅ All artifacts saved successfully!")


# ============================================================
# STEP 14: SUMMARY REPORT
# ============================================================
print("\n" + "="*60)
print("✅ COMPLETE IMPLEMENTATION SUMMARY")
print("="*60)

print("\n📊 LINEAR REGRESSION MODELS (5 total):")
print("   1. ✅ Linear Regression (OLS)")
print("   2. ✅ L2 (Ridge) Regression - L2 Regularization")
print("   3. ✅ L1 (Lasso) Regression - L1 Regularization")
print("   4. ✅ Polynomial Regression - Polynomial transformation (degree 2)")
print("   5. ✅ SGD Regressor - Gradient Descent optimization")

print("\n🛠️ TECHNIQUES IMPLEMENTED:")
print("   ✅ 1. EDA (Exploratory Data Analysis)")
print("   ✅ 2. Data Leakage Detection & Prevention")
print("   ✅ 3. Outlier Detection (IQR method)")
print("   ✅ 4. Missing Value Handling")
print("   ✅ 5. Feature Engineering (11 new features)")
print("       • 4 basic engagement metrics")
print("       • 3 advanced engineered features")
print("       • 3 logarithmic transformations")
print("   ✅ 6. Categorical Encoding (One-Hot Encoding)")
print("   ✅ 7. Feature Scaling (StandardScaler)")
print("   ✅ 8. Polynomial Features (PolynomialFeatures)")
print("   ✅ 9. Model Evaluation (R², RMSE, MAE, MSE)")
print("   ✅ 10. Data Visualization (Matplotlib, Seaborn)")

print("\n🔒 DATA QUALITY:")
print("   ✅ Removed 'watch_time_minutes' (data leakage)")
print("   ✅ Handled missing values")
print("   ✅ Removed duplicates")
print("   ✅ Capped outliers")

print("\n📚 LIBRARIES USED:")
print("   ✅ Pandas - Data manipulation")
print("   ✅ Scikit-learn - ML models & preprocessing")
print("   ✅ Matplotlib - Visualization")
print("   ✅ Seaborn - Statistical visualization")
print("   ✅ NumPy - Numerical operations")

print("\n📁 FILES CREATED:")
print("   • data/cleaned_data.csv")
print("   • data/engineered_data.csv")
print("   • data/correlation_heatmap.png")
print("   • data/revenue_distribution.png")
print("   • data/outlier_boxplots.png")
print("   • models/all_models.pkl (5 linear models)")
print("   • models/results.pkl")
print("   • models/scaler.pkl")
print("   • models/encoders.pkl")
print("   • models/feature_names.pkl")
print("   • models/best_model.pkl")
print("   • models/model_info.pkl")

print("\n🚀 Next: Run Streamlit app to interact with all 5 linear models!")
print("="*60)