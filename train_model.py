import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('car_insurance_premium_regression_dataset (1) (1).csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())
print("\nStatistical summary:")
print(df.describe())

# Data Preprocessing
# Handle missing values
df['engine_cc'].fillna(df['engine_cc'].median(), inplace=True)
df['car_value'].fillna(df['car_value'].median(), inplace=True)
df['fuel_type'].fillna(df['fuel_type'].mode()[0], inplace=True)
df['transmission'].fillna(df['transmission'].mode()[0], inplace=True)
df['owner_age'].fillna(df['owner_age'].median(), inplace=True)
df['ncb_percent'].fillna(df['ncb_percent'].median(), inplace=True)
df['city_tier'].fillna(df['city_tier'].mode()[0], inplace=True)

print("\nAfter handling missing values:")
print(df.isnull().sum())

# Encode categorical variables
le_fuel = LabelEncoder()
le_transmission = LabelEncoder()
le_accident = LabelEncoder()
le_city = LabelEncoder()

df['fuel_type'] = le_fuel.fit_transform(df['fuel_type'])
df['transmission'] = le_transmission.fit_transform(df['transmission'])
df['accident_history'] = le_accident.fit_transform(df['accident_history'])
df['city_tier'] = le_city.fit_transform(df['city_tier'])

print("\nAfter encoding categorical variables:")
print(df.head())

# Prepare features and target
X = df.drop('annual_car_premium', axis=1)
y = df['annual_car_premium']

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
print("\n" + "="*60)
print("TRAINING MODELS")
print("="*60)

# Linear Regression
print("\n1. Linear Regression")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
lr_r2 = r2_score(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
lr_mae = mean_absolute_error(y_test, y_pred_lr)
print(f"   R² Score: {lr_r2:.4f}")
print(f"   RMSE: {lr_rmse:.2f}")
print(f"   MAE: {lr_mae:.2f}")

# Random Forest
print("\n2. Random Forest Regressor")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_mae = mean_absolute_error(y_test, y_pred_rf)
print(f"   R² Score: {rf_r2:.4f}")
print(f"   RMSE: {rf_rmse:.2f}")
print(f"   MAE: {rf_mae:.2f}")

# Gradient Boosting
print("\n3. Gradient Boosting Regressor")
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
gb_r2 = r2_score(y_test, y_pred_gb)
gb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_gb))
gb_mae = mean_absolute_error(y_test, y_pred_gb)
print(f"   R² Score: {gb_r2:.4f}")
print(f"   RMSE: {gb_rmse:.2f}")
print(f"   MAE: {gb_mae:.2f}")

# Select best model
models = {
    'Linear Regression': (lr_model, lr_r2, 'scaled'),
    'Random Forest': (rf_model, rf_r2, 'unscaled'),
    'Gradient Boosting': (gb_model, gb_r2, 'unscaled')
}

best_model_name = max(models, key=lambda x: models[x][1])
best_model, best_r2, scaling_type = models[best_model_name]

print("\n" + "="*60)
print(f"BEST MODEL: {best_model_name} with R² = {best_r2:.4f}")
print("="*60)

# Save models and encoders
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump({
        'fuel_type': le_fuel,
        'transmission': le_transmission,
        'accident_history': le_accident,
        'city_tier': le_city
    }, f)

# Save feature names
feature_names = X.columns.tolist()
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

print("\nModels saved successfully!")
print("Files saved: best_model.pkl, scaler.pkl, label_encoders.pkl, feature_names.pkl")
