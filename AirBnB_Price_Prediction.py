# Airbnb Price Prediction Project
# By Uche Onyejiaka

# 1. Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 2. Load the dataset
file_path = 'listings.csv'  # Assuming the file is in the same directory
listings = pd.read_csv(file_path)

# 3. Basic Data Cleaning
# Drop rows missing important fields
listings.dropna(subset=['price', 'room_type', 'neighbourhood'], inplace=True)

# Fill missing 'reviews_per_month' with 0 since no reviews likely means 0 monthly reviews
listings['reviews_per_month'] = listings['reviews_per_month'].fillna(0)

# Remove listings with extremely high prices (outliers)
listings = listings[listings['price'] <= 1000]

# 4. Selecting Features for Modeling
features = [
    'neighbourhood',
    'latitude',
    'longitude',
    'room_type',
    'minimum_nights',
    'number_of_reviews',
    'reviews_per_month',
    'calculated_host_listings_count',
    'availability_365'
]

X = listings[features]
y = listings['price']

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=['neighbourhood', 'room_type'], drop_first=True)

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Training the Models
# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Random Forest Regressor Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# 7. Model Evaluation Function
def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"--- {model_name} Results ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared Score: {r2:.2f}")
    print()

# Evaluating both models
evaluate_model(y_test, y_pred_linear, "Linear Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest Regressor")

# 8. Feature Importance for Random Forest
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
important_features = feature_importances.sort_values(ascending=False)

# Visualize top 20 important features
plt.figure(figsize=(12, 7))
sns.barplot(x=important_features.values[:20], y=important_features.index[:20])
plt.title('Top 20 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# 9. Analyzing Model Errors
results = X_test.copy()
results['Actual Price'] = y_test
results['Predicted Price'] = y_pred_rf
results['Prediction Error'] = results['Actual Price'] - results['Predicted Price']
results['Absolute Error'] = results['Prediction Error'].abs()

# Displaying top 5 biggest errors
top_errors = results.sort_values('Absolute Error', ascending=False).head(5)
print("Top 5 Listings with Largest Prediction Errors:")
print(top_errors[['Actual Price', 'Predicted Price', 'Prediction Error', 'Absolute Error']])

# Project Complete
