import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Path to your dataset
data = r'C:\Users\vamsi\Documents\python problems\synthetic_thunderstorm_data.csv'

# Step 1: Read the CSV file into a DataFrame
synthetic_data = pd.read_csv(data)

# Step 2: Separate features and target variable
y = synthetic_data['Thunderstorm_Probability']  # Target variable
X = synthetic_data.drop(columns=['Thunderstorm_Probability'])  # Features

# Step 3: Convert categorical variables to dummy/indicator variables
X = pd.get_dummies(X, columns=['Front_Type', 'Time_of_Day', 'Season', 'Land_Use_Type', 'Urban_Density', 'Soil_Type'])

# Step 4: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Initialize the XGBoost model with parameters
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.01, max_depth=6, random_state=42)

# Step 6: Train the model on the training data
model.fit(X_train, y_train)

# Step 7: Evaluate the model on the test data
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Step 8: Define the feature names used in the model
feature_names = X_train.columns.tolist()

# Step 9: Create new data for prediction
new_data1 = pd.DataFrame({
    'Latitude': [37.6831],
    'Longitude': [-117.2031],
    'Temperature': [30.5],
    'Humidity': [78.3],
    'Pressure': [1000.1],
    'Wind_Speed': [1.3],
    'Precipitation': [45.66],
    'Stability_Index': [21.3],
    'Front_Type_Cold Front': [0],
    'Front_Type_Warm Front': [0],
    'Front_Type_No Front': [1],
    'Time_of_Day_Morning': [0],
    'Time_of_Day_Afternoon': [1],
    'Time_of_Day_Evening': [0],
    'Season_Spring': [0],
    'Season_Summer': [1],
    'Season_Fall': [0],
    'Season_Winter': [0],
    'Cloud_Cover': [65.5],
    'Solar_Radiation': [558.3],
    'Land_Use_Type_Urban': [0],
    'Land_Use_Type_Suburban': [0],
    'Land_Use_Type_Rural': [1],
    'Elevation': [164.7],
    'Urban_Density_High': [0],
    'Urban_Density_Medium': [1],
    'Urban_Density_Low': [0],
    'Soil_Type_Loam': [1],
    'Soil_Type_Sandy': [0],
    'Soil_Type_Clay': [0]
})

# Step 10: Make predictions on the new data
predicted_probability = model.predict(new_data1[feature_names])
predicted_percentage = predicted_probability[0] * 100
print(f"Predicted Thunderstorm Probability for new_data1: {predicted_percentage:.2f}%")

# Repeat the same steps for new_data2 and new_data3
new_data2 = pd.DataFrame({
    'Latitude': [38.5971],
    'Longitude': [-107.3359],
    'Temperature': [34.4],
    'Humidity': [73.9],
    'Pressure': [1010.2],
    'Wind_Speed': [0.9],
    'Precipitation': [37.78],
    'Stability_Index': [11.2],
    'Front_Type_Cold Front': [0],
    'Front_Type_Warm Front': [0],
    'Front_Type_No Front': [1],
    'Time_of_Day_Morning': [1],
    'Time_of_Day_Afternoon': [0],
    'Time_of_Day_Evening': [0],
    'Season_Spring': [0],
    'Season_Summer': [1],
    'Season_Fall': [0],
    'Season_Winter': [0],
    'Cloud_Cover': [84.2],
    'Solar_Radiation': [697.3],
    'Land_Use_Type_Urban': [0],
    'Land_Use_Type_Suburban': [1],
    'Land_Use_Type_Rural': [0],
    'Elevation': [57.8],
    'Urban_Density_High': [1],
    'Urban_Density_Medium': [0],
    'Urban_Density_Low': [0],
    'Soil_Type_Loam': [1],
    'Soil_Type_Sandy': [0],
    'Soil_Type_Clay': [0]
})

predicted_probability2 = model.predict(new_data2[feature_names])
predicted_percentage2 = predicted_probability2[0] * 100
print(f"Predicted Thunderstorm Probability for new_data2: {predicted_percentage2:.2f}%")

new_data3 = pd.DataFrame({
    'Latitude': [33.3612],
    'Longitude': [-80.0781],
    'Temperature': [25.1],
    'Humidity': [74.9],
    'Pressure': [997.7],
    'Wind_Speed': [12.5],
    'Precipitation': [42.14],
    'Stability_Index': [24.7],
    'Front_Type_Cold Front': [1],
    'Front_Type_Warm Front': [0],
    'Front_Type_No Front': [0],
    'Time_of_Day_Morning': [1],
    'Time_of_Day_Afternoon': [0],
    'Time_of_Day_Evening': [0],
    'Season_Spring': [0],
    'Season_Summer': [0],
    'Season_Fall': [0],
    'Season_Winter': [1],
    'Cloud_Cover': [72.5],
    'Solar_Radiation': [690.3],
    'Land_Use_Type_Urban': [0],
    'Land_Use_Type_Suburban': [1],
    'Land_Use_Type_Rural': [0],
    'Elevation': [227.4],
    'Urban_Density_High': [0],
    'Urban_Density_Medium': [1],
    'Urban_Density_Low': [0],
    'Soil_Type_Loam': [1],
    'Soil_Type_Sandy': [0],
    'Soil_Type_Clay': [1]
})

predicted_probability3 = model.predict(new_data3[feature_names])
predicted_percentage3 = predicted_probability3[0] * 100
print(f"Predicted Thunderstorm Probability for new_data3: {predicted_percentage3:.2f}%")
