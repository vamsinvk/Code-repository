import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Path to your dataset
data = r'C:\Users\vamsi\Documents\python problems\synthetic_thunderstorm_data.csv'

# Read the CSV file into a DataFrame
synthetic_data = pd.read_csv(data)

# Separate features and target variable
y = synthetic_data['Thunderstorm_Probability']  # Target variable
X = synthetic_data.drop(columns=['Thunderstorm_Probability'])  # Features

# Convert categorical variables to dummy/indicator variables
X = pd.get_dummies(X, columns=['Front_Type', 'Time_of_Day', 'Season', 'Land_Use_Type', 'Urban_Density', 'Soil_Type'])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Define new data and ensure it has the same columns
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

# Ensure new_data1 has the same columns as the training data
new_data1 = new_data1.reindex(columns=X.columns, fill_value=0)

# Make predictions on the new data
predicted_probability = model.predict(new_data1)
predicted_percentage = predicted_probability[0] * 100

# Print given data and prediction
print("Given Data for new_data1:")
print(new_data1)
print("\nPredicted Thunderstorm Probability for new_data1:")
print(f"{predicted_percentage:.2f}%")
