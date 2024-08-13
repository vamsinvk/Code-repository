import random
import pandas as pd
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
import os

random.seed(42)

data_records = []

for _ in range(5000):
    latitude = round(random.uniform(30, 40), 4)
    longitude = round(random.uniform(-120, -70), 4)
    temperature = round(random.uniform(10, 35), 1)
    humidity = round(random.uniform(30, 90), 1)
    pressure = round(random.uniform(980, 1030), 1)
    wind_speed = round(random.uniform(0, 20), 1)
    precipitation = round(random.uniform(0, 50), 2)
    stability_index = round(random.uniform(10, 30), 1)
    front_type = random.choice(['Cold Front', 'Warm Front', 'No Front'])
    time_of_day = random.choice(['Morning', 'Afternoon', 'Evening'])
    season = random.choice(['Spring', 'Summer', 'Fall', 'Winter'])
    cloud_cover = round(random.uniform(0, 100), 1)
    solar_radiation = round(random.uniform(400, 800), 1)
    land_use_type = random.choice(['Urban', 'Suburban', 'Rural'])
    elevation = round(random.uniform(50, 300), 1)
    urban_density = random.choice(['Low', 'Medium', 'High'])
    soil_type = random.choice(['Loam', 'Sandy', 'Clay'])
    thunderstorm_probability = round(random.uniform(0.0, 1.0), 2)

    data_records.append([
        latitude, longitude, temperature, humidity, pressure, wind_speed, precipitation,
        stability_index, front_type, time_of_day, season, cloud_cover, solar_radiation,
        land_use_type, elevation, urban_density, soil_type, thunderstorm_probability
    ])

columns = ['Latitude', 'Longitude', 'Temperature', 'Humidity', 'Pressure', 'Wind_Speed',
           'Precipitation', 'Stability_Index', 'Front_Type', 'Time_of_Day', 'Season',
           'Cloud_Cover', 'Solar_Radiation', 'Land_Use_Type', 'Elevation', 'Urban_Density',
           'Soil_Type', 'Thunderstorm_Probability']
synthetic_data = pd.DataFrame(data_records, columns=columns)



#synthetic_data.to_csv('synthetic_thunderstorm_data.csv', index=False)
# Saving DataFrame to a CSV file with '|' as the delimiter
synthetic_data.to_csv('synthetic_thunderstorm_data.csv', sep='|', index=False)

print(synthetic_data.head())
# Ensure the directory exists
directory = r'C:\Users\vamsi\Documents\python problems'
if not os.path.exists(directory):
    os.makedirs(directory)

# Specify the file path
file_path = os.path.join(directory, 'synthetic_thunderstorm_data.csv')

# Save the DataFrame to a CSV file
synthetic_data.to_csv(file_path, index=False)

print(f"\nData saved to {file_path}")