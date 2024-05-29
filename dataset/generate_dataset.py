import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

def generate_dataset():
    np.random.seed(0)

    # Define parameters
    months = [1, 2, 3, 4, 5, 6]
    days_in_month = 30
    hours_in_day = 24

    num_samples = len(months) * days_in_month * hours_in_day

    # Generate datetime values
    start_date = datetime(year=2023, month=1, day=1)
    datetimes = [start_date + timedelta(hours=i) for i in range(num_samples)]

    current_temp = []
    outside_temp = []

    for month in months:
        avg_temp_month = 5 + (month * 3)  # Increase average temperature with each month
        for day in range(days_in_month):
            for hour in range(hours_in_day):
                if 6 <= hour < 18:  # Daytime
                    outside_temp_hourly = np.random.normal(avg_temp_month + 5, 2)
                else:  # Nighttime
                    outside_temp_hourly = np.random.normal(avg_temp_month - 5, 2)

                current_temp_hourly = np.random.normal(avg_temp_month + 2, 1)  # Less variation due to insulation

                current_temp.append(current_temp_hourly)
                outside_temp.append(outside_temp_hourly)

    # Generate other variables
    humidity = np.random.uniform(20, 90, num_samples)
    co2 = np.random.uniform(300, 500, num_samples)
    predicted_temp = 0.5 * np.array(current_temp) + 0.3 * np.array(outside_temp) + 0.1 * humidity + 0.1 * co2 / 1000 + np.random.normal(0, 0.5, num_samples)

    data = {
        'datetime': datetimes,
        'current_temp': current_temp,
        'outside_temp': outside_temp,
        'humidity': humidity,
        'co2': co2,
        'predicted_temp': predicted_temp
    }

    df = pd.DataFrame(data)

    # Save dataset in the train directory at the same level as generate_dataset.py
    output_dir = os.path.join(os.path.dirname(__file__), 'train')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'temperature_data.csv')
    df.to_csv(output_path, index=False)
    print(f'Dataset generated and saved to {output_path}')

if __name__ == '__main__':
    generate_dataset()
