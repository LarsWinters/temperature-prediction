import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

# Average temperatures per month (degree Celsius)
avg_temps = {
    1: {'day': 2.3, 'night': -6.0},
    2: {'day': 4.5, 'night': -5.2},
    3: {'day': 8.5, 'night': -2.3},
    4: {'day': 13.3, 'night': 1.7},
    5: {'day': 16.7, 'night': 5.7},
    6: {'day': 21.2, 'night': 9.9},
    7: {'day': 22.8, 'night': 11.4},
    8: {'day': 21.8, 'night': 11.0},
    9: {'day': 18.3, 'night': 7.9},
    10: {'day': 14.0, 'night': 4.0},
    11: {'day': 7.7, 'night': -0.8},
    12: {'day': 3.1, 'night': -4.6}
}

def generate_dataset():
    np.random.seed(0)

    # Define parameters
    train_months = list(range(1, 12)) # January to November
    print(train_months)
    test_month = [12]  # December
    days_in_month = 30
    hours_in_day = 24

    def generate_data_for_month(months):
        num_samples = len(months) * days_in_month * hours_in_day
        datetimes = []
        current_temp = []
        outside_temp = []

        for month in months:
            avg_temp_day = avg_temps[month]['day']
            avg_temp_night = avg_temps[month]['night']
            start_date = datetime(year=2023, month=month, day=1)
            for day in range(days_in_month):
                for hour in range(hours_in_day):
                    current_datetime = start_date + timedelta(hours=hour + day * hours_in_day)
                    datetimes.append(current_datetime)
                    if 6 <= hour < 18:  # Daytime
                        outside_temp_hourly = np.random.normal(avg_temp_day, 2)
                    else:  # Nighttime
                        outside_temp_hourly = np.random.normal(avg_temp_night, 2)
                    current_temp_hourly = np.random.normal(avg_temp_day + 2, 1)  # Less variation due to insulation
                    current_temp.append(current_temp_hourly)
                    outside_temp.append(outside_temp_hourly)

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

        return pd.DataFrame(data)

    # Generate train dataset
    train_df = generate_data_for_month(train_months)
    output_dir = os.path.join(os.path.dirname(__file__), 'train')
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'temperature_data.csv')
    train_df.to_csv(train_path, index=False)
    print(f'Train dataset generated and saved to {train_path}')

    # Generate test dataset
    test_df = generate_data_for_month(test_month)
    output_dir = os.path.join(os.path.dirname(__file__), 'test')
    os.makedirs(output_dir, exist_ok=True)
    test_path = os.path.join(output_dir, 'temperature_data.csv')
    test_df.to_csv(test_path, index=False)
    print(f'Test dataset generated and saved to {test_path}')

if __name__ == '__main__':
    generate_dataset()
