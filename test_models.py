import argparse
import os

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tabulate import tabulate


class PredictionModel:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, current_temp, outside_temp, humidity, co2, timestamp):
        # Convert timestamp to numerical format (seconds since epoch)
        timestamp = pd.to_datetime(timestamp).timestamp()
        features = [[timestamp, current_temp, outside_temp, humidity, co2]]
        predicted_temp = self.model.predict(features)[0]
        return predicted_temp

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, rmse, r2

def main():
    parser = argparse.ArgumentParser(description='Test multiple temperature prediction models and compare the results.')
    parser.add_argument('--current_temp', type=float, required=True, help='Current temperature')
    parser.add_argument('--outside_temp', type=float, required=True, help='Outside temperature')
    parser.add_argument('--humidity', type=float, required=True, help='Humidity')
    parser.add_argument('--co2', type=float, required=True, help='CO2 level')
    parser.add_argument('--timestamp', type=str, required=True, help='Timestamp (e.g., "2023-06-01 12:00:00")')
    parser.add_argument('--model_paths', nargs='+', required=True, help='Directories of the trained models')

    args = parser.parse_args()

    results = []
    for model_dir in args.model_paths:
        model_name = os.path.basename(os.path.normpath(model_dir))
        model_path = os.path.join(model_dir, 'models', 'temperature_model.pkl')
        model = PredictionModel(model_path)
        predicted_temp = model.predict(args.current_temp, args.outside_temp, args.humidity, args.co2, args.timestamp)

        # Load dataset for evaluation
        dataset_path = os.path.join('dataset', 'train', 'temperature_data.csv')
        df = pd.read_csv(dataset_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['timestamp'] = df['datetime'].astype(int) / 10**9  # Convert to seconds
        X = df[['timestamp', 'current_temp', 'outside_temp', 'humidity', 'co2']]
        y = df['predicted_temp']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Evaluate the model
        mae, mse, rmse, r2 = evaluate_model(model.model, X_test, y_test)

        results.append([model_name, predicted_temp, mae, mse, rmse, r2])

    # Create a DataFrame for the results
    df_results = pd.DataFrame(results, columns=['Model', 'Predicted Temperature', 'MAE', 'MSE', 'RMSE', 'R^2 Score'])

    # Save the results to a CSV file
    df_results.to_csv('model_comparison_results.csv', index=False)

    # Print the results in a pretty format
    print(tabulate(df_results, headers='keys', tablefmt='pretty'))

if __name__ == "__main__":
    main()
