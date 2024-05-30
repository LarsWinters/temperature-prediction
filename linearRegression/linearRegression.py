import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Load the training dataset
train_dataset_path = os.path.join('..', 'dataset', 'train', 'temperature_data.csv')
train_df = pd.read_csv(train_dataset_path)

# Convert datetime to numerical format (e.g., timestamp)
train_df['datetime'] = pd.to_datetime(train_df['datetime'])
train_df['timestamp'] = train_df['datetime'].astype(int) / 10**9  # Convert to seconds

# Split the training data into features and target
X_train = train_df[['timestamp', 'current_temp', 'outside_temp', 'humidity', 'co2']]
y_train = train_df['predicted_temp']

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/temperature_model.pkl')
print('Model trained and saved to models/temperature_model.pkl')

# Load the test dataset
test_dataset_path = os.path.join('..', 'dataset', 'test', 'temperature_data.csv')
test_df = pd.read_csv(test_dataset_path)

# Convert datetime to numerical format (e.g., timestamp)
test_df['datetime'] = pd.to_datetime(test_df['datetime'])
test_df['timestamp'] = test_df['datetime'].astype(int) / 10**9  # Convert to seconds

# Split the test data into features and target
X_test = test_df[['timestamp', 'current_temp', 'outside_temp', 'humidity', 'co2']]
y_test = test_df['predicted_temp']

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R^2 Score: {r2}')

# Define the prediction model class
class PredictionModel:
    def __init__(self):
        self.model = joblib.load('models/temperature_model.pkl')

    def predict(self, input_data):
        current_temp = input_data['current_temperature']
        outside_temp = input_data['outside_temperature']
        humidity = input_data['humidity']
        co2 = input_data['co2']
        timestamp = pd.to_datetime(input_data['timestamp']).timestamp()

        features = [[timestamp, current_temp, outside_temp, humidity, co2]]
        predicted_temp = self.model.predict(features)[0]
        return predicted_temp
