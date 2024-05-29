# Temperature Prediction

This repository provides tools and scripts for predicting temperatures using different machine learning models. It includes dataset generation, model training, evaluation, and comparison of different models.

## Table of Contents

- [Setup](#setup)
- [Dataset Generation](#dataset-generation)
- [Training Models](#training-models)
- [Testing and Comparing Models](#testing-and-comparing-models)
- [Managing Virtual Environments](#managing-virtual-environments)
- [Creating a new model](#creating-a-new-model)
- [Notes](#notes)

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/LarsWinters/temperature_prediction.git
   cd temperature_prediction
    ```
   Install Python and virtualenv:

Ensure you have Python 3.8+ and virtualenv installed. You can install virtualenv using pip:

```bash
pip install virtualenv
```

## Dataset Generation

The datasets train and test are already generated and stored in the dataset directory. 
It is based on average values from: https://www.laenderdaten.info/Europa/Oesterreich/Klima-Tirol.php
Those can be adjusted in the generate_dataset.py file.
If you want to generate a new dataset, follow these steps:

1. Navigate to the dataset directory and create a virtual environment:

    ```bash
    cd dataset
    python3 -m venv venv
    source venv/bin/activate  # For Unix/Linux/Mac
    # venv\Scripts\activate  # For Windows
    ```

2. Install required packages and generate the dataset:

    ```bash
    pip install numpy pandas
    python3 generate_dataset.py
    ```

This will generate a dataset and save it to the train directory.

3. Deactivate the virtual environment (when switching to another directory):

    ```bash
    deactivate
    ```

## Training Models

### Linear Regression Model

1. Navigate to the linearRegression directory and create a virtual environment:

```bash
cd ../linearRegression
python3 -m venv venv
source venv/bin/activate  # For Unix/Linux/Mac
# venv\Scripts\activate  # For Windows
```

2. Install required packages and train the model:

```bash
pip install -r requirements.txt
python3 linearRegression.py
```

3. deactivate the virtual environment (when switching to another directory):

```bash
deactivate
```

### Random Forest Model

1. Navigate to the randomForest directory and create a virtual environment:

```bash
cd ../randomForest
python3 -m venv venv
source venv/bin/activate  # For Unix/Linux/Mac
# venv\Scripts\activate  # For Windows
```

2. Install required packages and train the model:

```bash
pip install -r requirements.txt
python3 model.py
```

3. deactivate the virtual environment (when switching to another directory):

```bash
deactivate
```

## Testing and Comparing Models

1. Ensure all virtual environments are set up and models are trained.

2. Navigate to the project root directory and create a virtual environment:

```bash
cd ..
python3 -m venv venv
source venv/bin/activate  # For Unix/Linux/Mac
# venv\Scripts\activate  # For Windows
```

3. Install required packages and run the evaluation script:

```bash
pip install -r requirements.txt
```

4. Set environment variables for the input values:

```bash
export CURRENT_TEMP=<current_temp>
export OUTSIDE_TEMP=<outside_temp>
export HUMIDITY=<humidity>
export CO2=<co2>
export TIMESTAMP=<timestamp> # Format: "YYYY-MM-DD HH:MM:SS"
```

5. Run the test_models.py script to test and compare models using environment variables:

```bash
python3 test_models.py --current_temp $CURRENT_TEMP --outside_temp $OUTSIDE_TEMP --humidity $HUMIDITY --co2 $CO2 --timestamp $TIMESTAMP --model_paths "linearRegression/" "randomForest/"
```

6. View the results in the console and in the model_comparison_results.csv file.

7. Deactivate the virtual environment:

```bash
deactivate
```

## Managing Virtual Environments

### Creating a virtual environment

To create a virtual environment for a new model or script:

```bash
python3 -m venv venv # Create a new virtual environment "venv", change second for name of venv
source venv/bin/activate  # For Unix/Linux/Mac
# venv\Scripts\activate  # For Windows
```

### Installing packages

```bash
pip install -r requirements.txt
```

### Deactivating a virtual environment

```bash
deactivate
```

### Switching between virtual environments

```bash
deactivate
cd ../<desired_directory>
source venv/bin/activate  # For Unix/Linux/Mac
# venv\Scripts\activate  # For Windows
```

## Creating a new model

To create a new model, follow these steps:

1. Go to root directory

2. Create a new directory for the model:

```bash
mkdir <model_name>
cd <model_name>
```

3. Create a virtual environment:

```bash
python3 -m venv <model_name>_venv
source <model_name>_venv/bin/activate  # For Unix/Linux/Mac
# <model_name>_venv\Scripts\activate  # For Windows
```

4. Create requirements.txt and __init__.py

```bash
touch requirements.txt
touch __init__.py
```

Write down all required packages for this model in the requirements.txt file.

5. Install required packages:

```bash
pip install -r requirements.txt
```

6. Create the model script:

    ```bash
    touch <model_name>.py
    ```
   
Write the model code in this file. It should follow those principles:

- Load the dataset from the train directory (dataset/train/temperature_data.csv).
- Provide a class:

```python
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
```
That is necessary so the model can be used by the prediction_wrapper. The model should be stored 
in the models/ directory as an "temperature_model.pkl" file.

## Notes

If there are any issues with the documentation or code of this repository reach out to me. Or use 
the issue tracker to report bugs or request new features.
