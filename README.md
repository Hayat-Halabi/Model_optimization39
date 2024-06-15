# Model_optimization39
redicting Diabetes Progression
# Scenario:
Medical researchers at a leading hospital want to predict patient diabetes progression based on certain health metrics. They've collected data from a group of patients, which includes various metrics like age, BMI (body mass index), average blood pressure, and six blood serum measurements.

# Problem Statement:
Develop a regression model that predicts the quantitative measure of disease progression one year after baseline.

# Objective:
The goal is to build and compare regression models to aid medical researchers in predicting diabetes progression, which can be crucial for patient care and treatment planning.

# Direction:
### Data Loading and Initial Exploration:

Load the dataset using load_diabetes() from the sklearn.datasets module.
Print the shape of the data to understand its dimensionality.
Explore the first few rows of the dataset to get a glimpse of the available features.
### Preprocessing the Data:

Normalization: Before feeding the data to a neural network, it's important to scale the features so that they have similar scales. Use StandardScaler from sklearn.preprocessing to standardize the dataset.
Train-Test Split: Split the dataset into a training set and a test set. This will allow you to train the model on one subset and validate its performance on another unseen subset.
A common split ratio is 80% for training and 20% for testing.
### Building and Training the Model:

Create a neural network model using build_model with a specified optimizer, either RMSprop or Adadelta.
Create two models: one with the RMSprop optimizer using model_rmsprop' and another with the Adadelta optimizer using'model_adadelta'.
Both models are trained on the same training data (X_train and y_train) for 100 epochs with a 10% validation split.
The training history for each model is stored in history_rmsprop and history_adadelta, respectively.
### Evaluating the Model:

Once both models are trained, evaluate their performance on the test dataset.
Calculate performance metrics like MSE and MAE on the test dataset for both models.
### Conclusion:

Compare the performance metrics (MSE and MAE) of the two models.
Based on the comparison, draw conclusions and decide on the best optimization algorithm for this problem.
# Data Loading and Initial Exploration:
```python
# Import necessary libraries/modules
from sklearn.datasets import load_diabetes  # Import the load_diabetes function from sklearn.datasets
import tensorflow as tf  # Import the TensorFlow library and alias it as 'tf'
from tensorflow import keras  # Import the Keras submodule from TensorFlow and alias it as 'keras'
from sklearn.model_selection import train_test_split  # Import train_test_split function from sklearn.model_selection
from sklearn.preprocessing import StandardScaler  # Import the StandardScaler class from sklearn.preprocessing

# Load the diabetes dataset using the load_diabetes function from sklearn.datasets
data = load_diabetes()

# Assign the features (input data) to the variable 'X'
X = data.data

# Assign the target values (output data) to the variable 'y'
y = data.target
X
```
### Observation
Load the diabetes dataset, assign input features to 'X,' and target values to 'y' using scikit-learn and TensorFlow/Keras libraries.

### Preprocessing the Data
Create a StandardScaler object for feature scaling.
Scale (normalize) the input features 'X' using the StandardScaler.
Split the dataset into training and testing datasets.
```python
# Create a StandardScaler object for feature scaling
scaler = StandardScaler()

# Scale (normalize) the input features 'X' using the StandardScaler
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing datasets
# X_train: Training features, X_test: Testing features, y_train: Training target values, y_test: Testing target values
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```
### Observation
The input features are scaled using StandardScaler, and the dataset has been split into training and testing sets.

# Building and Training the Model
Define a function build_model that creates a Sequential model with specific architecture.
Compile the model with mean squared error (MSE) loss and the specified optimizer.
```
X_train.shape[1]
# Define a function 'build_model' that creates a Sequential model with specific architecture
def build_model(optimizer):
    # Create a Sequential model with layers: 64-node input layer, 32-node hidden layer, and 1-node output layer
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    # Compile the model with mean squared error (MSE) loss and the specified optimizer
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    return model

# Build and train a model with RMSprop optimizer
model_rmsprop = build_model(optimizer='rmsprop')
history_rmsprop = model_rmsprop.fit(X_train, y_train, epochs=100, validation_split=0.1, verbose=0)

# Build and train a model with Adadelta optimizer
model_adadelta = build_model(optimizer='adadelta')
history_adadelta = model_adadelta.fit(X_train, y_train, epochs=100, validation_split=0.1, verbose=0)
```
Observation
It defines a function to build neural network models with different optimizers (RMSprop and Adadelta) and trains them for 100 epochs with a 10% validation split.

### Evaluating the Model
Evaluate the RMSProp model on the testing data and retrieve MSE and MAE.
Evaluate the Adadelta model on the testing data and retrieve MSE and MAE.
Print the MSE and MAE for both models.
```
# Evaluate the RMSprop model on the testing data and retrieve MSE and MAE
mse_rmsprop, mae_rmsprop = model_rmsprop.evaluate(X_test, y_test, verbose=0)

# Evaluate the Adadelta model on the testing data and retrieve MSE and MAE
mse_adadelta, mae_adadelta = model_adadelta.evaluate(X_test, y_test, verbose=0)

# Print the MSE and MAE for both models
print(f"RMSprop - MSE: {mse_rmsprop}, MAE: {mae_rmsprop}")
print(f"Adadelta - MSE: {mse_adadelta}, MAE: {mae_adadelta}")
```
### Conclusion:
Here we calculate and display the mean squared error (MSE) and mean absolute error (MAE) for two trained models (RMSprop and Adadelta) on the testing data.i.e..,RMSprop and Adadelta - MSE, MAE are as given above

RMSProp has optimized the model better than Adadelta.

