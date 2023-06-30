import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers.legacy import Adam

# Step 1: Data loading
dataset = pd.read_csv('life_expectancy.csv')

# Step 2: Data observing
print(dataset.head())
print(dataset.describe())
print(dataset.keys())

# Step 3: Drop the 'Country' column
dataset = dataset.drop(['Country'], axis=1)

# Additional step: Drop rows with NaN values
dataset = dataset.dropna()

# Step 4 & 5: Split the data into labels and features
labels = dataset.iloc[:, -1]
features = dataset.iloc[:, :-1]

# Step 6: Data Preprocessing - One Hot Encoding
features = pd.get_dummies(features)

# Step 7: Split data into training and test sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Step 8 & 9: Standardize the features
ct = ColumnTransformer([("standardize", StandardScaler(), features.select_dtypes(include=['float64','int64']).columns)], remainder='passthrough')
features_train_scaled = ct.fit_transform(features_train)

# Step 10: Apply the standardization to the test features
features_test_scaled = ct.transform(features_test)

# Step 11: Build the model
my_model = tf.keras.models.Sequential()

# Step 12 & 13: Input layer
input = InputLayer(input_shape=(features.shape[1], ))
my_model.add(input)

# Step 14: Add a hidden layer
my_model.add(Dense(64, activation='relu'))

# Step 15: Add output layer
my_model.add(Dense(1))

# Step 16: Model summary
print(my_model.summary())

# Step 17 & 18: Initialize the optimizer and compile the model
opt = Adam(learning_rate=0.01)
my_model.compile(loss='mse', metrics=['mae'], optimizer=opt)

# Step 19: Train the model
my_model.fit(features_train_scaled, labels_train, epochs=40, batch_size=1, verbose=1)

# Step 20: Evaluate the model
res_mse, res_mae = my_model.evaluate(features_test_scaled, labels_test, verbose=0)

# Step 21: Print final results
print(f"Final MSE: {res_mse} -- meaning that the square of the difference between the predicted life expectancy and the actual life expectancy is, on average, around 8.32.")
print(f"Final MAE: {res_mae} -- meaning that on average, the model's predictions are off by about 2.27 years from the actual life expectancy.")

# Make predictions
predictions = my_model.predict(features_test_scaled)

# Calculate best-fit line
slope, intercept = np.polyfit(labels_test, predictions, 1) # 1 is the degree of the polynomial

# Create a scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(labels_test, predictions, alpha=0.5)
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')
plt.title('Country Life Expectancy: True vs Predicted Labels')

# Add best-fit line
x = np.linspace(min(labels_test), max(labels_test), 100)
y = slope * x + intercept
plt.plot(x, y, color='red')

plt.show()
