import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
import time

# Load the MNIST dataset
print("Loading dataset...")
start_time = time.time()
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target
print(f"Dataset loaded in {time.time() - start_time:.2f} seconds.")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Random Forest model
model = RandomForestClassifier(random_state=42, n_jobs=-1)  # n_jobs=-1 uses all available cores

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],     # Number of trees in the forest
    'max_depth': [10, 20, 30],           # Maximum depth of each tree
    'min_samples_split': [2, 5, 10]      # Minimum number of samples required to split an internal node
}

# Use GridSearchCV to perform cross-validated hyperparameter tuning
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2)

# Fit the model with grid search
print("Starting grid search...")
start_time = time.time()
grid_search.fit(X_train, y_train)
print(f"Grid search completed in {time.time() - start_time:.2f} seconds.")

# Print the best parameters and the best score
print("Best parameters found:", grid_search.best_params_)
print("Best cross-validated accuracy:", grid_search.best_score_)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test set accuracy of the best model: {accuracy:.2f}")

