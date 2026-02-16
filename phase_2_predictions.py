# phase_2_predictions.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
import joblib

# Function to implement shot-based features
def extract_shot_based_features(data):
    # Example feature extraction
    data['shot_difference'] = data['shots_on_target'] - data['total_shots']
    data['shot_accuracy'] = data['shots_on_target'] / data['total_shots']
    return data

# Function for time-series validation
def time_series_validation(data, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_index, test_index in tscv.split(data):
        yield data.iloc[train_index], data.iloc[test_index]

# Function for Bayesian diagnostics
def run_bayesian_diagnostics(data):
    with pm.Model() as model:
        # Define priors
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=1, shape=data.shape[1])
        sigma = pm.HalfNormal('sigma', sigma=1)

        # Define likelihood
        mu = alpha + pm.math.dot(data.iloc[:, :-1], beta)
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=data.iloc[:, -1])
        
        # Posterior sampling
        trace = pm.sample(2000, return_inferencedata=False)
    return trace

# Function to track feature importance
def track_feature_importance(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    importance = model.feature_importances_
    return importance

if __name__ == "__main__":
    # Example usage
    # Load your dataset
    data = pd.read_csv('your_data.csv')
    
    # Step 1: Extract features
    data = extract_shot_based_features(data)

    # Step 2: Validate time-series
    for train_data, test_data in time_series_validation(data):
        print("Train:", train_data.shape, "Test:", test_data.shape)
    
    # Step 3: Run Bayesian Diagnostics
    trace = run_bayesian_diagnostics(data)
    
    # Step 4: Feature Importance
    importance = track_feature_importance(data.iloc[:, :-1], data.iloc[:, -1])
    print("Feature Importance:", importance)

    # Save the model if needed
    joblib.dump(model, 'forest_model.pkl')