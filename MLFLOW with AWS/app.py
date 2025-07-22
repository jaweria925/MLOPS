# pylint: disable=syntax-error
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score #performance metrics
from sklearn.linear_model import ElasticNet
import mlflow
from mlflow.models import infer_signature # setup the schema of the model
import mlflow.sklearn   
import logging
import urllib
from urllib.parse import urlparse
# Configure logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
# Set the tracking URI for MLflow


# Set the experiment name


# read the data

wine_url = "https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/master/winequality-white.csv"

try:
    df = pd.read_csv(wine_url, sep=';')
except Exception as e:
    logger.error(f"Error reading the data: {e}")
    sys.exit(1)

# Train-test split
X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

alpha = 1
l1_ratio = 1

# Train the model

with mlflow.start_run():
    try:
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(yTest, y_pred)
        mse = mean_squared_error(yTest, y_pred)
        r2 = r2_score(yTest, y_pred)

        # Log parameters and metrics
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

    # set up the remote server on aws using EC2 Instance

 # Set your remote server URI here
        mlflow.set_tracking_uri("http://ec2-3-25-226-147.ap-southeast-2.compute.amazonaws.com:5000")
        mlflow.set_experiment("Wine Quality Prediction Experiment")

        tracking_url_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_store != "file":
            mlflow.sklearn.log_model(model, "model", registered_model_name="WineQualityModel")
        else:
            mlflow.sklearn.log_model(model, "model")

    except Exception as e:
        # if there is an error in training the model, log the error and exit
        logger.error(f"Error training the model: {e}")
        sys.exit(1)


    