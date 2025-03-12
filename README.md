

# Advanced Fraud Detection & Analytics

![Fraud Detection Banner](https://via.placeholder.com/1200x300?text=Advanced+Fraud+Detection+%26+Analytics)

Welcome to the **Advanced Fraud Detection & Analytics** project! This repository demonstrates an end-to-end solution to detect fraudulent transactions using state-of-the-art machine learning techniques, anomaly detection methods, and API deployment with FastAPI.
## Deployed WebApp

You can access the live version of the web app here:  
[Deployed WebApp on Azure](https://fraud-api-app.azurewebsites.net/)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

Financial fraud is an ever-evolving threat. This project harnesses the power of both supervised and unsupervised learning to:
- **Detect fraudulent transactions** using models such as Random Forest, XGBoost, and Neural Networks.
- **Identify anomalies** with Isolation Forest, One-Class SVM, and Local Outlier Factor.
- **Visualize data and model performance** with interactive dashboards and detailed reports.
- **Deploy a scalable API** with FastAPI for real-time fraud prediction.

## Features

- **Data Exploration & Visualization:**  
  Gain insights into transaction patterns with interactive plots, PCA visualization, and distribution analysis.
  
- **Data Preprocessing & Imbalance Handling:**  
  Leverage techniques like scaling and SMOTE to ensure robust model training on imbalanced datasets.
  
- **Multiple Machine Learning Models:**  
  Train and compare several models including:
  - **Random Forest**
  - **XGBoost**
  - **Neural Networks (with Early Stopping)**
  
- **Anomaly Detection:**  
  Integrate unsupervised techniques to catch emerging fraud patterns.
  
- **API Deployment:**  
  Expose predictions through a lightweight, high-performance API using FastAPI.
  
- **Containerization:**  
  Dockerized environment for consistent deployments across different platforms.

## Tech Stack

- **Programming Language:** Python 3.11
- **Libraries & Frameworks:**  
  - Data Processing: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`, `xgboost`, `tensorflow`
  - Imbalanced Data Handling: `imbalanced-learn`
  - API: `fastapi`, `uvicorn`
  - Deployment: `docker`
  - Model Persistence: `joblib`
  
## Installation & Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/advanced-fraud-detection.git
   cd advanced-fraud-detection
2. **Create Virtual Enviroment:**
    python -m venv fraud_detection_env
    source fraud_detection_env/bin/activate  # On Windows: fraud_detection_env\Scripts\activate
3. **Intsall Dependencies:**
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
4. **Built Docker Image:**
    docker build -t fraud_api .
    docker run -p 8000:8000 fraud_api
## Usage
## Data Analysis & Model Training:
Run the training script to load data, explore, preprocess, train models, and evaluate performance. This script will generate visualizations and save trained models to the models/ directory.
  python train.py

Running the API:
After training, start the FastAPI server to serve real-time predictions.
  uvicorn main:app --host 0.0.0.0 --port 8000
Open your browser and navigate to http://127.0.0.1:8000/docs to access the interactive API documentation.

## API Endpoints
POST /predict
Send transaction data as a JSON payload to receive fraud predictions.
Request Body Example:
json

{
  "features": [0.1, 2.3, 4.5, 6.7, ...]  // List of numerical features for the transaction
}
Response Example:
json

{
  "RandomForest_Prediction": 0,
  "XGBoost_Prediction": 1
}

## Project Structure
advanced-fraud-detection/
├── data/
│   └── data.csv                 # Dataset file
├── models/
│   ├── random_forest_model.pkl  # Saved Random Forest model
│   ├── xgboost_model.pkl        # Saved XGBoost model
│   └── neural_network_model.keras  # Saved Neural Network model
├── notebooks/                   # Jupyter Notebooks for EDA and experiments
├── Dockerfile                   # Dockerfile for containerizing the API
├── main.py                      # FastAPI application
├── requirements.txt             # Project dependencies with exact versions
├── analysis.py                     # Training script for models & analysis
└── README.md                    # Project overview and documentation
