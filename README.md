# Automatic Fraud Detection 🦊
📇 Context
Fraud is a major challenge for financial institutions, with the European Central Bank estimating that in 2019 alone, fraudulent credit card transactions in the EU exceeded €1 billion! AI has proven to be an effective tool in combating fraud, particularly in detecting fraudulent payments with high precision.

However, deploying these powerful algorithms into production environments presents a significant challenge. The goal is not only to predict fraudulent payments accurately but to do so in real-time and respond appropriately.

## Project Goals 🎯

Build a Fraudulent Payment Detector: develop an algorithm capable of identifying fraudulent transactions.

Create Real-time Data Ingestion Infrastructure: Design an infrastructure that ingests real-time payment data.

Classify Payments and Provide Real-time Feedback: Automatically classify each payment and instantly send predictions to a notification center.

## Data Sources 📊
To accomplish this project, two data sources have been utilized:

Fraudulent Payments Dataset: A comprehensive dataset of transactions labeled as fraudulent or not. This dataset was used to train and validate the fraud detection algorithm.

**[Real-time Payment API](https://real-time-payments-api.herokuapp.com/)**: This API provides real-time payment data, enabling the real-time prediction functionality.

## Project Structure 🛠️
new_train_fraud.py: Script for building and training the fraud detection model, logging the model, and metrics using MLflow.

consumer.py: Script to consume real-time payment data from Kafka, preprocess it, predict fraud using the model, and store the results in a database. Also handles notification sending.

producer.py: Script to produce and send real-time payment data to the Kafka topic.

## Dashboard Overview
In addition to the fraud detection system, I have created a Streamlit dashboard to explore different characteristics of fraudulent transactions and view the results of fraud alerts. The dashboard includes:

- Visualizations of fraud transaction data.
- A transaction verification tool where users can check specific transactions.
- Graphical representations of fraud distribution, amount distribution, and category distribution by fraud status.

The dashboard was built using various tools such as Pandas, Plotly, and Lottie animations.

Link : **[Dashboard](https://streamlit-fraud-0216b16dc2bd.herokuapp.com/)**

---
Created by Eugenia M.
