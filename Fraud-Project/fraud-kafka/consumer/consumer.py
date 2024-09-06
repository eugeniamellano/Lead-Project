from confluent_kafka import Consumer, Producer
import json
import ccloud_lib
import pandas as pd
import mlflow
import mlflow.pyfunc
import numpy as np
import os
import time
import boto3
from geopy.distance import great_circle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pickle
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine
from dotenv import load_dotenv

# start config from "python.config"
CONF = ccloud_lib.read_ccloud_config("python.config")
TOPIC_INPUT = "fraud_detection"  # este donde llegan los datos
TOPIC_OUTPUT = "fraud_prediction"  # este para la prediccion

# Create Consumer instance
consumer_conf = ccloud_lib.pop_schema_registry_params_from_config(CONF)
consumer_conf["group.id"] = "fraud_detection_group"
consumer_conf["auto.offset.reset"] = "earliest"
consumer = Consumer(consumer_conf)

# topic suscribe
consumer.subscribe([TOPIC_INPUT])

# URI from my MLFLOW
mlflow.set_tracking_uri("https://mlflow-lead-a5e0197c9b59.herokuapp.com/")

# List of models
for run in mlflow.search_runs():
    print(run)

# model from MLflow
logged_model = "runs:/83293412ba6c4240a6ccb6c5706c3314/fraud_xgb_model"
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Create Producer instance
producer_config = ccloud_lib.pop_schema_registry_params_from_config(CONF)
producer = Producer(producer_config)

# Create topic if doesnt exists
ccloud_lib.create_topic(CONF, TOPIC_OUTPUT)  # to be sure if topic exists

# distance merch - transaction
def calculate_distance(row):
    try:
        coord1 = (row["lat"], row["long"])
        coord2 = (row["merch_lat"], row["merch_long"])
        return great_circle(coord1, coord2).kilometers
    except:
        return None

# delete columns if we dont use them in mlflow
def delete_columns(df):
    columns_to_drop = [
        "current_time",
        "cc_num",
        "merchant",
        "job",
        "first",
        "last",
        "street",
    ]
    df = df.drop(
        columns=[col for col in columns_to_drop if col in df.columns], errors="ignore"
    )
    return df


def acked(err, msg):
    if err is not None:
        print(f"Failed to deliver message: {err}")
    else:
        print(
            f"Produced record to topic {msg.topic()} partition [{msg.partition()}] @ offset {msg.offset()}"
        )


# num at cat features
categorical_features = [
    "category",
    "gender",
    "city",
    "state",
    "zip",
    "dob",
    "trans_num",
]
numeric_features = [
    "amt",
    "lat",
    "long",
    "city_pop",
    "merch_lat",
    "merch_long",
    "distance",
]

# preprocessor of the model with pickle
with open("preprocessor_fraud.pkl", "rb") as file:
    loaded_preprocessor = pickle.load(file)


# Send email with transaction info
def send_email(predictions):  
    sender_email = os.getenv("EMAIL_USER")   
    receiver_email = os.getenv("EMAIL_USER") #here you can change email user. this is just to test 
    password = os.getenv("EMAIL_PASSWORD")

    subject = "Fraud Detection Alert"  
    #body = f"The following transactions were flagged as fraud:\n{predictions}" 
    body = f"The following transaction ID was processed as a Fraud: {transaction_id[0]}"

    msg = MIMEMultipart()  
    msg["From"] = sender_email  
    msg["To"] = receiver_email  
    msg["Subject"] = subject  

    msg.attach(MIMEText(body, "plain"))  

    try:  
        server = smtplib.SMTP("smtp.gmail.com", 587)  
        server.starttls()  
        server.login(sender_email, password)  
        text = msg.as_string()  
        server.sendmail(sender_email, receiver_email, text)  
        server.quit()  
        print("Email sent successfully")  
    except Exception as e:  
        print(f"Failed to send email: {e}")  


#Connection
database_url = os.getenv("DATABASE_URL")
conn = psycopg2.connect(database_url)


#Create table  - SQLAlchemy
cur = conn.cursor()

#Verify if table already exist
cur.execute("""
    SELECT to_regclass('fraud_table_sql')
""")

if cur.fetchone()[0] is None:
    # Create table
    cur.execute("""
        CREATE TABLE fraud_table_sql (
                transaction_id VARCHAR(255),
                prediction_result INTEGER
 )
    """)
    conn.commit()
    print("Table created")
else:
    print("Table already exist")   


def upload_csv_to_dh(df):
    # checking columns
    df = df[['trans_num', 'is_fraud']]
    df.columns = ['transaction_id', 'prediction_result']  # rename
    database_url = os.getenv("DATABASE_URL")
    

    engine = create_engine(database_url)

    df.to_sql('fraud_table_sql', engine, if_exists='append', index=False)

    print('df loaded')

    engine.dispose()

def create_dataframe_from_json(data):
    df_cols = [
        "cc_num",
        "merchant",
        "category",
        "amt",
        "first",
        "last",
        "gender",
        "street",
        "city",
        "state",
        "zip",
        "lat",
        "long",
        "city_pop",
        "job",
        "dob",
        "trans_num",
        "merch_lat",
        "merch_long",
        "is_fraud",
        "current_time",
    ]
    
    # Extract list of transactions
    data_list = data["data"]
    
    # Create df
    data_api = pd.DataFrame(data_list, columns=df_cols)
    
    return data_api

# Procesar 
try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        elif msg.error():
            print(f"Error: {msg.error()}")
        else:
            record_key = msg.key()
            record_value = msg.value()
            data = json.loads(record_value)
            #print(f"New transaction recorded: {record_key} with {record_value}")
            print("New transaction recorded")

            # Create df from JSON
            data_api = create_dataframe_from_json(data)

            # Data preprocessing
            data_api = delete_columns(data_api)
            data_api["distance"] = data_api.apply(calculate_distance, axis=1)
            X_after_prep = loaded_preprocessor.transform(data_api)

            # predictions 
            predictions = loaded_model.predict(X_after_prep)
            data_api["is_fraud"] = predictions


            # Convert df to JSON
            df_to_json = data_api.to_json(orient="records")
            
            # RENDER DB 
            upload_csv_to_dh(data_api)  

            # Sent result to topic output
            producer.produce(
                TOPIC_OUTPUT, key=record_key, value=df_to_json, callback=acked
            )
            producer.poll(0)
            producer.flush()


            # filter of last fraud transaction
            latest_fraudulent_transaction = data_api[data_api["is_fraud"] == 1].iloc[-1:].to_dict(orient="records")

            if latest_fraudulent_transaction:
                if latest_fraudulent_transaction != last_fraudulent_transaction:
                    send_email(latest_fraudulent_transaction[0])
                    last_fraudulent_transaction = latest_fraudulent_transaction

            
            ###################################
            # THIS IS A SIMULATION OF A FRAUD NOTIFICATION (email sent for all transactions recorded)
            # This line is only for testing the consumer. It sends an email for every transaction recorded.
            # In production, this line should be removed or modified to only send emails for fraudulent transactions.
            transaction_id = data_api["trans_num"] #delete this line
            
            send_email(transaction_id)

            # print prediction
            prediction_label = "Fraud" if predictions[0] == 1 else "No Fraud" 
            print(f"Prediction: {prediction_label}")

            time.sleep(1)  # wait one sec before continue

except KeyboardInterrupt:
    pass
finally:
    consumer.close()
    producer.flush()
    conn.close()
