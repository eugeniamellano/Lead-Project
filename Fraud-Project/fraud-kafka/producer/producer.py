from confluent_kafka import Producer
import json
import ccloud_lib
import numpy as np
import time
import requests
import datetime

# configurations from python.config 
CONF = ccloud_lib.read_ccloud_config("python.config") 
TOPIC = "fraud_detection" # use this topic


# Create Producer instance
producer_conf = ccloud_lib.pop_schema_registry_params_from_config(CONF)
producer = Producer(producer_conf) 

# Create topic 
ccloud_lib.create_topic(CONF, TOPIC)

delivered_records = 0

def acked(err, msg):
    global delivered_records
    
    if err is not None:
        print("Failed to deliver message: {}".format(err))
    else:
        delivered_records += 1
        print("Produced record to topic {} partition [{}] @ offset {}"
                .format(msg.topic(), msg.partition(), msg.offset()))

try: 
    # Infinite loop to continuously fetch transaction data from the API and send it to the Kafka topic.
    while True:
        url = "https://real-time-payments-api.herokuapp.com/current-transactions"
        response = requests.get(url)
        transaction = json.loads(response.json())["data"]
         
        record_value = json.dumps({"data": transaction})
        record_key  = str(datetime.datetime.now())


        print(record_key, record_value)

        # Send data to Kafka topic
        producer.produce(
            TOPIC,
            key=record_key, 
            value=record_value, 
            on_delivery=acked
        )
        
        
        producer.poll(0)
        time.sleep(15)

 # Interrupt infinite loop when hitting CTRL+C
except KeyboardInterrupt:
    pass
finally:
    producer.flush() # Finish producing the latest event before stopping the whole script