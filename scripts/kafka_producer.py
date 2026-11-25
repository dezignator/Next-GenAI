"""
Simulates a live DoH network tap sending traffic to Kafka.
Usage: python scripts/kafka_producer.py
"""
import json
import time
import random
from kafka import KafkaProducer
from src.data_acquisition.toy_dataset import ToyDoHDatasetGenerator
from src.utils.logger import configure_logging, get_logger

def main():
    configure_logging()
    log = get_logger("kafka_producer")
    
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    log.info("Starting DoH Traffic Producer...")
    gen = ToyDoHDatasetGenerator(n_benign=100000)
    
    try:
        while True:
            # Generate a small batch of traffic
            df = gen._generate_benign()
            # Pick 5 random packets to send
            for _, row in df.sample(5).iterrows():
                payload = row.to_dict()
                producer.send('doh-traffic-raw', payload)
                log.debug(f"Sent query: {payload['qname'][:20]}...")
                time.sleep(0.1)
    except KeyboardInterrupt:
        log.info("Producer stopped.")

if __name__ == "__main__":
    main()
