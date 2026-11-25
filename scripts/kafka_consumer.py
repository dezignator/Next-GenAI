"""
Listens for DoH traffic, runs the AI model, and emits alerts.
Usage: python scripts/kafka_consumer.py
"""
import json
import torch
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
from src.models.hybrid_model import HybridModel, HybridConfig
from src.data_acquisition.preprocessing import ALL_ENGINEERED_COLS
from src.utils.logger import configure_logging, get_logger

def main():
    configure_logging()
    log = get_logger("kafka_consumer")
    
    # Setup Kafka
    consumer = KafkaConsumer(
        'doh-traffic-raw',
        bootstrap_servers=['localhost:9092'],
        group_id='nextgenai-detector',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    alert_producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    # Setup Model
    model = HybridModel(HybridConfig(capsnet={"input_dim": 24})).eval()
    log.info("Model loaded. Listening for DoH traffic on Kafka...")
    
    for msg in consumer:
        data = msg.value
        
        # Dummy feature extraction (matching toy dataset length)
        features = np.random.randn(1, 24).astype(np.float32)
        x = torch.tensor(features)
        
        with torch.no_grad():
            probs, _ = model(x, return_aux=True)
            
        mal_prob = probs[0, 1].item()
        if mal_prob > 0.5:
            log.warning(f"🚨 THREAT DETECTED! Prob: {mal_prob:.4f} | Domain: {data.get('qname')}")
            alert_producer.send('doh-alerts', {
                "domain": data.get('qname'),
                "probability": mal_prob,
                "action": "BLOCK"
            })

if __name__ == "__main__":
    main()
