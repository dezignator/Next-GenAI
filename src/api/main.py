"""
src/api/main.py
===============
FastAPI endpoint for real-time DoH threat detection.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from typing import List, Dict

from src.models.hybrid_model import HybridModel, HybridConfig
from src.explainability.shap_explainer import SHAPExplainer

app = FastAPI(title="Next-GenAI DoH Detector", version="2.0.0")

# Global state
model = None
explainer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

class DoHFeatures(BaseModel):
    features: List[float]

@app.on_event("startup")
def load_model():
    global model, explainer
    # In production, load actual weights via torch.load()
    model = HybridModel(HybridConfig(capsnet={"input_dim": 24})).to(device).eval()
    explainer = SHAPExplainer(model, background_data=np.zeros((1, 24)), feature_names=[f"f{i}" for i in range(24)])

@app.post("/predict")
def predict(payload: DoHFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        x = torch.tensor([payload.features], dtype=torch.float32).to(device)
        with torch.no_grad():
            probs, _ = model(x, return_aux=True)
        
        mal_prob = float(probs[0, 1].item())
        verdict = "malicious" if mal_prob >= 0.5 else "benign"
        
        return {
            "verdict": verdict,
            "malicious_probability": round(mal_prob, 4),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/explain")
def explain(payload: DoHFeatures):
    if explainer is None:
        raise HTTPException(status_code=503, detail="Explainer not ready")
    
    x = np.array(payload.features, dtype=np.float32)
    explanation = explainer.explain(x)
    return explanation
