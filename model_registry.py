# model_registry.py
from model import CandleModel

MODELS = {
    "1": CandleModel("1"),
    "5": CandleModel("5"),
}

def get_model(tf: str):
    if tf in MODELS:
        return MODELS[tf]
    return MODELS["1"]  # Fallback на 1m