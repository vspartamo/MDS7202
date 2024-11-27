from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np


model_path = "./Lab_12/models/best_model.pkl"

with open(model_path, "rb") as file:
    best_model = pickle.load(file)
app = FastAPI()


class WaterSample(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float


# Get
@app.get("/")
def home():
    return {
        "description": "Este modelo predice la potabilidad del agua con base en parámetros físico-químicos.",
        "problem": "Determinar si el agua es potable o no.",
        "input": {
            "ph": "float",
            "Hardness": "float",
            "Solids": "float",
            "Chloramines": "float",
            "Sulfate": "float",
            "Conductivity": "float",
            "Organic_carbon": "float",
            "Trihalomethanes": "float",
            "Turbidity": "float",
        },
        "output": {"potabilidad": "int (0: no potable, 1: potable)"},
    }


# Post
@app.post("/potabilidad/")
def predict_potabilidad(sample: WaterSample):
    features = np.array(
        [
            [
                sample.ph,
                sample.Hardness,
                sample.Solids,
                sample.Chloramines,
                sample.Sulfate,
                sample.Conductivity,
                sample.Organic_carbon,
                sample.Trihalomethanes,
                sample.Turbidity,
            ]
        ]
    )
    prediction = best_model.predict(features)[0]

    return {"potabilidad": int(prediction)}
