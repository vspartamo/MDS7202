from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn
from optimize import optimize_model


optimize_model()

model_path = "./docs/models/best_model.pkl"
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
    """
    Home page

    Parameters:
    None

    Returns:
    dict: Information about the model
    """
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
    """
    Predicts the potability of water

    Parameters:
    sample (WaterSample): Water sample to predict

    Returns:
    dict: Prediction of potability
    """
    with open(model_path, "rb") as file:
        best_model = pickle.load(file)
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


if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=7888)
    except KeyboardInterrupt:
        print("Server stopped. Goodbye! :)")
