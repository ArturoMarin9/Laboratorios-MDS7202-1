from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import xgboost as xgb

# Definir la aplicación FastAPI
app = FastAPI()

# Definir la estructura del body para la solicitud POST
class WaterQuality(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

# Cargar el modelo entrenado
model_path = "models/best_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Definir la ruta raíz "/"
@app.get("/")
def home():
    return {"message": "Bienvenido a la API de predicción de potabilidad de agua.",
            "model_description": "Este modelo predice si una muestra de agua es potable basado en diferentes características físico-químicas.",
            "input": "Enviar una solicitud POST a /potabilidad/ con los parámetros de la muestra de agua.",
            "output": "Respuesta con la predicción de potabilidad (0 o 1)."}

# Definir la ruta POST "/potabilidad/"
@app.post("/potabilidad/")
def predict_potability(data: WaterQuality):
    # Convertir los datos del cuerpo de la solicitud a DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Realizar la predicción
    prediction = model.predict(input_data)[0]

    # Retornar la respuesta
    return {"potabilidad": int(prediction)}

# Punto de entrada para ejecutar la aplicación con uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
