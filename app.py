from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load files
model = joblib.load("logistic_regression_scaled_model.joblib")
scaler = joblib.load("scaler.joblib")

@app.get("/")
def home():
    return {"message": "API is running 🚀"}

@app.post("/predict")
def predict(data: list):
    try:
        data = np.array(data).reshape(1, -1)
        data = scaler.transform(data)
        prediction = model.predict(data)

        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}