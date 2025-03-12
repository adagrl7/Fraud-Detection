from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load pre-trained models and scaler
try:
    rf_model = joblib.load("models/random_forest_model.pkl")
    xgb_model = joblib.load("models/xgboost_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    raise RuntimeError("Error loading models or scaler: " + str(e))

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "message": "Enter transaction data to predict fraud."})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, features: str = Form(...)):
    try:
        # Convert comma-separated string into a list of floats
        features_list = [float(f.strip()) for f in features.split(",")]
        data = np.array(features_list).reshape(1, -1)
        data_scaled = scaler.transform(data)
        
        rf_pred = int(rf_model.predict(data_scaled)[0])
        xgb_pred = int(xgb_model.predict(data_scaled)[0])
        
        prediction = {"rf": rf_pred, "xgb": xgb_pred}
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": "Enter transaction data to predict fraud.",
            "prediction": prediction
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
