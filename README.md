# Lung Cancer Risk Prediction
 
A machine learning project that predicts lung cancer risk from patient demographic and exposure data, served through a FastAPI REST API.
 
## Overview
 
This project trains an XGBoost classifier on a lung cancer dataset (age, smoking history, environmental exposures, medical history) and exposes the trained model through a simple `/` POST endpoint that returns a binary prediction.
## Setup
 
1. Clone the repository and navigate into it.
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Run the API from the `src/` directory (the model paths in `function.py` are relative to `src/`, so the joblib models must be reachable at `joblib_models/` from wherever you launch uvicorn):
```bash
   cd src
   uvicorn main:app --reload
```
   The API will be available at `http://127.0.0.1:8000`.
 
## Usage
 
Send a POST request to `/` with patient data as JSON:
 
```bash
curl -X POST http://127.0.0.1:8000/ \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "gender": "male",
    "pack_years": 65,
    "radon_exposure": "high",
    "asbestos_exposure": "yes",
    "secondhand_smoke_exposure": "yes",
    "copd_diagnosis": "no",
    "alcohol_consumption": "heavy",
    "family_history": "yes"
  }'
```
 
Response:
```json
{
  "model_prediction": 1
}
```

`model_prediction` is `1` if the model predicts lung cancer, `0` otherwise (decision threshold: 0.45 on the positive-class probability).
 
A ready-to-run client example is provided in `src/request_example.py`.

