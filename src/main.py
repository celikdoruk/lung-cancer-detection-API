import schemas
import pandas as pd
import function

from fastapi import FastAPI, HTTPException

app = FastAPI()

FIELD_NAMES = [
    "age",
    "gender",
    "pack_years",
    "radon_exposure",
    "asbestos_exposure",
    "secondhand_smoke_exposure",
    "copd_diagnosis",
    "alcohol_consumption",
    "family_history",
]

@app.post("/", response_model=schemas.ModelOutput)
def index(data: schemas.Request):
    try:
        df = pd.DataFrame([data.model_dump()], columns=FIELD_NAMES)
        df = function.preprocess_data(df)
        df = function.normalize_data(df)

        y_pred = function.predict_for_data(df)
        return {"model_prediction": int(y_pred)}

    except Exception as e:
        raise HTTPException(400, f"An unexpected error occured: {e}")
