import pandas as pd
import joblib

model = joblib.load("joblib_models/xgbclassifier.joblib")
encoder = joblib.load("joblib_models/encoder.joblib")
scaler = joblib.load("joblib_models/scaler.joblib")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().str.lower()

    bool_cols = ["asbestos_exposure", "secondhand_smoke_exposure", "copd_diagnosis", "family_history"]
    for col in bool_cols:
        df[col] = df[col].replace({"no": 0, "yes": 1})
        if df[col].isin([0,1]).all() is False:
            raise ValueError(f"Invalid value in column {col}, expected yes or no.")
    
    df["gender"] = df["gender"].replace({"male": 1, "female": 0})
    if df["gender"].isin([0, 1]).all() is False:
        raise ValueError("Invalid value for gender. Expected 'male' or 'female'.")
    
    df["alcohol_consumption"] = df["alcohol_consumption"].replace({"moderate": 0, "heavy": 1})
    if df["alcohol_consumption"].isin([0, 1]).all() is False:
        raise ValueError("Invalid value for alcohol_consumption. Expected 'moderate' or 'heavy'.")


    return df

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    column_to_encode = ["radon_exposure"]
    df[encoder.get_feature_names_out()] = encoder.transform(df[column_to_encode])
    df = df.drop(columns=column_to_encode)

    df[scaler.get_feature_names_out()] = scaler.transform(df)

    return df

def predict_for_data(df: pd.DataFrame) -> int:
    y_pred_prob = model.predict_proba(df)
    y_pred = (y_pred_prob[: ,1] >= 0.45).astype(int)
    
    return y_pred

