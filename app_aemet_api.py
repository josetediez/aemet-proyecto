from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import psycopg2
import boto3
import joblib
import io
from datetime import timedelta
import os

# -----------------------------
# Configuración DB Aurora
# -----------------------------
DB_HOST = os.environ.get("DB_HOST", "datosaemet.c16uosue6hjy.eu-north-1.rds.amazonaws.com")
DB_PORT = int(os.environ.get("DB_PORT", 5432))
DB_NAME = os.environ.get("DB_NAME", "datosaemet")
DB_USER = os.environ.get("DB_USER", "posgrest")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "MA3696dd")

# -----------------------------
# Configuración S3 modelos
# -----------------------------
BUCKET_MODELOS = "modelos-forecasting"
MODEL_TMAX_KEY = "model_tmax.pkl"
MODEL_TMIN_KEY = "model_tmin.pkl"

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="API Forecast Aemet")

# -----------------------------
# Requests
# -----------------------------
class ForecastRequest(BaseModel):
    ubicacion: str
    dias: int = 7

# -----------------------------
# Conexión Aurora
# -----------------------------
def get_conn():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

# -----------------------------
# Cargar modelos desde S3
# -----------------------------
s3 = boto3.client("s3")
modelo_max = None
modelo_min = None

def load_models():
    global modelo_max, modelo_min
    if modelo_max is not None and modelo_min is not None:
        return

    obj_max = s3.get_object(Bucket=BUCKET_MODELOS, Key=MODEL_TMAX_KEY)
    modelo_max = joblib.load(io.BytesIO(obj_max["Body"].read()))

    obj_min = s3.get_object(Bucket=BUCKET_MODELOS, Key=MODEL_TMIN_KEY)
    modelo_min = joblib.load(io.BytesIO(obj_min["Body"].read()))

# -----------------------------
# Endpoint principal
# -----------------------------
@app.get("/")
def estado():
    return {"message": "API Aemet Forecast ML activa"}

# -----------------------------
# Endpoint forecast
# -----------------------------
@app.post("/forecast")
def forecast(req: ForecastRequest):
    load_models()
    conn = get_conn()

    # Traemos histórico suficiente (mínimo 7 días)
    query = """
        SELECT
            "Fecha",
            "Temperatura Maxima",
            "Temperatura Minima",
            "Humedad Relativa",
            "Presión atmosférica Mínima",
            "Presión atmosférica Máxima",
            "Dirección media del aire"
        FROM meteo
        WHERE "Ubicación de la estación" = %s
        ORDER BY "Fecha" ASC
    """

    df = pd.read_sql(query, conn, params=(req.ubicacion,))
    conn.close()

    if len(df) < 8:
        return {"error": "No hay suficientes datos históricos"}

    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df.sort_values("Fecha").reset_index(drop=True)

    # Calcular lags
    df["tmax_lag1"] = df["Temperatura Maxima"].shift(1)
    df["tmax_lag7"] = df["Temperatura Maxima"].shift(7)
    df["tmin_lag1"] = df["Temperatura Minima"].shift(1)
    df["tmin_lag7"] = df["Temperatura Minima"].shift(7)

    df = df.dropna().reset_index(drop=True)
    last = df.iloc[-1].copy()
    fecha_actual = last["Fecha"]

    # Buffers para lags
    tmax_hist = list(df["Temperatura Maxima"].iloc[-7:])
    tmin_hist = list(df["Temperatura Minima"].iloc[-7:])

    features = [
        "dia", "mes", "dia_anyo",
        "Humedad Relativa",
        "Presión atmosférica Mínima",
        "Presión atmosférica Máxima",
        "Dirección media del aire",
        "tmax_lag1", "tmax_lag7",
        "tmin_lag1", "tmin_lag7"
    ]

    preds = []

    for i in range(req.dias):
        fecha_pred = fecha_actual + timedelta(days=1)
        fila = {
            "dia": fecha_pred.day,
            "mes": fecha_pred.month,
            "dia_anyo": fecha_pred.timetuple().tm_yday,
            "Humedad Relativa": last["Humedad Relativa"],
            "Presión atmosférica Mínima": last["Presión atmosférica Mínima"],
            "Presión atmosférica Máxima": last["Presión atmosférica Máxima"],
            "Dirección media del aire": last["Dirección media del aire"],
            "tmax_lag1": tmax_hist[-1],
            "tmax_lag7": tmax_hist[0],
            "tmin_lag1": tmin_hist[-1],
            "tmin_lag7": tmin_hist[0],
        }

        X_future = pd.DataFrame([fila])[features]

        tmax_pred = modelo_max.predict(X_future)[0]
        tmin_pred = modelo_min.predict(X_future)[0]

        preds.append({
            "Fecha": fecha_pred.strftime("%Y-%m-%d"),
            "Temp_Max_Pred": round(float(tmax_pred), 2),
            "Temp_Min_Pred": round(float(tmin_pred), 2)
        })

        # Actualizar lags
        tmax_hist.append(tmax_pred)
        tmin_hist.append(tmin_pred)
        tmax_hist.pop(0)
        tmin_hist.pop(0)

        fecha_actual = fecha_pred

    return {
        "ubicacion": req.ubicacion,
        "predicciones": preds
    }


