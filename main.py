from fastapi import FastAPI
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import requests

app = FastAPI(
    title="Motor de Inferencia IA - Portafolio",
    description="API predictiva de Bitcoin con Redes Neuronales LSTM"
)

# 1. CARGAMOS EL MODELO Y EL ESCALADOR AL INICIAR
try:
    modelo_lstm = tf.keras.models.load_model("modelo_bitcoin.h5")
    scaler = joblib.load("scaler_bitcoin.pkl")
    print("✅ Modelo y Scaler cargados exitosamente en memoria.")
except Exception as e:
    modelo_lstm = None
    scaler = None
    print(f"❌ Error al cargar los archivos: {e}")


@app.get("/")
def estado_servidor():
    return {"estado": "Servidor IA activo", "archivos_cargados": modelo_lstm is not None}


@app.get("/prediccion/bitcoin")
def obtener_prediccion():
    if modelo_lstm is None or scaler is None:
        return {"error": "Faltan los archivos .h5 o .pkl en el servidor."}

    try:
        # ---------------------------------------------------------
        # 1. OBTENER DATOS (Ejemplo con la API pública de Binance)
        # Necesitamos los últimos 60 días (o los que use tu modelo)
        # Usamos la api de USA al tener todo en servidor de alla
        # ---------------------------------------------------------
        url = "https://api.binance.us/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=60"
        respuesta = requests.get(url)
        datos_binance = respuesta.json()

        # Extraer solo el precio de cierre (índice 4 en la API de Binance)
        precios_cierre = [float(dia[4]) for dia in datos_binance]
        df = pd.DataFrame(precios_cierre, columns=['Close'])

        ultimo_precio_real = df['Close'].iloc[-1]

        # ---------------------------------------------------------
        # 2. PREPARAR DATOS (Usando TU scaler entrenado)
        # ---------------------------------------------------------
        # Aplicamos la transformación matemática original
        datos_escalados = scaler.transform(df)

        # Le damos la forma tridimensional que pide LSTM: (muestras, pasos_de_tiempo, características)
        # Cambia el 60 por la cantidad de días que usaste en tu entrenamiento (time_steps)
        X_pred = np.reshape(datos_escalados, (1, 60, 1))

        # ---------------------------------------------------------
        # 3. HACER LA PREDICCIÓN Y DESESCALAR
        # ---------------------------------------------------------
        prediccion_escalada = modelo_lstm.predict(X_pred)

        # Volvemos a convertir el valor de 0-1 a Dólares Reales
        precio_predicho = scaler.inverse_transform(prediccion_escalada)
        precio_final_usd = float(precio_predicho[0][0])

        return {
            "moneda": "BTC",
            "ultimoPrecioReal": float(ultimo_precio_real),
            "precioPredichoMañana": precio_final_usd,
            "tendencia": "alcista" if precio_final_usd > ultimo_precio_real else "bajista"
        }

    except Exception as e:
        return {"error": f"Fallo en el proceso de inferencia: {str(e)}"}