# chatbot.py (salvo em notebooks/)
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from geopy.geocoders import Nominatim
from haversine import haversine
import openai
from dotenv import load_dotenv
import time
from geopy.exc import GeocoderTimedOut

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

current_dir = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    'uberx': os.path.join(current_dir, 'model', 'modelo_uberx.joblib'),
    'comfort': os.path.join(current_dir, 'model', 'modelo_comfort.joblib'),
    'black': os.path.join(current_dir, 'model', 'modelo_black.joblib')
}
SCALER_PATHS = {
    'uberx': os.path.join(current_dir, 'model', 'scaler_uberx.joblib'),
    'comfort': os.path.join(current_dir, 'model', 'scaler_comfort.joblib'),
    'black': os.path.join(current_dir, 'model', 'scaler_black.joblib')
}
#DATA_PATH = os.path.join(current_dir, '..', 'data_final', 'data.csv')
DATA_PATH = os.path.join(current_dir, '..', 'data_sample', 'data_sample.csv')

try:
    df_uber = pd.read_csv(DATA_PATH)
except Exception as e:
    print("Erro ao carregar CSV:", e)
    df_uber = None

df_uber['travel_time_seconds'] = (pd.to_datetime(df_uber['Updated']) - pd.to_datetime(df_uber['Create'])).dt.total_seconds()
df_uber['speed_kmh'] = (df_uber['distance_km'] / (df_uber['travel_time_seconds'] / 3600)).round(2)

conditions = [
    df_uber['speed_kmh'] > 40,
    (df_uber['speed_kmh'] <= 40) & (df_uber['speed_kmh'] >= 20),
    df_uber['speed_kmh'] < 20
]
choices = [0, 1, 2]
df_uber['traffic_indicator'] = np.select(conditions, choices, default=-1)

def gerar_resposta_llm(pergunta):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você é um assistente inteligente que responde dúvidas sobre preços de corridas da Uber."},
                {"role": "user", "content": pergunta}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Ocorreu um erro: {e}"

def interpretar_mensagem(texto):
    prompt = f"""
Extraia origem, destino e categoria de Uber da seguinte frase:
"{texto}"
Retorne no formato JSON com as chaves: origem, destino, categoria. Exemplo:
{{"origem": "Av. Paulista", "destino": "Pinheiros", "categoria": "black"}}
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    content = response.choices[0].message['content']
    try:
        data = eval(content)
    except:
        data = {"error": "Erro ao interpretar mensagem."}
    return data

def geocodificar_endereco(endereco, tentativas=3):
    geolocator = Nominatim(user_agent="chatbot_taxi", timeout=10)
    for i in range(tentativas):
        try:
            location = geolocator.geocode(endereco)
            if location:
                return (location.latitude, location.longitude)
            else:
                raise ValueError(f"Endereço não encontrado: {endereco}")
        except GeocoderTimedOut:
            if i < tentativas - 1:
                time.sleep(1)  # espera 1 segundo e tenta de novo
                continue
            else:
                raise

def calcular_distancia_km(origem, destino):
    return haversine(origem, destino)

def calcular_distance_pico(dist_km, hora):
    return dist_km if 7 <= hora <= 9 or 17 <= hora <= 19 else 0

def calcular_effective_distance(dist_km, traffic_indicator):
    return dist_km * (1 + (traffic_indicator / 10))

def prever_preco(origem, destino, categoria):
    latlong_origem = geocodificar_endereco(origem)
    latlong_destino = geocodificar_endereco(destino)

    agora = datetime.now()
    hora = agora.hour

    distance_km = calcular_distancia_km(latlong_origem, latlong_destino)
    distance_pico = calcular_distance_pico(distance_km, hora)
    effective_distance = calcular_effective_distance(distance_km, df_uber['traffic_indicator'].mean())

    X = np.array([[distance_km, distance_pico, effective_distance]])
    modelo = joblib.load(MODEL_PATHS[categoria])
    scaler = joblib.load(SCALER_PATHS[categoria])

    X_scaled = scaler.transform(X)
    preco_previsto = modelo.predict(X_scaled)[0]

    return preco_previsto, distance_km