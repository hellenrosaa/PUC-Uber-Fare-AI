import pandas as pd
import numpy as np
import joblib
import os
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from haversine import haversine
from datetime import datetime
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder='../templates', static_folder='../static')

current_dir = os.path.dirname(os.path.abspath(__file__))

# Caminhos dos arquivos
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

DATA_PATH = os.path.join(current_dir, '..', 'data_final', 'data.csv')

# Verifica se os arquivos existem
for category in MODEL_PATHS:
    if not os.path.exists(MODEL_PATHS[category]) or not os.path.exists(SCALER_PATHS[category]):
        print(f"Verifique se os arquivos '{MODEL_PATHS[category]}' e '{SCALER_PATHS[category]}' existem nos diretórios corretos.")
        exit()
if not os.path.exists(DATA_PATH):
    print("Verifique se o arquivo 'data.csv' existe no diretório correto.")
    exit()

# Carregar o dataset
df_uber = pd.read_csv(DATA_PATH)

# Calcular o tempo gasto (em segundos)
df_uber['travel_time_seconds'] = (pd.to_datetime(df_uber['Updated']) - pd.to_datetime(df_uber['Create'])).dt.total_seconds()

# Calcular velocidade média (km/h)
df_uber['speed_kmh'] = (df_uber['distance_km'] / (df_uber['travel_time_seconds'] / 3600)).round(2)

# Criar indicador de trânsito
conditions = [
    df_uber['speed_kmh'] > 40,  # Trânsito leve
    (df_uber['speed_kmh'] <= 40) & (df_uber['speed_kmh'] >= 20),  # Trânsito moderado
    df_uber['speed_kmh'] < 20  # Trânsito pesado
]
choices = [0, 1, 2]  # Leve, Moderado, Pesado
df_uber['traffic_indicator'] = np.select(conditions, choices, default=-1)

# Função para geocodificar um endereço
def geocodificar_endereco(endereco):
    geolocator = Nominatim(user_agent="chatbot_taxi")
    location = geolocator.geocode(endereco)
    if location:
        return (location.latitude, location.longitude)
    else:
        raise ValueError(f"Endereço não encontrado: {endereco}")

# Função para calcular distância haversine
def calcular_distancia_km(origem, destino):
    return haversine(origem, destino)

# Função para identificar se está em horário de pico
def calcular_distance_pico(dist_km, hora):
    if 7 <= hora <= 9 or 17 <= hora <= 19:
        return dist_km
    return 0

# Função para calcular effective_distance
def calcular_effective_distance(dist_km, traffic_indicator):
    return dist_km * (1 + (traffic_indicator / 10))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prever_preco', methods=['POST'])
def prever_preco():
    data = request.json
    origem = data.get('origem')
    destino = data.get('destino')
    categoria = data.get('categoria').strip().lower()

    if categoria not in ["uberx", "comfort", "black"]:
        return jsonify({"error": "Categoria inválida."}), 400

    try:
        latlong_origem = geocodificar_endereco(origem)
        latlong_destino = geocodificar_endereco(destino)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


    # Simular horário atual
    agora = datetime.now()
    hora = agora.hour

    # Calcular features
    distance_km = calcular_distancia_km(latlong_origem, latlong_destino)
    distance_pico = calcular_distance_pico(distance_km, hora)
    effective_distance = calcular_effective_distance(distance_km, df_uber['traffic_indicator'].mean())

    # Formar vetor de entrada
    X = np.array([[distance_km, distance_pico, effective_distance]])

    # Carregar scaler e modelo específicos para a categoria
    modelo = joblib.load(MODEL_PATHS[categoria])
    scaler = joblib.load(SCALER_PATHS[categoria])

    X_scaled = scaler.transform(X)

    preco_previsto = modelo.predict(X_scaled)[0]

    
    return jsonify({
        "origem": origem,
        "destino": destino,
        "categoria": categoria.title(),
        "hora": hora,
        "distancia_km": distance_km,
        "preco_estimado": preco_previsto
    })

if __name__ == "__main__":
    app.run(debug=True)