from flask import Flask, render_template, request, jsonify
from notebooks.chatbot import gerar_resposta_llm, interpretar_mensagem, prever_preco
from datetime import datetime

app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prever_preco', methods=['POST'])
def prever_preco_route():
    dados = request.get_json()
    origem = dados.get('origem')
    destino = dados.get('destino')
    categoria = dados.get('categoria')

    if not origem or not destino or not categoria:
        return jsonify({'error': 'Origem, destino e categoria são obrigatórios.'}), 400

    try:
        preco_estimado, distancia_km = prever_preco(origem, destino, categoria)
        hora = datetime.now().hour

        return jsonify({
            'origem': origem,
            'destino': destino,
            'categoria': categoria,
            'preco_estimado': preco_estimado,
            'distancia_km': distancia_km,
            'hora': hora
        })
    except Exception as e:
        return jsonify({'error': f"Ocorreu um erro: {str(e)}"}), 500

@app.route('/responder', methods=['POST'])
def responder():
    # Você pode manter o código que já tinha aqui se quiser
    return jsonify({'resposta': 'Rota responder não implementada nesta versão.'})

if __name__ == "__main__":
    app.run(debug=True)

