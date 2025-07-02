# app.py
from flask import Flask, request, render_template, jsonify, session
from flask_cors import CORS
import logging
from datetime import datetime
# from urllib.parse import quote # Ya no se usa directamente aquí, se movió a utils.py

from config import SECRET_KEY, OPENROUTER_API_KEY
from models import detectar_toxicidad
from utils import generar_respuesta_contextual # SOLO esta función se importa de utils
from rag_embedder import crear_embeddings
from data_manager import load_knowledge_base, load_tramites_urls

app = Flask(__name__)
app.secret_key = SECRET_KEY
CORS(app)

# Configura logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

with app.app_context():
    logger.info("Cargando URLs de trámites desde JSON...")
    tramites_urls = load_tramites_urls()
    logger.info(f"{len(tramites_urls)} URLs de trámites cargadas.")

    logger.info("Cargando base de conocimiento desde JSON...")
    knowledge_base = load_knowledge_base()
    logger.info(f"Base de conocimiento con {len(knowledge_base)} entradas cargada.")

    logger.info("Creando/actualizando embeddings...")
    crear_embeddings() # Asegura que los embeddings existan y estén actualizados
    logger.info("Embeddings actualizados.")


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    if not request.is_json:
        return jsonify({"respuesta": "Formato inválido, se esperaba JSON.", "error": True}), 400

    data = request.get_json()
    mensaje = data.get('mensaje', '').strip()

    if not mensaje:
        return jsonify({"respuesta": "Por favor, escribe tu consulta.", "error": True}), 400

    es_toxico, razon_toxicidad = detectar_toxicidad(mensaje)
    if es_toxico:
        logger.warning(f"Mensaje tóxico detectado: '{mensaje}' - {razon_toxicidad}")
        return jsonify({"respuesta": "Usá un lenguaje respetuoso, por favor. Estoy para ayudarte.", "tipo": "toxicidad"}), 200

    historial_conversacion = session.get('historial', [])
    current_tramite_data = session.get('current_tramite_data', None)

    respuesta_generada = generar_respuesta_contextual(
        mensaje,
        historial_conversacion,
        current_tramite_data=current_tramite_data
    )
    
    if not respuesta_generada:
        logger.error("❌ La función generar_respuesta_contextual devolvió None.")
        return jsonify({
            "respuesta": "Ocurrió un error inesperado en el servidor. Por favor, intentá más tarde.",
            "tipo": "error_interno",
            "sugerencias": []
        }), 500

    if respuesta_generada.get('error'):
        logger.error(f"Error detectado en la respuesta de utils: {respuesta_generada.get('mensaje', 'Error desconocido')}")
        return jsonify({
            "respuesta": respuesta_generada.get('mensaje', "Ocurrió un error inesperado al procesar tu solicitud."),
            "tipo": respuesta_generada.get('tipo', 'error_interno'),
            "sugerencias": respuesta_generada.get('sugerencias', [])
        }), 500 # Devolver 500 para errores del servidor/IA

    datos_identificado = respuesta_generada.get('datos_tramite_identificado')
    if datos_identificado is not None:
        session['current_tramite_data'] = datos_identificado 
    else:
        logger.error("current_tramite_data es None a pesar de necesitar selección de ubicación.")

    response_data_to_send = {
        "respuesta": respuesta_generada.get('mensaje', 'No se pudo obtener una respuesta.'),
        "tipo": respuesta_generada.get('tipo'),
        "sugerencias": respuesta_generada.get('sugerencias', []),
        "necesita_seleccion": respuesta_generada.get('necesita_seleccion', False),
        "opciones_ubicacion": respuesta_generada.get('opciones_ubicacion', [])
    }
    
    if respuesta_generada.get('tipo') == 'tramite_especifico':
        response_data_to_send['info'] = respuesta_generada.get('info')
    session['historial'] = session.get('historial', [])
    session['historial'].append({
        'usuario': mensaje,
        'asistente': response_data_to_send['respuesta'],
        'timestamp': datetime.now().isoformat()
    })
    session['historial'] = session['historial'][-10:] # Limitar historial

    return jsonify(response_data_to_send), 200


@app.route('/api/limpiar_historial', methods=['POST'])
def limpiar_historial():
    session.pop('historial', None)
    session.pop('current_tramite_data', None) 
    logger.info("Historial limpiado")
    return jsonify({"mensaje": "Historial de conversación eliminado."})

if __name__ == '__main__':
    if not OPENROUTER_API_KEY:
        print("⚠️ ADVERTENCIA: No configuraste OPENROUTER_API_KEY en .env")
    else:
        print("✅ OPENROUTER_API_KEY configurada correctamente.")

    print("🚀 Iniciando servidor Flask en http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)