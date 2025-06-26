from flask import Flask, request, render_template, jsonify, session
from flask_cors import CORS
import logging
from datetime import datetime
from urllib.parse import quote

from config import SECRET_KEY, OPENROUTER_API_KEY
from models import detectar_toxicidad
from utils import generar_respuesta_contextual, llamar_ia_openrouter, buscar_tramites_inteligente # Import buscar_tramites_inteligente
from rag_system import build_knowledge_base_embeddings
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

    logger.info("Cargando knowledge base desde JSON...")
    knowledge_base = load_knowledge_base()
    logger.info(f"Knowledge base con {len(knowledge_base)} entradas cargada.")

    logger.info("Generando embeddings de knowledge base...")
    crear_embeddings()
    logger.info("Embeddings generados.")




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
        return jsonify({"respuesta": "Usá un lenguaje respetuoso, por favor. Estoy para ayudarte.", "error": True}), 400

    historial = session.get('historial', [])
    current_tramite_data = session.get('current_tramite_data', None)

    response_data = None

    if "seleccion_ubicacion" in session and mensaje.isdigit():
        try:
            selected_index = int(mensaje) - 1
            opciones = session["seleccion_ubicacion"]["opciones"]
            if 0 <= selected_index < len(opciones):
                selected_location = opciones[selected_index]
                categoria_id = session["seleccion_ubicacion"]["categoria"]
                original_datos_tramite = session["seleccion_ubicacion"].get("original_datos_tramite", {})

                datos_tramite_con_ubicacion = original_datos_tramite.copy()
                datos_tramite_con_ubicacion.update({
                    'direccion': selected_location.get('direccion', ''),
                    'horarios': selected_location.get('horarios', ''),
                    'telefono': selected_location.get('telefono', ''),
                    'email': selected_location.get('email', ''),
                    'responsable': selected_location.get('responsable', ''),
                    'necesita_seleccion': False
                })

                if datos_tramite_con_ubicacion['direccion']:
                    dir_comp = datos_tramite_con_ubicacion['direccion']
                    if "formosa" not in dir_comp.lower():
                        dir_comp += ", Formosa"
                    encoded_address = quote(dir_comp)
                    datos_tramite_con_ubicacion['mapa_url'] = f"http://google.com/maps?q={encoded_address}" # Corrected Google Maps URL

                session.pop("seleccion_ubicacion", None)
                session['current_tramite_data'] = datos_tramite_con_ubicacion # Update current_tramite_data after location selection

                from utils import _generar_respuesta_con_datos
                response_data = _generar_respuesta_con_datos(datos_tramite_con_ubicacion, mensaje, categoria_id)
                respuesta_ia = llamar_ia_openrouter(mensaje, historial, context_override=response_data).get('respuesta', response_data['mensaje'])

                respuesta = {
                    "respuesta": respuesta_ia,
                    "datos_estructurados": response_data,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                respuesta = {
                    "respuesta": "Número de opción inválido. Elegí un número correcto de la lista.",
                    "error": True
                }
        except Exception as e:
            logger.error(f"Error procesando selección de ubicación: {e}")
            # Fallback to general AI call if error in selection processing
            respuesta = llamar_ia_openrouter(mensaje, historial)
            if respuesta.get('datos_estructurados'):
                session['current_tramite_data'] = respuesta['datos_estructurados'].get('info', None) or respuesta['datos_estructurados'].get('datos_tramite_identificado', None)

    else:
        use_current_tramite_context = False
        if current_tramite_data:
            specific_keywords = ["requisitos", "costo", "formularios", "ubicacion", "horarios", "pasos", "observaciones", "dónde", "cuánto", "descargar"]
            if any(k in mensaje.lower() for k in specific_keywords):
                use_current_tramite_context = True
            else:
                pass 

        if use_current_tramite_context and current_tramite_data:
            logger.info(f"Passing current_tramite_data as context_override: {current_tramite_data.get('titulo')}")
            respuesta = llamar_ia_openrouter(mensaje, historial, context_override={"tipo": "tramite_especifico", "info": current_tramite_data})
        else:
            response_data_from_gen = generar_respuesta_contextual(mensaje, historial)
            
            if response_data_from_gen.get('tipo') == 'no_encontrado':
                respuesta = llamar_ia_openrouter(mensaje, historial) 
                session.pop('current_tramite_data', None) 
            else:
                respuesta = llamar_ia_openrouter(mensaje, historial, context_override=response_data_from_gen)
                if response_data_from_gen.get('tipo') == 'seleccion_ubicacion':
                    session['current_tramite_data'] = response_data_from_gen.get('original_datos_tramite', {}) or response_data_from_gen.get('datos_tramite_identificado', {})
                else: 
                    session['current_tramite_data'] = response_data_from_gen.get('info', {}) or response_data_from_gen.get('datos_tramite_identificado', {})

        if respuesta.get('datos_estructurados') and respuesta['datos_estructurados'].get('tipo') == 'seleccion_ubicacion':
            session["seleccion_ubicacion"] = {
                "categoria": respuesta['datos_estructurados']['categoria'],
                "opciones": respuesta['datos_estructurados']['opciones_ubicacion'],
                "original_datos_tramite": respuesta['datos_estructurados'].get('original_datos_tramite', {})
            }
            session['current_tramite_data'] = respuesta['datos_estructurados'].get('datos_tramite_identificado', {}) or respuesta['datos_estructurados'].get('original_datos_tramite', {})

    if respuesta.get('error'):
        return jsonify(respuesta), 500

    # Guardar historial en sesión
    if 'historial' not in session:
        session['historial'] = []

    session['historial'].append({
        'usuario': mensaje,
        'asistente': respuesta.get('respuesta', ''),
        'datos_estructurados': respuesta.get('datos_estructurados'), 
        'timestamp': datetime.now().isoformat()
    })
    session['historial'] = session['historial'][-10:]  # Limitar historial

    return jsonify(respuesta)


@app.route('/api/limpiar_historial', methods=['POST'])
def limpiar_historial():
    session.pop('historial', None)
    session.pop('seleccion_ubicacion', None)
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