# app.py
from flask import Flask, request, render_template, jsonify, session
from flask_cors import CORS
import logging
from datetime import datetime
from urllib.parse import quote

from config import SECRET_KEY, OPENROUTER_API_KEY
from models import detectar_toxicidad
# Asegúrate de importar _generar_respuesta_con_datos directamente de utils
from utils import generar_respuesta_contextual, llamar_ia_openrouter, _generar_respuesta_con_datos 
from rag_system import build_knowledge_base_embeddings # No se usa directamente en app.py, pero puede permanecer
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
    crear_embeddings() # Asegúrate de que esto crea los embeddings si no existen
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

    respuesta_para_usuario = None # Esta será la respuesta final que se envía al front-end

    # =========================================================================
    # Lógica para manejo de selección de ubicación
    # =========================================================================
    if "seleccion_ubicacion" in session and mensaje.isdigit():
        try:
            selected_index = int(mensaje) - 1
            opciones = session["seleccion_ubicacion"]["opciones"]
            if 0 <= selected_index < len(opciones):
                selected_location = opciones[selected_index]
                categoria_id = session["seleccion_ubicacion"]["categoria"]
                original_datos_tramite = session["seleccion_ubicacion"].get("original_datos_tramite", {})

                # Clonar y actualizar los datos del trámite con la ubicación seleccionada
                datos_tramite_con_ubicacion = original_datos_tramite.copy()
                datos_tramite_con_ubicacion.update({
                    'direccion': selected_location.get('direccion', ''),
                    'horarios': selected_location.get('horarios', ''),
                    'telefono': selected_location.get('telefono', ''),
                    'email': selected_location.get('email', ''),
                    'responsable': selected_location.get('responsable', ''),
                    'necesita_seleccion': False # Ya se seleccionó
                })

                # Generar URL de mapa si hay dirección
                if datos_tramite_con_ubicacion['direccion']:
                    dir_comp = datos_tramite_con_ubicacion['direccion']
                    # Asegurarse de que "Formosa" se añade si no está presente
                    if "formosa" not in dir_comp.lower():
                        dir_comp += ", Formosa"
                    encoded_address = quote(dir_comp)
                    datos_tramite_con_ubicacion['mapa_url'] = f"http://google.com/maps?q={encoded_address}"

                # Limpiar la sesión de selección de ubicación
                session.pop("seleccion_ubicacion", None)
                # Actualizar el trámite actual en sesión con la ubicación ya elegida
                session['current_tramite_data'] = datos_tramite_con_ubicacion

                # AHORA AQUÍ: Usar _generar_respuesta_con_datos directamente
                # Esto es crucial: ya tenemos la información y el formato.
                # NO LLAMAR A llamar_ia_openrouter AQUÍ.
                respuesta_pre_formateada = _generar_respuesta_con_datos(datos_tramite_con_ubicacion, mensaje, categoria_id)
                
                respuesta_para_usuario = {
                    "respuesta": respuesta_pre_formateada['mensaje'],
                    "datos_estructurados": respuesta_pre_formateada, # Incluye todos los datos para depuración
                    "timestamp": datetime.now().isoformat()
                }
                logger.info(f"Respuesta generada por selección de ubicación: {respuesta_para_usuario['respuesta']}")

            else:
                respuesta_para_usuario = {
                    "respuesta": "Número de opción inválido. Por favor, elegí un número correcto de la lista.",
                    "error": True
                }
        except Exception as e:
            logger.error(f"Error procesando selección de ubicación: {e}")
            # Si hay un error en la selección, se recurre al LLM para una respuesta general
            respuesta_para_usuario = llamar_ia_openrouter(mensaje, historial)
            # Si el LLM devuelve datos estructurados, actualizar current_tramite_data
            if respuesta_para_usuario.get('datos_estructurados'):
                session['current_tramite_data'] = respuesta_para_usuario['datos_estructurados'].get('info', None) or \
                                                  respuesta_para_usuario['datos_estructurados'].get('datos_tramite_identificado', None)

    # =========================================================================
    # Lógica principal de RAG y LLM (cuando no hay selección de ubicación en curso)
    # =========================================================================
    else:
        # Primero, intentar obtener una respuesta formateada por nuestra lógica RAG
        response_data_from_gen = generar_respuesta_contextual(mensaje, historial, current_tramite_data)

        if response_data_from_gen.get('tipo') == 'no_encontrado' or response_data_from_gen.get('tipo') == 'error':
            # Si la RAG no encontró un trámite o hubo un error en la búsqueda RAG,
            # entonces y SOLO ENTONCES, llama al LLM para una respuesta general/fallback.
            logger.info("RAG no encontró un trámite o hubo un error. Llamando a LLM para respuesta general.")
            respuesta_para_usuario = llamar_ia_openrouter(mensaje, historial)
            session.pop('current_tramite_data', None) # Limpiar contexto si el LLM no encontró nada
        else:
            # Si la RAG SÍ encontró un trámite y generó una respuesta formateada,
            # usa esa respuesta directamente. NO LLAMAR AL LLM AQUÍ.
            logger.info(f"RAG encontró trámite y generó respuesta: {response_data_from_gen.get('tipo')}")
            respuesta_para_usuario = {
                "respuesta": response_data_from_gen['mensaje'],
                "datos_estructurados": response_data_from_gen, # Pasamos la estructura completa
                "timestamp": datetime.now().isoformat()
            }
            # Actualizar current_tramite_data con la información del trámite identificado
            session['current_tramite_data'] = response_data_from_gen.get('info', {}) or \
                                              response_data_from_gen.get('datos_tramite_identificado', {})

        # Si la respuesta de la RAG indica que se necesita selección de ubicación, guardar en sesión
        if respuesta_para_usuario.get('datos_estructurados') and \
           respuesta_para_usuario['datos_estructurados'].get('tipo') == 'seleccion_ubicacion':
            session["seleccion_ubicacion"] = {
                "categoria": respuesta_para_usuario['datos_estructurados']['categoria'],
                "opciones": respuesta_para_usuario['datos_estructurados']['opciones_ubicacion'],
                "original_datos_tramite": respuesta_para_usuario['datos_estructurados'].get('original_datos_tramite', {})
            }
            # current_tramite_data ya se actualizó arriba con el trámite original

    # Manejo de errores final
    if respuesta_para_usuario.get('error'):
        return jsonify(respuesta_para_usuario), 500

    # Guardar historial en sesión
    if 'historial' not in session:
        session['historial'] = []

    session['historial'].append({
        'usuario': mensaje,
        'asistente': respuesta_para_usuario.get('respuesta', ''),
        'datos_estructurados': respuesta_para_usuario.get('datos_estructurados'),
        'timestamp': datetime.now().isoformat()
    })
    session['historial'] = session['historial'][-10:]  # Limitar historial

    return jsonify(respuesta_para_usuario)


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