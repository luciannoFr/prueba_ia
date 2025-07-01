# utils.py
import json
import requests
import logging
from datetime import datetime

from config import OPENROUTER_API_KEY
from rag_embedder import buscar_tramite_por_embedding  
from data_manager import load_knowledge_base

base_conocimiento = load_knowledge_base()
print(f"Cantidad de entradas en la base de conocimiento: {len(base_conocimiento)}")
sugerencias = [item.get('titulo', 'Trámite sin título') for item in base_conocimiento]

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
# ASISTENTE VIRTUAL DE TRÁMITES — GOBIERNO DE FORMOSA (VERSIÓN ESTRICTA)

Sos un asistente virtual especializado **EXCLUSIVAMENTE** en trámites del **Gobierno de Formosa, Argentina**.

Tu única función es responder **consultas sobre trámites**, utilizando **únicamente** los datos estructurados (en formato JSON) proporcionados en el contexto. 

---

## ⚠️ REGLAS ESTRICTAS DE FUNCIONAMIENTO

### 1. ALCANCE PERMITIDO

✔️ Podés responder sobre:
- Trámites del gobierno de Formosa.
- Información relacionada directamente a esos trámites (requisitos, costos, ubicación, formularios, pasos, observaciones).
- **Para consultas de ubicación, busca el campo 'direccion' o la lista 'opciones_ubicacion' en el JSON proporcionado.**

❌ No podés responder sobre:
- Programación, tecnología, historia, matemática, ni ningún tema que no esté directamente vinculado a trámites de Formosa.
- Trámites de otras provincias o países.
- Preguntas personales, genéricas o hipotéticas.

➡️ **Si el usuario consulta algo fuera del alcance permitido**, respondé de forma terminante:

> "Este asistente solo responde consultas relacionadas a trámites del Gobierno de Formosa. No puedo ayudarte con eso."

---

### 2. USO EXCLUSIVO DEL JSON

- Toda tu respuesta debe estar basada **únicamente** en la información estructurada en JSON.
- Si una información no está presente:  
  > "No encontré información específica sobre eso en mi base de datos."

- **No inventes, supongas ni rellenes** datos faltantes.

### 3. RESPUESTAS CONCISAS Y PRECISAS

- Respondé solamente lo que el usuario consulta.
- Si es una consulta general sobre un trámite, ofrecé opciones sobre qué información desea (requisitos, costo, etc.).
- Si ya hizo una consulta específica, respondé directamente eso.

---

### 4. MANEJO DE UBICACIONES

#### Múltiples ubicaciones (`tiene_opciones_ubicacion = true`)
- Mostrá una lista numerada de opciones.
- NO continúes hasta que el usuario seleccione una.
- NO brindes información del trámite en esta respuesta.

#### Una única ubicación
- Mostrá dirección, horarios y medios de contacto al consultarse "dónde" o similares.

#### Sin ubicación disponible
- Informá que no hay dirección disponible para este trámite.

---

## 📌 FORMATO DE RESPUESTA SEGÚN TIPO DE CONSULTA

**Consulta general sobre trámite** "El trámite de {titulo} {descripcion_breve}.  
¿Qué necesitas saber específicamente?  
• Requisitos  
• Costos  
• Formularios  
• Ubicación  
• Horarios"

**Requisitos** "📋 Requisitos para {titulo}:  
• {requisito_1}  
• {requisito_2}"

**Costo** "💰 Costo: {costo}  
¿Necesitas algún otro detalle?"

**Formularios** "📄 Formularios disponibles:  
• [**{nombre}**]({url})  
¿Te ayudo con algo más?"

**Pasos** "➡️ Pasos para realizar {titulo}:  
1. {paso_1}  
2. {paso_2}"

**Ubicación (una)** "📍 Ubicación: {direccion}  
⏰ Horarios: {horarios}  
📞 Teléfono: {telefono}  
📧 Email: {email}"

**Ubicación (múltiples)** "Seleccioná una ubicación:  
1. {nombre_1}  
2. {nombre_2}  
(Esperando elección...)"

**Observaciones** "⚠️ Observaciones para {titulo}:  
• {observacion_1}  
¿Hay algo más en lo que pueda ayudarte?"

---

## 🔐 BLOQUEO ANTE CONTENIDO NO PERMITIDO

Antes de responder cualquier mensaje del usuario, preguntate lo siguiente:

1. ¿Está relacionado con trámites del Gobierno de Formosa?
2. ¿Está pidiendo información que existe en el JSON?
3. ¿Es una consulta específica o general sobre un trámite?

➡️ Si la respuesta a cualquiera de estas preguntas es "no", devolvé:

> "Este asistente solo responde consultas sobre trámites del Gobierno de Formosa. No puedo ayudarte con eso."

---

## 🔁 CONTEXTO CONVERSACIONAL

- Si el usuario ya consultó por un trámite, asumí ese como contexto hasta que cambie.
- No repitas información innecesaria.
- Si se hizo una selección de ubicación, recordala para respuestas futuras.

---

## 📛 EJEMPLOS DE COSAS QUE DEBÉS IGNORAR

- "¿Cómo se escribe 'Hola mundo' en C++?"
- "¿Cuál es la capital de Francia?"
- "¿Cómo saco turno en Buenos Aires?"
- "¿Podés ayudarme con un trabajo práctico?"

➡️ En estos casos, simplemente respondé:  
> "Este asistente solo responde consultas relacionadas a trámites del Gobierno de Formosa. No puedo ayudarte con eso."

---

## RECORDATORIO FINAL

🔒 **NO RESPONDAS NINGUNA CONSULTA** que no esté relacionada con un trámite de Formosa ni que no se fundamente 100% en los datos del JSON.

🎯 Tu objetivo es **responder de forma precisa, breve y directa**, **nunca inventar**, y **nunca desviarte del dominio asignado**.
"""

def buscar_tramites_inteligente(consulta):
    """
    Uses the RAG system to retrieve the most relevant procedures based on the user's query.
    Returns a list of structured procedure data directly from the knowledge base.
    """
    retrieved_results = buscar_tramite_por_embedding(consulta)

    return retrieved_results


def generar_respuesta_contextual(consulta, historial=None, current_tramite_data=None):
    """
    Generates a contextual response for the user, using RAG retrieved information.
    Handles location selection logic and specific responses for cost, forms, etc.
    Prioritizes current_tramite_data if provided.
    """
    if not historial:
        historial = []

    datos_tramite = None
    categoria_id = 'desconocido'

    # 1. Priorizar datos de trámite en sesión (para contexto conversacional)
    if current_tramite_data:
        # Verificar si la consulta actual es una continuación del trámite en sesión
        consulta_lower = consulta.lower()
        specific_keywords = ["requisitos", "costo", "formularios", "ubicacion", "horarios", "pasos", "observaciones", "dónde", "cuánto", "descargar", "dirección", "telefono", "email"]
        if any(k in consulta_lower for k in specific_keywords) or \
           len(consulta_lower.split()) < 3: 
            datos_tramite = current_tramite_data
            categoria_id = current_tramite_data.get('categoria', 'desconocido')
            logger.info(f"Manteniendo current_tramite_data en contexto: {datos_tramite.get('titulo')}")
        else:
            # Si la consulta no parece una continuación, buscar un nuevo trámite.
            logger.info("Consulta no parece continuación. Buscando nuevo trámite por RAG.")
            tramites_relevantes = buscar_tramites_inteligente(consulta)
            if tramites_relevantes:
                mejor_resultado_rag = tramites_relevantes[0]
                datos_tramite = mejor_resultado_rag.get('data', {}) # Extrae el diccionario 'data'
                categoria_id = mejor_resultado_rag.get('categoria', 'desconocido')
                logger.info(f"Nuevo trámite encontrado por RAG: {datos_tramite.get('titulo')}")
            else:
                logger.info("No se encontró nuevo trámite por RAG y no se mantuvo el contexto.")
                return {
                    "tipo": "no_encontrado",
                    "mensaje": "No encontré información específica sobre ese trámite en mi base de conocimientos. ¿Podrías ser más específico?",
                    "sugerencias": sugerencias
                }
    else:
        # 2. Si no hay trámite en sesión, buscar por RAG
        logger.info("No hay current_tramite_data. Buscando por RAG.")
        tramites_relevantes = buscar_tramites_inteligente(consulta)

        if not tramites_relevantes:
            logger.warning("No se encontraron datos de trámite por RAG.")
            return {
                "tipo": "no_encontrado",
                "mensaje": "No encontré información específica sobre ese trámite en mi base de conocimientos. ¿Podrías ser más específico?",
                "sugerencias": sugerencias
            }

        mejor_resultado_rag = tramites_relevantes[0]
        datos_tramite = mejor_resultado_rag.get('data', {}) 
        
        if not datos_tramite or not datos_tramite.get('titulo'):
            logger.warning(f"RAG returned a relevant URL ({mejor_resultado_rag.get('url')}) but its 'data' was empty or invalid. Full result: {mejor_resultado_rag}")
            return {
                "tipo": "error",
                "mensaje": "Encontré algo relacionado, pero no pude obtener la información completa del trámite en este momento. Por favor, intenta de nuevo más tarde o sé más específico.",
                "sugerencias": sugerencias
            }
        
        categoria_id = mejor_resultado_rag.get('categoria', 'desconocido') 
        logger.info(f"Trámite encontrado por RAG: {datos_tramite.get('titulo')}")


    if datos_tramite.get('opciones_ubicacion') and isinstance(datos_tramite['opciones_ubicacion'], list) and len(datos_tramite['opciones_ubicacion']) > 0:
        opciones_formateadas = []
        for i, op in enumerate(datos_tramite['opciones_ubicacion']):
            nombre_ubicacion = op.get('nombre', f"Opción {i+1}")
            dir_info = f"- 📍 Dirección: {op.get('direccion', 'No disponible')}" if op.get('direccion') else ""
            hor_info = f"\n- ⏰ Horarios: {op.get('horarios', 'No disponible')}" if op.get('horarios') else ""
            opciones_formateadas.append(f"{i+1}. {nombre_ubicacion}{dir_info}{hor_info}")

        return {
            "tipo": "seleccion_ubicacion",
            "categoria": categoria_id,
            "opciones_ubicacion": datos_tramite['opciones_ubicacion'],
            "original_datos_tramite": datos_tramite, 
            "mensaje": f"Para el trámite de **{datos_tramite['titulo']}**, hay varias ubicaciones disponibles. Por favor, selecciona una:\n" +
                         "\n".join(opciones_formateadas) + "\n¿Cuál de estas ubicaciones prefieres?",
            "necesita_seleccion": True,
            "datos_tramite_identificado": datos_tramite 
        }

    return _generar_respuesta_con_datos(datos_tramite, consulta, categoria_id)

def _generar_respuesta_con_datos(datos_tramite, consulta, categoria_id):
    """
    Helper function to generate the textual response with the procedure data.
    This function is called once the complete data and location (if applicable) are defined.
    The 'datos_tramite' here is the fully structured dictionary from the knowledge base.
    """
    mensaje = ""
    info = {
        "titulo": datos_tramite.get('titulo', 'Trámite no especificado'),
        "descripcion": datos_tramite.get('descripcion', ''),
        "requisitos": datos_tramite.get('requisitos', []),
        "observaciones": datos_tramite.get('observaciones', []), 
        "pasos": datos_tramite.get('pasos', []), 
        "costo": datos_tramite.get('costo', 'No especificado'),
        "direccion": datos_tramite.get('direccion', 'No disponible'),
        "coordenadas": datos_tramite.get('coordenadas', ''),
        "horarios": datos_tramite.get('horarios', ''),
        "telefono": datos_tramite.get('telefono', ''),
        "email": datos_tramite.get('email', ''),
        "sitio": datos_tramite.get('sitio_oficial', ''), # Corregido: en tu JSON es 'sitio_oficial'
        "responsable": datos_tramite.get('responsable', ''),
        "modalidad": datos_tramite.get('modalidad', ''),
        "mapa_url": datos_tramite.get('mapa_url', ''),
        "formularios": datos_tramite.get('formularios', [])
    }

    consulta_lower = consulta.lower()
    
    # Lógica de detección de intención (ubicación, costo, etc.)
    es_consulta_ubicacion = any(k in consulta_lower for k in ["ubicacion", "dónde", "cómo llegar", "dirección", "lugar", "oficina", "dependencia"])
    es_consulta_costo = any(k in consulta_lower for k in ["costo", "cuánto sale", "valor", "precio", "arancel", "pago"])
    es_consulta_formularios = any(k in consulta_lower for k in ["formulario", "formularios", "documento", "descargar", "archivo", "papel", "modelo"])
    es_consulta_requisitos = any(k in consulta_lower for k in ["requisito", "requisitos", "necesito", "qué llevar", "qué presentar", "documentos"])
    es_consulta_pasos = any(k in consulta_lower for k in ["pasos", "cómo se hace", "procedimiento", "proceso", "realizar"])
    es_consulta_observaciones = any(k in consulta_lower for k in ["observaciones", "observacion", "notas", "detalles adicionales", "importante", "tener en cuenta"])

    if es_consulta_ubicacion:
        if info['direccion'] and info['direccion'] != "No disponible":
            mensaje = f"La ubicación para **{info['titulo']}** es:\n- 📍 **Dirección:** {info['direccion']}."
            if info['horarios']:
                mensaje += f"\n- ⏰ **Horarios:** {info['horarios']}."
            if info['telefono']:
                mensaje += f"\n- 📞 **Teléfono:** {info['telefono']}."
            if info['email']:
                mensaje += f"\n- 📧 **E-mail:** {info['email']}."
            if info['responsable']:
                mensaje += f"\n- 👤 **Responsable:** {info['responsable']}."
            if info['mapa_url']:
                mensaje += f"\n[Ver en Google Maps]({info['mapa_url']})"
            mensaje += "\n¿Necesitas saber cómo llegar o algún otro detalle?"
        else:
            mensaje = f"No pude encontrar la ubicación exacta para **{info['titulo']}** en mi base de datos. Te recomiendo contactar al organismo directamente."
    elif es_consulta_costo:
        if info['costo'] != "No especificado":
            if isinstance(info['costo'], list):
                mensaje = f"Los costos para el trámite de **{info['titulo']}** son:\n"
                for c in info['costo']:
                    mensaje += f"- {c.get('descripcion', 'Costo')}: {c.get('valor', 'No especificado')}\n"
            else:
                mensaje = f"El costo para el trámite de **{info['titulo']}** es: **{info['costo']}**."
        else:
            mensaje = f"No se encontró información específica sobre el costo para el trámite de **{info['titulo']}**."
    elif es_consulta_formularios:
        if info['formularios']:
            mensaje = f"Para el trámite de **{info['titulo']}**, puedes descargar los siguientes formularios:\n"
            for form in info['formularios']:
                form_name = form.get('nombre', 'Formulario')
                form_url = form.get('url', '#')
                mensaje += f"- 📄 [{form_name}]({form_url})\n" 
        else:
            mensaje = f"No se encontraron formularios específicos para el trámite de **{info['titulo']}**."
    elif es_consulta_requisitos:
        if info['requisitos']:
            if isinstance(info['requisitos'], list):
                mensaje = f"📋 **Requisitos para el trámite de {info['titulo']}**:\n"
                for i, req in enumerate(info['requisitos'], 1):
                    mensaje += f"{i}. {req}\n"
            else: 
                mensaje = f"📋 **Requisitos para el trámite de {info['titulo']}**: {info['requisitos']}."
        else:
            mensaje = f"No se encontraron requisitos específicos para el trámite de **{info['titulo']}**."
    elif es_consulta_observaciones:
        if info['observaciones']:
            if isinstance(info['observaciones'], list): 
                mensaje = f"⚠️ **Observaciones importantes para el trámite de {info['titulo']}**:\n"
                for i, obs in enumerate(info['observaciones'], 1):
                    mensaje += f"{i}. {obs}\n"
            else: 
                mensaje = f"⚠️ **Observaciones importantes para el trámite de {info['titulo']}**: {info['observaciones']}."
        else:
            mensaje = f"No se encontraron observaciones específicas para el trámite de **{info['titulo']}**."
    elif es_consulta_pasos:
        if info['pasos']:
            if isinstance(info['pasos'], list): 
                mensaje = f"➡️ **Pasos para realizar el trámite de {info['titulo']}**:\n"
                for i, paso in enumerate(info['pasos'], 1):
                    mensaje += f"{i}. {paso}\n"
            else: 
                mensaje = f"➡️ **Pasos para realizar el trámite de {info['titulo']}**: {info['pasos']}."
        else:
            mensaje = f"No se encontraron pasos detallados para el trámite de **{info['titulo']}**."
    else:
        # Consulta general sobre el trámite
        mensaje = f"Información sobre **{info['titulo']}**:\n"
        if info['descripcion']:
            mensaje += f"{info['descripcion']}\n"

        summary_info = []
        if info['direccion'] != "No disponible":
            summary_info.append(f"📍 Ubicación disponible")
        if info['costo'] != "No especificado" and (isinstance(info['costo'], str) and info['costo'].lower() != 'ninguno' or isinstance(info['costo'], list) and info['costo']):
            summary_info.append(f"💰 Costo: {info['costo'] if isinstance(info['costo'], str) else 'Ver detalles'}")
        if info['modalidad']:
            summary_info.append(f"💻 Modalidad: {info['modalidad']}")
        if info['formularios']:
            summary_info.append(f"📄 Formularios disponibles ({len(info['formularios'])})")

        if summary_info:
            mensaje += "\n" + "\n".join(summary_info) + "\n"

        mensaje += "\n¿Qué más te gustaría saber? Por ejemplo: requisitos, pasos, horarios, teléfono, email, o si hay formularios."

    return {
        "tipo": "tramite_especifico",
        "categoria": categoria_id,
        "info": info, 
        "mensaje": mensaje,
        "necesita_seleccion": False, 
        "datos_tramite_identificado": info 
    }

def llamar_ia_openrouter(mensaje_usuario, historial):
    """
    **Función revisada: Ahora SOLO se usa para consultas NO directamente cubiertas por la lógica RAG + formateo.**
    Llama a la API de OpenRouter con un mensaje general si el RAG no encontró un trámite específico
    o si la intención es muy general/conversacional.
    """
    if not OPENROUTER_API_KEY:
        logger.error("OpenRouter API key not configured")
        return {"respuesta": "Error de configuración. Por favor, contacta al administrador.", "error": True}

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    ]
    
    historial_formateado = []
    for item in historial or []:
        if isinstance(item, dict):
            if item.get("usuario"):
                historial_formateado.append({"role": "user", "content": [{"type": "text", "text": item["usuario"]}]})
            if item.get("asistente"):
                historial_formateado.append({"role": "assistant", "content": [{"type": "text", "text": item["asistente"]}]})

    if historial_formateado:
        messages.extend(historial_formateado)
    
    messages.append({"role": "user", "content": [{"type": "text", "text": mensaje_usuario}]})

    data = {
        "model": "google/gemini-2.0-flash-exp:free", # Asegúrate de que este es el modelo que quieres usar
        "messages": messages,
        "max_tokens": 1000, 
        "temperature": 0.5
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(data, indent=2, ensure_ascii=False),
            timeout=45
        )
        response.raise_for_status()
        resultado = response.json()
        if 'choices' not in resultado or not resultado['choices']:
            logger.error(f"Unexpected response from OpenRouter: {resultado}")
            return {"respuesta": "No pude generar una respuesta adecuada. ¿Podrías reformular tu pregunta?", "error": True}

        respuesta_ia = resultado['choices'][0]['message']['content']

        # En este punto, como llamar_ia_openrouter es el fallback, no tenemos un 'info' de trámite específico
        # La respuesta estructurada aquí podría indicar que no se encontró un trámite.
        return {"respuesta": respuesta_ia, "datos_estructurados": {"tipo": "respuesta_general_ia"}, "timestamp": datetime.now().isoformat()}

    except requests.RequestException as e:
        logger.error(f"Network or HTTP error when calling OpenRouter: {e}")
        return {"respuesta": "Hubo un problema técnico al conectar con la IA. Intenta de nuevo más tarde.", "error": True}
    except Exception as e:
        logger.error(f"Ocurrió un error inesperado al procesar la respuesta de la IA: {e}", exc_info=True)
        return {"respuesta": "Ocurrió un error inesperado. Por favor, contacta al soporte.", "error": True}