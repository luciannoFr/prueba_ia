import json
import requests
import logging
from datetime import datetime

from config import OPENROUTER_API_KEY
from rag_embedder import buscar_tramite_por_embedding  
from data_manager import load_knowledge_base

base_conocimiento = load_knowledge_base()
sugerencias = [item.get('titulo', 'Trámite sin título') for item in base_conocimiento]

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
# ASISTENTE VIRTUAL ESPECIALIZADO EN TRÁMITES DE FORMOSA

Eres un asistente virtual especializado en trámites del gobierno de Formosa, Argentina. Tu función es proporcionar información precisa y accesible extraída exclusivamente del contexto estructurado (JSON) que recibes.

## REGLAS FUNDAMENTALES (NO NEGOCIABLES)

### 1. FUENTE DE INFORMACIÓN
- **SOLO** usa información del contexto estructurado (JSON) proporcionado.
- Si un dato no está en el contexto: "No encontré información específica sobre eso en mi base de datos".
- NUNCA inventes o supongas información.

### 2. COMUNICACIÓN PROGRESIVA
- **Paso 1**: Respuesta inicial breve (2-3 líneas) si es una consulta general.
- **Paso 2**: Pregunta qué específicamente necesita saber (requisitos, costos, etc.).
- **Paso 3**: Proporciona SOLO la información solicitada cuando la consulta es específica.

### 3. MANEJO ESTRICTO DE UBICACIONES

#### MÚLTIPLES UBICACIONES (si `tiene_opciones_ubicacion` es True y `opciones_ubicacion` tiene > 1 elemento):
OBLIGATORIO: DETENER todo otro flujo.

Mostrar SOLO lista numerada de opciones.

NO dar información adicional del trámite en esta respuesta.

Esperar selección del usuario.

Solo continuar después de la elección.


#### UNA UBICACIÓN (si `tiene_opciones_ubicacion` es False pero `direccion` está presente):
Mostrar información completa de ubicación cuando se pregunte por "dónde" o similar.


#### SIN UBICACIÓN (si `direccion` y `opciones_ubicacion` están vacíos):
Informar que no hay ubicación específica disponible.


---

## ESTRUCTURA DE RESPUESTA POR TIPO DE CONSULTA

### CONSULTA GENERAL SOBRE TRÁMITE
Formato:
"El trámite de {titulo} {descripcion_breve}.

¿Qué necesitas saber específicamente?
• Requisitos
• Costos

• Formularios
• Ubicación
• Horarios"


### CONSULTA ESPECÍFICA DE UBICACIÓN
**SI HAY MÚLTIPLES UBICACIONES (el sistema ya debería haber pedido una selección y esta respuesta solo se da tras la selección):**
"📍 Ubicación: {nombre_ubicacion_seleccionada}
📍 Dirección: {direccion}
⏰ Horarios: {horarios}
📞 Teléfono: {telefono}
📧 Email: {email}
Ver en Google Maps"


**SI HAY UNA UBICACIÓN (o ya se ha seleccionado una):**
"📍 Ubicación: {direccion}
⏰ Horarios: {horarios}
📞 Teléfono: {telefono}
📧 Email: {email}
Ver en Google Maps"

*(Nota: Si el campo `nombre` de la ubicación es relevante, se puede incluir.)*

### CONSULTA ESPECÍFICA DE COSTO
"💰 Costo: {costo_formateado_o_lista_de_costos}
¿Necesitas algún otro detalle?"


### CONSULTA ESPECÍFICA DE FORMULARIOS
"📄 Formularios disponibles:
• [**{nombre_formulario_1}**](**{url_formulario_1}**): Descargar
• [**{nombre_formulario_2}**](**{url_formulario_2}**): Descargar

¿Te ayudo con algo más?"


### CONSULTA ESPECÍFICA DE REQUISITOS
"📋 Requisitos para {titulo}:
• {requisito_1}
• {requisito_2}
• {requisito_3}

¿Necesitas más información?"


### CONSULTA ESPECÍFICA DE OBSERVACIONES
"⚠️ Observaciones importantes para {titulo}:
• {observacion_1}
• {observacion_2}

¿Hay algo más en lo que pueda ayudarte?"


### CONSULTA ESPECÍFICA DE PASOS
"➡️ Pasos para realizar {titulo}:

{paso_1}

{paso_2}

¿Algo más sobre el procedimiento?"

*(Nota: Si 'pasos' está vacío, responder que no hay información detallada de pasos.)*


---

## PALABRAS CLAVE DE ACTIVACIÓN

### Para UBICACIÓN:
- "dónde", "ubicación", "dirección", "lugar", "oficina", "dependencia", "llegar"

### Para COSTO:
- "cuánto", "costo", "precio", "vale", "sale", "arancel", "pago"

### Para FORMULARIOS:
- "formulario", "formularios", "planilla", "papel", "documento", "descargar", "modelo"

### Para REQUISITOS:
- "requisito", "requisitos", "necesito", "qué llevar", "qué presentar", "documentos"

### Para OBSERVACIONES:
- "observación", "observaciones", "notas", "detalles adicionales", "importante", "tener en cuenta"

### Para PASOS:
- "pasos", "cómo se hace", "procedimiento", "proceso", "realizar"

---

## MANEJO DE CONTEXTO CONVERSACIONAL

### PRIMERA CONSULTA
1. Identifica el trámite mencionado usando RAG.
2. Da respuesta inicial breve.
3. Ofrece opciones específicas.

### CONSULTAS DE SEGUIMIENTO
1. Revisa `historial_ia` para contexto y el `datos_estructurados` del último intercambio.
2. Responde basado en trámite ya establecido o nueva consulta.
3. Mantén coherencia con respuestas anteriores.

---

## FORMATO DE PRESENTACIÓN

### EMOJIS OBLIGATORIOS:
- 📍 Ubicación/Dirección
- ⏰ Horarios
- 📞 Teléfono
- 📧 Email
- 📄 Formularios
- 💰 Costo
- 📋 Requisitos
- ⚠️ Advertencias/Observaciones
- ➡️ Pasos

### ESTRUCTURA VISUAL:
- Usa **negritas** para títulos y datos importantes.
- Un salto de línea entre secciones.
- Viñetas (•) o numeración (1., 2.) para listas.
- Enlaces con formato [Texto](URL).

---

## ALGORITMO DE DECISIÓN

RECIBIR CONSULTA
↓
¿Hay una selección de ubicación pendiente en la sesión? (Gestionado en app.py)
├─ SÍ → Procesar selección numérica → Generar respuesta de ubicación.
└─ NO → Continuar
↓
Buscar trámite relevante usando RAG (en buscar_tramites_inteligente).
↓
¿Se encontró un trámite relevante?
├─ SÍ →
│  ¿El trámite tiene múltiples ubicaciones Y NO se ha seleccionado una aún?
│  ├─ SÍ → Mostrar SOLO lista numerada de opciones de ubicación → Esperar selección.
│  └─ NO →
│     ¿Es consulta específica (costo/formularios/requisitos/observaciones/pasos/ubicacion)?
│     ├─ SÍ → Mostrar SOLO esa información específica.
│     └─ NO → Dar respuesta general + opciones.
└─ NO → Informar que no se encontró información específica.


---

## CASOS ESPECIALES

### SIN INFORMACIÓN DISPONIBLE PARA ASPECTO ESPECÍFICO:
"No encontré información sobre {aspecto_consultado} para este trámite en mi base de datos."


### CONSULTA AMBIGUA (si RAG retorna múltiples trámites con similar score, o no queda claro a qué se refiere):
"¿Te refieres a {opcion_1}, {opcion_2} o {opcion_3}?"


---

**RECORDATORIO CRÍTICO**: La eficiencia viene de la precisión en seguir estas reglas, no de respuestas largas. Cada interacción debe resolver exactamente lo que el usuario necesita en ese momento.
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

    # If current_tramite_data is provided, use it directly
    if current_tramite_data:
        datos_tramite = current_tramite_data
        categoria_id = current_tramite_data.get('categoria', 'desconocido')
        logger.info(f"Using current_tramite_data for response generation: {datos_tramite.get('titulo')}")
    else:
        tramites_relevantes = buscar_tramites_inteligente(consulta)

        if not tramites_relevantes:
            return {
                "tipo": "no_encontrado",
                "mensaje": "No encontré información específica sobre ese trámite en mi base de conocimientos. ¿Podrías ser más específico?",
                "sugerencias": sugerencias
            }

        mejor_resultado_rag = tramites_relevantes[0]
        datos_tramite = mejor_resultado_rag.get('data', {})
        categoria_id = mejor_resultado_rag.get('categoria', 'desconocido')
        
        if not datos_tramite or not datos_tramite.get('titulo'):
            logger.warning(f"RAG returned a relevant URL ({mejor_resultado_rag.get('url')}) but its 'data' was empty or invalid.")
            return {
                "tipo": "error",
                "mensaje": "Encontré algo relacionado, pero no pude obtener la información completa del trámite en este momento. Por favor, intenta de nuevo más tarde o sé más específico.",
                "sugerencias": sugerencias
            }

    if datos_tramite.get('tiene_opciones_ubicacion', False) and datos_tramite.get('opciones_ubicacion'):
        opciones_formateadas = []
        for i, op in enumerate(datos_tramite['opciones_ubicacion']):
            dir_info = f"- 📍 Dirección: {op.get('direccion', 'No disponible')}" if op.get('direccion') else ""
            hor_info = f"\n- ⏰ Horarios: {op.get('horarios', 'No disponible')}" if op.get('horarios') else ""
            opciones_formateadas.append(f"{i+1}. {op['nombre']}{dir_info}{hor_info}")

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
        "observaciones": datos_tramite.get('observaciones', ''),
        "pasos": datos_tramite.get('pasos', []),
        "costo": datos_tramite.get('costo', 'No especificado'),
        "direccion": datos_tramite.get('direccion', 'No disponible'),
        "coordenadas": datos_tramite.get('coordenadas', ''),
        "horarios": datos_tramite.get('horarios', ''),
        "telefono": datos_tramite.get('telefono', ''),
        "email": datos_tramite.get('email', ''),
        "sitio": datos_tramite.get('sitio', ''),
        "responsable": datos_tramite.get('responsable', ''),
        "modalidad": datos_tramite.get('modalidad', ''),
        "mapa_url": datos_tramite.get('mapa_url', ''),
        "tiene_opciones_ubicacion": datos_tramite.get('tiene_opciones_ubicacion', False),
        "opciones_ubicacion": datos_tramite.get('opciones_ubicacion', []),
        "necesita_seleccion": datos_tramite.get('necesita_seleccion', False),
        "formularios": datos_tramite.get('formularios', [])
    }

    consulta_lower = consulta.lower()
    
    es_consulta_ubicacion = any(k in consulta_lower for k in ["ubicacion", "dónde", "cómo llegar", "dirección", "lugar", "oficina", "dependencia"])
    es_consulta_costo = any(k in consulta_lower for k in ["costo", "cuánto sale", "valor", "precio", "arancel", "pago"])
    es_consulta_formularios = any(k in consulta_lower for k in ["formulario", "formularios", "documento", "descargar", "archivo", "papel", "modelo"])
    es_consulta_requisitos = any(k in consulta_lower for k in ["requisito", "requisitos", "necesito", "qué llevar", "qué presentar", "documentos"])
    es_consulta_pasos = any(k in consulta_lower for k in ["pasos", "cómo se hace", "procedimiento", "proceso", "realizar"])
    es_consulta_observaciones = any(k in consulta_lower for k in ["observaciones", "observacion", "notas", "detalles adicionales", "importante", "tener en cuenta"])

    if es_consulta_ubicacion:
        if info['direccion'] != "No disponible":
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
        "necesita_seleccion": info['necesita_seleccion'], 
        "datos_tramite_identificado": info 
    }

def llamar_ia_openrouter(mensaje_usuario, historial, context_override=None):
    """
    Llama a la API de OpenRouter siguiendo el formato oficial.
    Usa contexto estructurado y el historial de la conversación.
    Prioriza `context_override` for the current trámite's data.
    """
    if not OPENROUTER_API_KEY:
        logger.error("OpenRouter API key not configured")
        return {"respuesta": "Error de configuración. Por favor, contacta al administrador.", "error": True}

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost:5000",  
        "X-Title": "Asistente de Trámites Formosa",  
        "HTTP-Referer": "http://localhost:5000",  # opcional según docs
        "X-Title": "Asistente de Trámites Formosa",  # opcional según docs
        "Content-Type": "application/json"
    }

    respuesta_contextual = None
    contexto_texto = ""

    if context_override and context_override.get('tipo') == 'tramite_especifico' and context_override.get('info'):
        respuesta_contextual = context_override['info']
        contexto_texto = f"""
INFORMACIÓN DEL TRÁMITE ACTUAL EN CONTEXTO:
Título: {respuesta_contextual.get('titulo', 'No disponible')}
Descripción: {respuesta_contextual.get('descripcion', 'No disponible')}
Requisitos: {', '.join(respuesta_contextual.get('requisitos', [])) if respuesta_contextual.get('requisitos') else 'No disponible'}
Costo: {respuesta_contextual.get('costo', 'No disponible')}
Dirección: {respuesta_contextual.get('direccion', 'No disponible')}
Teléfono: {respuesta_contextual.get('telefono', 'No disponible')}
Horarios: {respuesta_contextual.get('horarios', 'No disponible')}
Email: {respuesta_contextual.get('email', 'No disponible')}
Observaciones: {respuesta_contextual.get('observaciones', 'No disponible')}
Pasos: {', '.join(respuesta_contextual.get('pasos', [])) if respuesta_contextual.get('pasos') else 'No disponible'}
Formularios: {', '.join([f.get('nombre', 'Formulario') for f in respuesta_contextual.get('formularios', [])]) if respuesta_contextual.get('formularios') else 'No disponible'}
URL del Trámite: {respuesta_contextual.get('url', 'No disponible')}
"""
        logger.info(f"Using context_override for LLM prompt (Trámite: {respuesta_contextual.get('titulo')})")
    else:
        # Otherwise, perform RAG search
        contextos = buscar_tramite_por_embedding(mensaje_usuario, top_k=1)
        contexto = contextos[0] if contextos else None

        if contexto:
            respuesta_contextual = contexto.get('data', {}) # The 'data' key contains the full scraped dict
            contexto_texto = f"""
INFORMACIÓN RELEVANTE DEL TRÁMITE IDENTIFICADO POR RAG:
Título: {respuesta_contextual.get('titulo', 'No disponible')}
Descripción: {respuesta_contextual.get('descripcion', 'No disponible')}
Requisitos: {', '.join(respuesta_contextual.get('requisitos', [])) if respuesta_contextual.get('requisitos') else 'No disponible'}
Costo: {respuesta_contextual.get('costo', 'No disponible')}
Dirección: {respuesta_contextual.get('direccion', 'No disponible')}
Teléfono: {respuesta_contextual.get('telefono', 'No disponible')}
Horarios: {respuesta_contextual.get('horarios', 'No disponible')}
Email: {respuesta_contextual.get('email', 'No disponible')}
Observaciones: {respuesta_contextual.get('observaciones', 'No disponible')}
Pasos: {', '.join(respuesta_contextual.get('pasos', [])) if respuesta_contextual.get('pasos') else 'No disponible'}
Formularios: {', '.join([f.get('nombre', 'Formulario') for f in respuesta_contextual.get('formularios', [])]) if respuesta_contextual.get('formularios') else 'No disponible'}
URL del Trámite: {respuesta_contextual.get('url', 'No disponible')}
"""
            logger.info(f"Performing RAG search for LLM prompt (Trámite: {respuesta_contextual.get('titulo')})")
        else:
            contexto_texto = "No se encontró un trámite exacto o muy relevante en la base de datos para la consulta actual. Responde de forma general o pide más aclaraciones."
            respuesta_contextual = {"tipo": "no_encontrado"}
            logger.info("No specific trámite found via RAG.")


    historial_formateado = []
    for item in historial:
        assistant_content = item.get("asistente", "") if isinstance(item, dict) else str(item)
        user_content = item.get("usuario", "") if isinstance(item, dict) else ""
        if user_content:
            historial_formateado.append({"role": "user", "content": [{"type": "text", "text": user_content}]})
        if assistant_content:
            historial_formateado.append({"role": "assistant", "content": [{"type": "text", "text": assistant_content}]})



    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{contexto_texto}"}] 
        }
    ] + historial_formateado + [
        {
            "role": "user",
            "content": [{"type": "text", "text": mensaje_usuario}]
        }
    ]

    data = {
        "model": "mistralai/mistral-nemo:free",
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.5
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(data),
            timeout=45
        )
        response.raise_for_status()

        resultado = response.json()
        if 'choices' not in resultado or not resultado['choices']:
            logger.error(f"Unexpected response from OpenRouter: {resultado}")
            return {"respuesta": "No pude generar una respuesta adecuada. ¿Podrías reformular tu pregunta?", "error": True}

        respuesta_ia = resultado['choices'][0]['message']['content']
        
        final_datos_estructurados = None
        if context_override and context_override.get('tipo') == 'tramite_especifico' and context_override.get('info'):
            final_datos_estructurados = {"tipo": "tramite_especifico", "info": context_override['info']}
        elif respuesta_contextual.get("tipo", "") != "no_encontrado":
            final_datos_estructurados = {"tipo": "tramite_especifico", "info": respuesta_contextual}
        
        return {
            "respuesta": respuesta_ia,
            "datos_estructurados": final_datos_estructurados,
            "timestamp": datetime.now().isoformat()
        }
    except requests.RequestException as e:
        logger.error(f"Network or HTTP error when calling OpenRouter: {e}")
        return {"respuesta": "Hubo un problema técnico al conectar con la IA. Intenta de nuevo más tarde.", "error": True}
    except Exception as e:
        logger.error(f"Ocurrió un error inesperado al procesar la respuesta de la IA: {e}", exc_info=True)
        return {"respuesta": "Ocurrió un error inesperado. Por favor, contacta al soporte.", "error": True}