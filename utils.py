# utils.py
import json
import requests
import logging
from datetime import datetime
from urllib.parse import quote # Añadir import aquí

from config import OPENROUTER_API_KEY
from rag_embedder import buscar_tramite_por_embedding  
from data_manager import load_knowledge_base

base_conocimiento = load_knowledge_base()
print(f"Cantidad de entradas en la base de conocimiento: {len(base_conocimiento)}")
# Cargar sugerencias al inicio
sugerencias_globales = [item.get('titulo', 'Trámite sin título') for item in base_conocimiento if item.get('titulo')]

logger = logging.getLogger(__name__)

# SYSTEM_PROMPT ajustado para ser más conversacional cuando no hay RAG exacta
SYSTEM_PROMPT = """
# ASISTENTE VIRTUAL DE TRÁMITES — GOBIERNO DE FORMOSA

Sos un asistente virtual especializado **EXCLUSIVAMENTE** en trámites del **Gobierno de Formosa, Argentina**.

Tu única función es responder **consultas sobre trámites**, utilizando **preferentemente** los datos estructurados (en formato JSON) que te serán proporcionados.

---

## ⚠️ REGLAS DE FUNCIONAMIENTO

### 1. ALCANCE PERMITIDO

✔️ Podés responder sobre:
- Trámites del gobierno de Formosa.
- Información relacionada directamente a esos trámites (requisitos, costos, ubicación, formularios, pasos, observaciones).

❌ No podés responder sobre:
- Programación, tecnología, historia, matemática, ni ningún tema que no esté directamente vinculado a trámites de Formosa.
- Trámites de otras provincias o países.
- Preguntas personales, genéricas o hipotéticas.

➡️ **Si el usuario consulta algo fuera del alcance permitido**, respondé de forma terminante:

> "Este asistente solo responde consultas relacionadas a trámites del Gobierno de Formosa. No puedo ayudarte con eso."

---

### 2. USO DEL CONTEXTO Y DATOS JSON

- **Priorizá siempre la información estructurada** si se te proporciona en el contexto.
- Si una información específica (ej. un requisito, un costo) no está presente en el JSON, indicá claramente:
  > "No encontré información específica sobre [lo que preguntó] en mi base de datos para este trámite."
- **No inventes, supongas ni rellenes** datos faltantes.

### 3. RESPUESTAS CONVERSACIONALES Y CLARAS

- Respondé de manera amigable, clara y concisa.
- Utilizá un tono servicial.
- Si la consulta es general sobre un trámite, ofrecé al usuario opciones sobre qué información desea (requisitos, costo, etc.) como sugerencia.
- Si el usuario ya hizo una consulta específica, respondé directamente eso.

---

### 4. MANEJO DE UBICACIONES

#### Múltiples ubicaciones
- Si un trámite tiene **varias opciones de ubicación**, presentá una lista numerada de forma clara y pedí al usuario que seleccione una.
- **NO brindes otra información del trámite** hasta que el usuario haya seleccionado una ubicación.

#### Una única ubicación
- Si el trámite tiene una **única ubicación**, al consultarse "dónde" o similar, proporcioná la dirección, horarios, teléfono y email si están disponibles.

#### Sin ubicación disponible
- Si no hay información de dirección para el trámite, informá que no hay dirección disponible.

---

## 📌 EJEMPLOS DE FORMATO (GUÍA)

**Consulta general sobre trámite (después de identificarlo)**
"¡Claro! El trámite de **{titulo}** se trata de: {descripcion_breve}.
¿Qué te gustaría saber específicamente? Por ejemplo:
• Requisitos
• Costos
• Formularios
• Ubicación
• Horarios"

**Requisitos**
"📋 Para el trámite de **{titulo}**, estos son los requisitos:
• {requisito_1}
• {requisito_2}"
(Formatear como lista numerada si aplica)

**Costo**
"💰 El costo para el trámite de **{titulo}** es: **{costo}**."
(Si es una lista de costos, presentarlos claramente)

**Formularios**
"📄 Para el trámite de **{titulo}**, puedes descargar los siguientes formularios:
• [**{nombre}**]({url})"
(Formatear como lista si aplica)

**Pasos**
"➡️ Para realizar el trámite de **{titulo}**, seguí estos pasos:
1. {paso_1}
2. {paso_2}"
(Formatear como lista numerada)

**Ubicación (una única)**
"📍 La ubicación para **{titulo}** es:
**Dirección:** {direccion}
**Horarios:** {horarios}
**Teléfono:** {telefono}
**E-mail:** {email}"
(Incluir URL de Google Maps si está disponible)

**Ubicación (múltiples - primera respuesta)**
"Para el trámite de **{titulo}**, hay varias ubicaciones disponibles. Por favor, selecciona una opción numerada de la siguiente lista:
1. {nombre_ubicacion_1} (Dirección: ..., Horarios: ...)
2. {nombre_ubicacion_2} (Dirección: ..., Horarios: ...)"

**Observaciones**
"⚠️ Tené en cuenta estas observaciones importantes para el trámite de **{titulo}**:
• {observacion_1}"
(Formatear como lista numerada si aplica)

---

## 🔁 CONTEXTO CONVERSACIONAL

- **Mantené el contexto del último trámite** que el usuario consultó hasta que cambie de tema explícitamente o inicie una nueva búsqueda.
- Si el usuario selecciona un número tras una pregunta de ubicación, interpretalo como la elección de la ubicación.
- Evitá repetir información que ya proporcionaste en el mismo hilo de conversación.

---

## 📛 EJEMPLOS DE COSAS QUE DEBÉS IGNORAR

- "¿Cómo se escribe 'Hola mundo' en C++?"
- "¿Cuál es la capital de Francia?"
- "¿Cómo saco turno en Buenos Aires?"

➡️ En estos casos, simplemente respondé:
> "Este asistente solo responde consultas relacionadas a trámites del Gobierno de Formosa. No puedo ayudarte con eso."

---

## RECORDATORIO FINAL

🔒 **NO RESPONDAS NINGUNA CONSULTA** que no esté relacionada con un trámite de Formosa ni que no se fundamente 100% en los datos de tu base de conocimiento.

🎯 Tu objetivo es **ser un asistente útil y amable**, **nunca inventar**, y **nunca desviarte de tu dominio**.
"""

def buscar_tramites_inteligente(consulta):
    """
    Uses the RAG system to retrieve the most relevant procedures based on the user's query.
    Returns a list of structured procedure data directly from the knowledge base.
    """
    retrieved_results = buscar_tramite_por_embedding(consulta)
    # logger.debug(f"Resultados de RAG para '{consulta}': {retrieved_results}")
    return retrieved_results


def generar_respuesta_contextual(mensaje_usuario, historial_conversacion=None, current_tramite_data=None):
    """
    Genera la respuesta contextual:
     1) Atiende selección de ubicación pendiente
     2) Atiende sub-preguntas (“requisitos”, “costos”, etc.) sobre current_tramite_data
     3) Detecta cambio explícito de trámite via RAG
     4) Fallback a LLM si no hay trámite
     5) Formatea selección de ubicaciones múltiples
    """
    if not historial_conversacion:
        historial_conversacion = []
    mensaje_lower = mensaje_usuario.lower()

    if current_tramite_data and current_tramite_data.get('necesita_seleccion') and mensaje_usuario.isdigit():
        return _procesar_seleccion_ubicacion(mensaje_usuario, current_tramite_data)

    campos_basicos = [
    "requisitos", "costo", "costos", "formularios",
    "ubicacion", "horarios", "pasos", "observaciones",
    "dónde", "cuánto", "descargar",
    "dirección", "teléfono", "email"
]
    if current_tramite_data and any(k in mensaje_lower for k in campos_basicos):
        categoria_id = current_tramite_data.get('categoria', 'desconocido')
        return _generar_respuesta_con_datos(current_tramite_data, mensaje_usuario, categoria_id)

    nuevos = buscar_tramites_inteligente(mensaje_usuario) or []
    if nuevos:
        logger.debug(f"[RAG] tras «{mensaje_usuario}»: {[t.get('data', {}).get('titulo', 'Sin título') for t in nuevos if 'data' in t]}")
    else:
        logger.debug(f"[RAG] tras «{mensaje_usuario}»: sin resultados.")
    primer = nuevos[0].get('data') if nuevos else None
    cambio_tramite = False
    if primer:
        tit_nuevo = primer.get('titulo', '').lower()
        tit_act   = (current_tramite_data or {}).get('titulo', '').lower()
        if tit_nuevo and tit_nuevo != tit_act:
            # Cambia sólo si mencionás el nuevo título o no pedís un campo básico
            if tit_nuevo in mensaje_lower or not any(k in mensaje_lower for k in campos_basicos):
                cambio_tramite = True

    if cambio_tramite:
        datos_tramite = primer
        categoria_id  = nuevos[0].get('categoria', 'desconocido')
    elif current_tramite_data:
        datos_tramite = current_tramite_data
        categoria_id  = current_tramite_data.get('categoria', 'desconocido')
    elif primer:
        datos_tramite = primer
        categoria_id  = nuevos[0].get('categoria', 'desconocido')
    else:
        ia = llamar_ia_openrouter(mensaje_usuario, historial_conversacion)

        if ia.get("tipo") == "error_red" and "Too Many Requests" in ia.get("respuesta", ""):
            logger.warning("Limitación de rate detectada. Mostrando mensaje amigable.")
            return {
                "mensaje": "El sistema está recibiendo muchas consultas en poco tiempo. Por favor, esperá unos segundos e intentá nuevamente.",
                "tipo": "error_limite",
                "sugerencias": sugerencias_globales,
                "necesita_seleccion": False,
                "opciones_ubicacion": [],
                "datos_tramite_identificado": None,
                "error": True
            }

    return {
        "mensaje": ia.get('respuesta', "Lo siento, no pude procesar tu solicitud."),
    }


    ubics = datos_tramite.get('opciones_ubicacion') or []
    if isinstance(ubics, list) and len(ubics) > 1:
        opciones = []
        for i, u in enumerate(ubics, 1):
            parts = []
            if u.get('direccion'): parts.append(f"Dirección: {u['direccion']}")
            if u.get('horarios'):  parts.append(f"Horarios: {u['horarios']}")
            opciones.append(f"{i}. {u.get('nombre', f'Opción {i}')} ({', '.join(parts)})")
        return {
            "tipo": "seleccion_ubicacion",
            "categoria": categoria_id,
            "opciones_ubicacion": ubics,
            "original_datos_tramite": datos_tramite,
            "mensaje": (
                f"Para el trámite **{datos_tramite['titulo']}**, hay varias sucursales:\n"
                + "\n".join(opciones)
                + "\nPor favor, escribí el número de la opción que prefieras."
            ),
            "necesita_seleccion": True,
            "datos_tramite_identificado": datos_tramite
        }

    return _generar_respuesta_con_datos(datos_tramite, mensaje_usuario, categoria_id)


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
        "sitio": datos_tramite.get('sitio_oficial', ''), 
        "responsable": datos_tramite.get('responsable', ''),
        "modalidad": datos_tramite.get('modalidad', ''),
        "mapa_url": datos_tramite.get('mapa_url', ''),
        "formularios": datos_tramite.get('formularios', [])
    }

    consulta_lower = consulta.lower()
    
    # Lógica de detección de intención (ubicación, costo, etc.)
    es_consulta_ubicacion = any(k in consulta_lower for k in ["ubicacion", "dónde", "cómo llegar", "dirección", "lugar", "oficina", "dependencia", "dirección:", "horarios:", "teléfono:", "email:"])
    es_consulta_costo = any(k in consulta_lower for k in ["costo", "cuánto sale", "valor", "precio", "arancel", "pago"])
    es_consulta_formularios = any(k in consulta_lower for k in ["formulario", "formularios", "documento", "descargar", "archivo", "papel", "modelo"])
    es_consulta_requisitos = any(k in consulta_lower for k in ["requisito", "requisitos", "necesito", "qué llevar", "qué presentar", "documentos"])
    es_consulta_pasos = any(k in consulta_lower for k in ["pasos", "cómo se hace", "procedimiento", "proceso", "realizar"])
    es_consulta_observaciones = any(k in consulta_lower for k in ["observaciones", "observacion", "notas", "detalles adicionales", "importante", "tener en cuenta"])

    # Prioridad para generar la respuesta detallada
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
        # Respuesta general sobre el trámite si no hay una intención específica
        mensaje = f"¡Claro! El trámite de **{info['titulo']}** se trata de: "
        if info['descripcion']:
            mensaje += f"{info['descripcion']}.\n"
        else:
            mensaje += "No tengo una descripción detallada, pero puedo darte más información. "

        summary_info = []
        # Solo añadir ubicación al resumen si no hay opciones_ubicacion o si hay una sola
        if not (datos_tramite.get('opciones_ubicacion') and len(datos_tramite['opciones_ubicacion']) > 1) and \
           info['direccion'] and info['direccion'] != "No disponible":
            summary_info.append(f"📍 Ubicación disponible")
        
        if info['costo'] != "No especificado" and (isinstance(info['costo'], str) and info['costo'].lower() != 'ninguno' or isinstance(info['costo'], list) and info['costo']):
            summary_info.append(f"💰 Costo: {info['costo'] if isinstance(info['costo'], str) else 'Ver detalles'}")
        if info['modalidad']:
            summary_info.append(f"💻 Modalidad: {info['modalidad']}")
        if info['formularios']:
            summary_info.append(f"📄 Formularios disponibles ({len(info['formularios'])})")

        if summary_info:
            mensaje += "\nAdemás, te cuento que:\n" + "\n".join(summary_info) + "\n"

        mensaje += "\n¿Qué más te gustaría saber sobre este trámite? Por ejemplo: requisitos, pasos, horarios, teléfono, email, o si hay formularios."
        return {
            "tipo": "tramite_especifico",
            "categoria": categoria_id,
            "info": info,  # <-- Este es un dict plano (sin ubicaciones, sin título completo)
            "mensaje": mensaje,
            "necesita_seleccion": False,
            "datos_tramite_identificado": datos_tramite
        }

def llamar_ia_openrouter(mensaje_usuario, historial):
    """
    Llama a la API de OpenRouter para una respuesta general cuando RAG no encuentra nada.
    """
    if not OPENROUTER_API_KEY:
        logger.error("OpenRouter API key not configured")
        return {"respuesta": "Error de configuración. Por favor, contacta al administrador.", "error": True, "tipo": "error_configuracion"}

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}, # SYSTEM_PROMPT ya es texto plano
    ]
    
    # Formatear el historial para el LLM
    historial_formateado = []
    for item in historial or []:
        # Asegurarse de que el historial tenga el formato correcto para el LLM
        if isinstance(item, dict):
            if item.get("usuario"):
                historial_formateado.append({"role": "user", "content": item["usuario"]})
            if item.get("asistente"):
                # Si el asistente fue un mensaje estructurado, tomar solo el campo 'respuesta'
                if isinstance(item["asistente"], dict) and "respuesta" in item["asistente"]:
                     historial_formateado.append({"role": "assistant", "content": item["asistente"]["respuesta"]})
                elif isinstance(item["asistente"], str):
                    historial_formateado.append({"role": "assistant", "content": item["asistente"]})


    if historial_formateado:
        messages.extend(historial_formateado)
    
    messages.append({"role": "user", "content": mensaje_usuario})

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
            return {"respuesta": "No pude generar una respuesta adecuada. ¿Podrías reformular tu pregunta?", "error": True, "tipo": "error_ia_vacia"}

        respuesta_ia = resultado['choices'][0]['message']['content']
        
        return {"respuesta": respuesta_ia, "tipo": "respuesta_general_ia", "sugerencias": sugerencias_globales}

    except requests.RequestException as e:
        logger.error(f"Network or HTTP error when calling OpenRouter: {e}")
        return {"respuesta": "Hubo un problema técnico al conectar con la IA. Intenta de nuevo más tarde.", "error": True, "tipo": "error_red"}
    except Exception as e:
        logger.error(f"Ocurrió un error inesperado al procesar la respuesta de la IA: {e}", exc_info=True)
        return {"respuesta": "Ocurrió un error inesperado. Por favor, contacta al soporte.", "error": True, "tipo": "error_interno_ia"}