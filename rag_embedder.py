# rag_embedder.py (asegúrate de que este archivo tenga estos cambios)

import os
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import logging

from config import KNOWLEDGE_BASE_FILE, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

model = SentenceTransformer(EMBEDDING_MODEL_NAME)

EMBEDDINGS_FILE = "data/tramites_embeddings.json"

SIMILARITY_THRESHOLD = 0.55

def crear_embeddings():
    """Genera embeddings para todos los trámites y los guarda."""
    with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as f:
        base = json.load(f)

    embeddings = []

    for tramite in base:
        titulo = tramite.get('data', {}).get('titulo', '')
        descripcion = tramite.get('data', {}).get('descripcion', '')
        
        texto = f"{titulo}. {descripcion}" 
        
        emb = model.encode(texto, convert_to_numpy=True).tolist()
        
        embeddings.append({
            "url": tramite["url"],
            "titulo": titulo, 
            "embedding": emb 
        })

    with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, indent=2, ensure_ascii=False)

    print(f"*** Embeddings generados para {len(embeddings)} trámites.")


def buscar_tramite_por_embedding(pregunta, top_k=1):
    logger.debug(f"Iniciando búsqueda RAG con pregunta: {pregunta}")
    
    try:
        if not os.path.exists(EMBEDDINGS_FILE):
            logger.warning(f"Embeddings file not found: {EMBEDDINGS_FILE}. Creating embeddings now.")
            crear_embeddings() 

        with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
            base_emb = json.load(f)
        logger.debug(f"Cargados {len(base_emb)} embeddings")

        with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as f:
            base = json.load(f)
        logger.debug(f"Cargados {len(base)} trámites en base de conocimiento")

        pregunta_emb = model.encode(pregunta, convert_to_numpy=True)
        pregunta_emb = torch.tensor(pregunta_emb).float()
        logger.debug("Embedding generado para la pregunta")

        similitudes = []
        for tramite in base_emb:
            try:
                emb_tramite = torch.tensor(tramite["embedding"]).float()
                score = util.cos_sim(pregunta_emb, emb_tramite).item()
                
                logger.info(f"Similarity score for '{tramite.get('titulo')}' with query '{pregunta}': {score}")

                if score >= SIMILARITY_THRESHOLD: 
                    if tramite.get("url") is not None: 
                        similitudes.append((score, tramite["url"]))
                    else:
                        logger.warning(f"Embedding encontrado sin URL válida: {tramite}")
                else:
                    logger.debug(f"Tramite '{tramite.get('titulo')}' excluido por baja similitud: {score}")

            except Exception as e:
                logger.error(f"Error procesando embedding: {e}")

        similitudes.sort(reverse=True)
        mejores = similitudes[:top_k]
        logger.debug(f"Mejores similitudes encontradas: {mejores}")

        resultados = []
        mejores_urls = {url for score, url in mejores if score >= SIMILARITY_THRESHOLD} 
        
        for t in base:
            try:
                if t.get("url") in mejores_urls:
                    resultados.append(t)
            except Exception as e:
                logger.error(f"Error procesando trámite de la base: {e}")

        if not resultados:
            logger.warning("No se encontraron datos válidos para las URLs relevantes (quizás por el umbral de similitud)")
        return resultados

    except Exception as e:
        logger.error(f"Error en la búsqueda RAG: {e}")
        return []