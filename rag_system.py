import logging
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json 

from config import EMBEDDING_MODEL_NAME
from data_manager import load_knowledge_base, get_all_urls_to_scrape

logger = logging.getLogger(__name__)

embedding_model = None
knowledge_base_embeddings = []

def load_embedding_model():
    """Loads the sentence embedding model."""
    global embedding_model
    try:
        if embedding_model is None:
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}")
    except Exception as e:
        logger.error(f"Error loading embedding model {EMBEDDING_MODEL_NAME}: {e}")
        embedding_model = None

def build_knowledge_base_embeddings():
    """
    Loads scraped data from data_manager, ensures all primary URLs are scraped,
    and then creates embeddings for search.
    This should be called once at startup or when the knowledge base is updated.
    """
    global knowledge_base_embeddings
    load_embedding_model() 

    if not embedding_model:
        logger.error("Embedding model not loaded. Cannot build knowledge base embeddings.")
        return

    from scraper import scrape_tramite_data 
    
    urls_to_ensure = get_all_urls_to_scrape()
    
    scraped_data_entries = load_knowledge_base() 
    
    for url_to_check in urls_to_ensure:
        found_in_scraped = any(entry.get('url') == url_to_check for entry in scraped_data_entries)
        if not found_in_scraped:
            logger.info(f"URL {url_to_check} not found in knowledge base. Attempting to scrape...")
            scraped_data = scrape_tramite_data(url_to_check)
            if scraped_data:
                scraped_data_entries = load_knowledge_base() 
            else:
                logger.warning(f"Failed to scrape data for {url_to_check}. It will not be in the RAG system.")


    if not scraped_data_entries:
        logger.warning("No scraped data found to build knowledge base embeddings.")
        return

    knowledge_base_embeddings = []
    texts_to_embed = []
    metadata_list = []

    for entry in scraped_data_entries:
        data = entry.get('data', {})
        url = entry.get('url', '')
        
        categoria = entry.get('categoria', 'desconocido')  

        combined_text = f"{data.get('titulo', '')}. "
        if data.get('descripcion'):
            combined_text += f"{data['descripcion']} "
        if data.get('requisitos'):
            if isinstance(data['requisitos'], list):
                combined_text += "Requisitos: " + " ".join(data['requisitos']) + " "
            else:
                combined_text += f"Requisitos: {data['requisitos']} "
        if data.get('costo'):
            if isinstance(data['costo'], list):
                combined_text += "Costo: " + " ".join([f"{item['descripcion']} {item['valor']}" for item in data['costo']]) + " "
            else:
                combined_text += f"Costo: {data['costo']} "
        if data.get('modalidad'):
            combined_text += f"Modalidad: {data['modalidad']} "
        if data.get('direccion'):
            combined_text += f"Dirección: {data['direccion']} "
        if data.get('opciones_ubicacion'):
             for loc in data['opciones_ubicacion']:
                combined_text += f"Ubicación: {loc.get('nombre', '')} en {loc.get('direccion', '')}. "
        if data.get('formularios'):
            form_names = ", ".join([f['nombre'] for f in data['formularios']])
            combined_text += f"Formularios: {form_names}. "
        
        combined_text = ' '.join(combined_text.split()).strip()

        if combined_text:
            texts_to_embed.append(combined_text)
            metadata_list.append({
                "categoria": categoria,
                "url": url,
                "data": data 
            })
    
    if texts_to_embed:
        embeddings = embedding_model.encode(texts_to_embed, convert_to_tensor=False, show_progress_bar=False)
        for i, emb in enumerate(embeddings):
            knowledge_base_embeddings.append({
                "text": texts_to_embed[i],
                "embedding": emb,
                "metadata": metadata_list[i]
            })
        logger.info(f"Built RAG knowledge base with {len(knowledge_base_embeddings)} embedded entries.")
    else:
        logger.warning("No texts generated to embed for RAG knowledge base.")


def retrieve_relevant_documents(query, top_k=3, min_similarity=0.4): 
    """
    Retrieves the most relevant documents (tramites) from the knowledge base
    based on the semantic similarity of the query.
    """
    load_embedding_model() 

    if not embedding_model or not knowledge_base_embeddings:
        logger.warning("Embedding model or RAG knowledge base not ready for retrieval.")
        return []

    try:
        query_embedding = embedding_model.encode(query, convert_to_tensor=False)
    except Exception as e:
        logger.error(f"Error encoding query for retrieval: {e}")
        return []

    similarities = []
    for item in knowledge_base_embeddings:
        doc_embedding = item['embedding']
        similarity = cosine_similarity(query_embedding.reshape(1, -1), doc_embedding.reshape(1, -1))[0][0]
        similarities.append({"item": item['metadata'], "similarity": float(similarity)})

    similarities.sort(key=lambda x: x["similarity"], reverse=True)

    return [s["item"] for s in similarities if s["similarity"] >= min_similarity][:top_k]