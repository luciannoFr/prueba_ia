import os
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

from config import KNOWLEDGE_BASE_FILE, EMBEDDING_MODEL_NAME

model = SentenceTransformer(EMBEDDING_MODEL_NAME)

EMBEDDINGS_FILE = "data/tramites_embeddings.json"


def crear_embeddings():
    """Genera embeddings para todos los trámites y los guarda."""
    with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as f:
        base = json.load(f)

    embeddings = []

    for tramite in base:
        texto = f"{tramite['data'].get('titulo', '')}. {tramite['data'].get('descripcion', '')}"
        emb = model.encode(texto, convert_to_numpy=True).tolist()
        embeddings.append({
            "url": tramite["url"],
            "titulo": tramite['data'].get("titulo", ""),
            "embedding": emb
        })

    with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, indent=2, ensure_ascii=False)

    print(f"*** Embeddings generados para {len(embeddings)} trámites.")


def buscar_tramite_por_embedding(pregunta, top_k=1):
    ...
    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        base_emb = json.load(f)

    with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as f:
        base = json.load(f)

    pregunta_emb = model.encode(pregunta, convert_to_numpy=True)
    pregunta_emb = torch.tensor(pregunta_emb).float()

    similitudes = []
    for tramite in base_emb:
        emb_tramite = torch.tensor(np.array(tramite["embedding"])).float()
        score = util.cos_sim(pregunta_emb, emb_tramite).item()
        similitudes.append((score, tramite["url"]))

    similitudes.sort(reverse=True)
    mejores = similitudes[:top_k]

    resultados = [t["data"] for t in base if t["url"] in [url for _, url in mejores]]
    return resultados

