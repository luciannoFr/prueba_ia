import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.urandom(24)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://www.formosa.gob.ar" 
TOXICITY_MODEL_NAME = "unitary/toxic-bert"
TOXICITY_THRESHOLD = 0.5

FORBIDDEN_WORDS = [
    "pelotudo", "boludo", "idiota", "estúpido", "imbécil", "cabrón", "hijo de puta",
    "concha", "mierda", "puto", "culiao", "forro", "carajo", "puta", "pelotuda",
    "boluda", "tarado", "tarada", "gil", "conchudo", "chupapija", "trolazo",
    "cagón", "cagona", "mierdoso", "pelotudazo", "ortiva", "orto", "reventado",
    "choto", "chota", "pajero", "pajera", "capo", "capo de mierda", "negro de mierda",
    "vago", "vaga", "hijueputa", "nderakore", "kyhyje", "mita'i", "pajagua",
    "kuña kue", "mbarete", "porombo", "japu", "kaigue", "ñemby","que me importa", "inutil", "asqueroso"
    "bobo", "payaso", "pendejo", "bobito", "la puta madre", "negro", "negra", "puta", "puto"
    
]

WHITELIST_WORDS = [
    "trámite", "solicitud", "documento", "ciudadano", "provincia", "formosa", "oficial","quiero hacerle", "me gustaria hacer"
]

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
KNOWLEDGE_BASE_FILE = 'data/tramites_knowledge_base.json'
TRAMITES_URLS_FILE = 'data/tramites_urls.json'
