import json
import os
import logging
from config import KNOWLEDGE_BASE_FILE, TRAMITES_URLS_FILE

logger = logging.getLogger(__name__)

def load_knowledge_base():
    if os.path.exists(KNOWLEDGE_BASE_FILE):
        try:
            with open(KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Loaded {len(data)} entries from knowledge base.")
                return data
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from knowledge base file '{KNOWLEDGE_BASE_FILE}': {e}. Returning empty list.")
            return []
        except Exception as e:
            logger.error(f"Unexpected error loading knowledge base from '{KNOWLEDGE_BASE_FILE}': {e}. Returning empty list.")
            return []
    logger.info(f"Knowledge base file not found at '{KNOWLEDGE_BASE_FILE}'. Starting with empty base.")
    return []

def save_knowledge_base(data):
    os.makedirs(os.path.dirname(KNOWLEDGE_BASE_FILE), exist_ok=True)
    try:
        with open(KNOWLEDGE_BASE_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"Knowledge base saved to '{KNOWLEDGE_BASE_FILE}' with {len(data)} entries.")
    except IOError as e:
        logger.error(f"Error saving knowledge base file to '{KNOWLEDGE_BASE_FILE}': {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving knowledge base to '{KNOWLEDGE_BASE_FILE}': {e}")

def load_tramites_urls():
    if os.path.exists(TRAMITES_URLS_FILE):
        try:
            with open(TRAMITES_URLS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Loaded {len(data)} tramites URLs.")
                return data
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from tramites URLs file '{TRAMITES_URLS_FILE}': {e}. Returning empty dict.")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error loading tramites URLs from '{TRAMITES_URLS_FILE}': {e}. Returning empty dict.")
            return {}
    logger.info(f"Tramites URLs file not found at '{TRAMITES_URLS_FILE}'. Returning empty dict.")
    return {}

def get_all_urls_to_scrape():
    urls_data = load_tramites_urls()
    return [info["url"] for info in urls_data.values()]
