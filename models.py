import torch
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import TOXICITY_MODEL_NAME, FORBIDDEN_WORDS, WHITELIST_WORDS, TOXICITY_THRESHOLD

logger = logging.getLogger(__name__)

try:
    tokenizer = AutoTokenizer.from_pretrained(TOXICITY_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(TOXICITY_MODEL_NAME)
    logger.info(f"Loaded toxicity model: {TOXICITY_MODEL_NAME}")
except Exception as e:
    logger.error(f"Error loading toxicity model {TOXICITY_MODEL_NAME}: {e}")
    tokenizer = None
    model = None

def detectar_toxicidad(texto):
    """
    Detects if a text is toxic using a pre-trained model and a list of forbidden words.
    Incorporates a whitelist to override detection for specific terms.
    """
    if not texto:
        return False, "Texto vacío"

    texto_lower = texto.lower()

    for word in WHITELIST_WORDS:
        if word in texto_lower:
            return False, "Contiene palabra en la lista blanca"

    for word in FORBIDDEN_WORDS:
        if word in texto_lower:
            return True, f"Contiene palabra prohibida: '{word}'"

    if tokenizer and model:
        try:
            inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=512)
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits)

            is_toxic_by_model = probs[0, 0].item() > TOXICITY_THRESHOLD

            if is_toxic_by_model:
                return True, f"Detectado como tóxico por el modelo (probabilidad de toxic: {probs[0, 0].item():.2f})"

        except Exception as e:
            logger.error(f"Error al ejecutar el modelo de toxicidad: {e}")
            return False, "Error en el modelo de toxicidad"
    else:
        logger.warning("Toxicity model not loaded, skipping AI-based toxicity detection. Relying only on black/whitelist.")

    return False, "No tóxico"