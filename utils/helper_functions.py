import gc
import torch
import spacy

# Load spaCy model for text preprocessing
nlp = spacy.load("en_core_web_sm")

# Memory Cleanup Function
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Text Preprocessing
def preprocess_text(text):
    doc = nlp(text)
    return " ".join([token.text for token in doc if not token.is_stop])
