from backend.settings import *
from sentence_transformers import SentenceTransformer

print('Downloading model')
model = SentenceTransformer(MODEL_NAME)

print("Saved model to" + MODEL_PATH)
model.save(MODEL_PATH)