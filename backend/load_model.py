from settings import *
from sentence_transformers import SentenceTransformer
import os

def load_model():
    if len(os.listdir(MODEL_PATH)) == 0:
        print('Downloading model')
        model = SentenceTransformer(MODEL_NAME)
        print("Saved model to" + MODEL_PATH)
        model.save(MODEL_PATH)
        return model
    else:
        print('Loading model from disk')
        return SentenceTransformer(MODEL_PATH)