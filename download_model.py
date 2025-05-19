from settings import *
from sentence_transformers import SentenceTransformer

print('Downloading model')
model = SentenceTransformer(modelName)

print("Saved model to" + modelPath)
model.save(modelPath)