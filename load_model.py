from settings import *
import torch
from sentence_transformers import SentenceTransformer
import os

def load_model():
    '''
    Loads the model given in the settings by either downloading it or loading it from cache
    It also loads it onto the gpu, if available.
    '''
    if len(os.listdir(MODEL_PATH)) == 0:
        print('Downloading model')
        model = SentenceTransformer(MODEL_NAME)
        print("Saved model to" + MODEL_PATH)
        model.save(MODEL_PATH) 
    else:
        print('Loading model from disk')
        model = SentenceTransformer(MODEL_PATH)
    
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print("Using device: " + device_str) 
    model.to(device)    
    return model