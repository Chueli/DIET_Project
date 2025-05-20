import sys
MODEL_NAME = 'all-MiniLM-L6-v2'
MODEL_PATH = (".\model\\" + MODEL_NAME)  if (sys.platform == "win32") else ("./model" + MODEL_NAME)
NUM_TOPICS = 10
NUM_WORDS = 4