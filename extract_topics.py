from settings import *
from sentence_transformers import SentenceTransformer, util
import torch
# PS. to make this work you also need pytorch installed.

# for the demo, we can jsut rely on a pre-defined set of topics and this is just a "random" list 
# but we can maybe tailor this ot the DiET course.
topics = [
    "Probability Distributions",
    "Bayes' Theorem",
    "Maximum Likelihood Estimation",
    "Linear Regression",
    "Decision Trees",
    "Overfitting and Regularization",
    "Neural Networks",
    "Gradient Descent",
    "Cross-Validation",
    "Support Vector Machines"
]

print('Loading Model')
model = SentenceTransformer(modelPath)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.version.cuda)

model.to(device)

print("embedding topics")
topic_embeddings = model.encode(topics, normalize_embeddings=True)

print("embedding notes")
# example to test
student_note = "I don't understand how overfitting affects decision trees."
note_embedding = model.encode(student_note, normalize_embeddings=True)

print("computing sim")
similarities = util.dot_score(note_embedding, topic_embeddings)

#print(similarities)

# Get top 3 similarities and their indices
top_values, top_idx = torch.topk(similarities[0], k=3)

print("Matching topics:")
print([(topics[i], similarities[0][i].item()) for i in top_idx])
