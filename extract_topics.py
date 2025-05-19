from sentence_transformers import SentenceTransformer, util
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

print('Downloading model')
model = SentenceTransformer('all-MiniLM-L6-v2')
print("embedding topics")
topic_embeddings = model.encode(topics, normalize_embeddings=True)

print("embedding notes")
# example to test
student_note = "I don't understand how overfitting affects decision trees."
note_embedding = model.encode(student_note, normalize_embeddings=True)

print("computing sim")
similarities = util.dot_score(note_embedding, topic_embeddings)

# match to top-n topics
import numpy as np
top_idx = np.argsort(similarities[0])[::-1][:3]
[(topics[i], similarities[0][i].item()) for i in top_idx]
