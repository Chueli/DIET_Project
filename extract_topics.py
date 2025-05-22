from sentence_transformers import SentenceTransformer, util
# PS. to make this work you also need pytorch installed.
import torch

from model_utils import *
from settings import diet_topics as topics
from utils import read_notes


# for the demo, we can jsut rely on a pre-defined set of topics and this is just a "random" list 
# but we can maybe tailor this ot the DiET course.
# topics = [
#     "Probability Distributions",
#     "Bayes' Theorem",
#     "Maximum Likelihood Estimation",
#     "Linear Regression",
#     "Decision Trees",
#     "Overfitting and Regularization",
#     "Neural Networks",
#     "Gradient Descent",
#     "Cross-Validation",
#     "Support Vector Machines"
# ]

model = load_model()

print("Computing topic embeddings")
topic_embeddings = get_embeddings(model, topics)


print("Computing note embeddings")
# example to test
# student_note = "I don't understand how overfitting affects decision trees."

student_notes_list = read_notes("notes.json") # list of student notes.
note_embeddings = get_embeddings(model, student_notes_list)


print("Computing similarities between notes and topics")
similarities = util.dot_score(note_embeddings, topic_embeddings)

#print(similarities)

# Get top 3 similarities and their indices
top_n = 3
results = []

for note_idx, similarity_vector in enumerate(similarities):
    top_values, top_indices = torch.topk(similarity_vector, k=top_n)
    matched_topics = [(topics[i], similarity_vector[i].item()) for i in top_indices]
    results.append({
        "note": student_notes_list[note_idx],
        "matched_topics": matched_topics
    })

# print results
for r in results:
    print(f"\nNote: {r['note']}")
    print("Top topics:")
    for topic, score in r['matched_topics']:
        print(f"  - {topic} ({score:.4f})")


## Later TODO: find common interests/topics ammong students based on their notes
"""
from collections import Counter

all_topic_matches = [topic for r in results for topic, _ in r["matched_topics"]]
topic_freq = Counter(all_topic_matches)

print("\nMost common topics:")
for topic, count in topic_freq.most_common():
    print(f"{topic}: {count} mentions")
"""