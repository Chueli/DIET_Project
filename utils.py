import io
import PyPDF2
import discord
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

import string
import json
from settings import *
from settings import diet_topics as topics

import torch
from sentence_transformers import SentenceTransformer, util
import os


def load_model():
    '''
    Loads the model given in the settings by either downloading it or loading it from cache
    It also loads it onto the gpu, if available.
    '''
    
    # check if model file exists in dir
    if not os.path.exists(MODEL_PATH) or not os.path.isdir(MODEL_PATH) or len(os.listdir(MODEL_PATH)) == 0:
        print('Downloading model')
        model = SentenceTransformer(MODEL_NAME)
        print(f"Saving model to {MODEL_PATH}")
        model.save(MODEL_PATH)
    else:
        print('Loading model from disk')
        model = SentenceTransformer(MODEL_PATH)
    
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print("Using device: " + device_str) 
    model.to(device)    
    return model

def get_embeddings(model, texts):
    '''
    Given a list of texts, it returns the embeddings for each text
    '''
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings

def read_notes(file_name):
    """
    Read notes from a json file and return them as a list of strings.
    
    Args:
        filename (str): name of the json file containing notes in the /data/ directory
        
    Returns:
        list: List of strings, where each string is a note
    """

    json_path = f'{DATA_PATH}{file_name}'
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            # extract just the 'note' field from each dictionary in the list
            return [item['note'] for item in data]
    except FileNotFoundError:
        print(f"Error: File not found at {json_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_path}")
        return []
    except KeyError:
        print(f"Error: Notes in {json_path} don't have the expected 'note' field")
        return []


def extract_topics(text, num_topics=NUM_TOPICS, num_words=NUM_WORDS):
    """
    Extract main topics from text using TF-IDF and NMF
    
    Args:
        text (str): The text to analyze
        num_topics (int): Number of topics to extract
        num_words (int): Number of words per topic
        
    Returns:
        list: List of topic strings
    """
    # Clean and preprocess text
    text = text.lower()
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    filtered_tokens = [word for word in tokens if word not in stop_words and word not in punctuation and len(word) > 2]
    
    # Check if we have enough content to analyze
    if len(filtered_tokens) < 20:
        return ["Not enough content to analyze properly."]
    
    # Prepare documents for TF-IDF (treat each paragraph as a document)
    paragraphs = [p for p in text.split('\n') if p.strip()]
    
    # Adjust num_topics if we don't have enough paragraphs
    if len(paragraphs) < num_topics:
        num_topics = max(1, len(paragraphs) // 2)
    
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(paragraphs)
    
    # Check if we have enough features
    if tfidf_matrix.shape[1] < num_topics:
        num_topics = max(1, tfidf_matrix.shape[1] // 2)
    
    # Apply NMF for topic modeling
    nmf_model = NMF(n_components=num_topics, random_state=42)
    nmf_model.fit(tfidf_matrix)
    
    # Get the top words for each topic
    feature_names = vectorizer.get_feature_names_out()
    topic_results = []
    
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_words_idx = topic.argsort()[:-num_words-1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topic_results.append(", ".join(top_words))
    
    return topic_results

def extract_topics_lm():
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

async def extract_text(message: discord.Message):
    text = ""
    text += message.content + " "
    if text.startswith("!extract ") or text.startswith("!topics "):
        text = text.split(' ', 1)[1]
    elif text.startswith("!extract\n") or text.startswith("!topics\n"):
        text = text.split('\n', 1)[1]
    for attachment in message.attachments:
        # Check if the attachment is a PDF
        if attachment.filename.endswith('.pdf'):
            # Download the PDF file
            pdf_bytes = await attachment.read()
            pdf_file = io.BytesIO(pdf_bytes)
            
            # Read PDF content
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            
            # Extract text from all pages
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                newtext = page.extract_text()
                text += newtext + "\n"
        if attachment.filename.endswith('.txt'):
            # Download the TXT file
            txt_bytes = await attachment.read()
            txt_content = txt_bytes.decode('utf-8', errors='ignore')  # decode to string
            text += txt_content.strip() + "\n"
    text.strip()
    return text