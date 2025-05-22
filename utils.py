import io
import PyPDF2
import discord
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict

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

def compute_user_topics(note_embeddings, topic_embeddings, topic_strings, top_n=-1):
    """
    Compute the top topics for each note using dot product similarity
    
    Args:
        note_embeddings: The embeddings of the notes
        topic_embeddings: The embeddings of all possible topics
        topic_strings: The string labels of the topics
        top_n: Number of top topics to return per note
        
    Returns:
        list: List of top topics with their scores
    """
    similarities = util.dot_score(note_embeddings, topic_embeddings)
    results = []
    
    if top_n == -1:
        top_n = len(topic_strings)
    top_values, top_indices = torch.topk(similarities, k=top_n)
    # Get the matched topics for all notes
    for note_idx, (values, indices) in enumerate(zip(top_values, top_indices)):
        matched_topics = [(topic_strings[i], similarities[note_idx][i].item()) for i in indices]
        results.append(matched_topics)
    
    return results

def compute_author_similarity(author1_topics, author2_topics):
    """
    Compute similarity between two authors based on their topics and find common topics
    
    Args:
        author1_topics: List of (topic, score) tuples for author 1
        author2_topics: List of (topic, score) tuples for author 2
        
    Returns:
        tuple: (similarity_score, common_topics) where common_topics is a list of (topic, score1, score2) tuples
    """
    # create topic-score dictionaries for both students
    topics1 = defaultdict(float)
    topics2 = defaultdict(float)
    
    # aggregate scores for each topic across all notes
    for note_topics in author1_topics:
        for topic, score in note_topics:
            topics1[topic] += score
            
    for note_topics in author2_topics:
        for topic, score in note_topics:
            topics2[topic] += score
    
    # find common topics and compute similarity
    similarity = 0
    common_topics = []
    all_topics = set(topics1.keys()) | set(topics2.keys())
    
    for topic in all_topics:
        score1 = topics1[topic]
        score2 = topics2[topic]
        topic_similarity = score1 * score2
        similarity += topic_similarity
        
        # if both authors have this topic with significant scores
        if score1 > 0.1 and score2 > 0.1:  # threshold to consider a topic "significant"
            common_topics.append((topic, score1, score2))
    
    # sort common topics by combined score
    common_topics.sort(key=lambda x: x[1] + x[2], reverse=True)
    
    return similarity, common_topics

# Entirely generated by copilot to compute similarity scores for more than just a pair.
def compute_group_similarity(author_topics_list):
    """
    Compute similarity for a group of authors based on their topics
    
    Args:
        author_topics_list: List of author topics to compare
        
    Returns:
        tuple: (group_similarity_score, common_topics)
    """
    # Create topic-score dictionaries for all authors
    author_topic_scores = []
    for topics in author_topics_list:
        scores = defaultdict(float)
        for note_topics in topics:
            for topic, score in note_topics:
                scores[topic] += score
        author_topic_scores.append(scores)
    
    # Find topics common to all authors
    all_topics = set.intersection(*[set(scores.keys()) for scores in author_topic_scores])
    common_topics = []
    
    # Compute group similarity
    group_similarity = 0
    for topic in all_topics:
        scores = [scores[topic] for scores in author_topic_scores]
        if all(score > 0.1 for score in scores):  # All authors have significant interest
            avg_score = sum(scores) / len(scores)
            common_topics.append((topic, avg_score))
            group_similarity += avg_score
    
    common_topics.sort(key=lambda x: x[1], reverse=True)
    return group_similarity, common_topics

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