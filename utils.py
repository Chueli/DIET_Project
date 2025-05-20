from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

import string
import json
from settings import NUM_TOPICS, NUM_WORDS, DATA_PATH


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

