import io
import PyPDF2
import discord
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

import string
from settings import NUM_TOPICS, NUM_WORDS


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

async def extract_text(message: discord.Message):
    text = ""
    text += message.content + " "
    if text.startswith("!extract ") or text.startswith("!topics "):
        text = text.split(' ', 1)[1]
    elif text.startswith("!extract\n") or text.startswith("!topics\n"):
        text = text.split('\n', 1)[1]
    if len(message.attachments) != 0:
        attachment = message.attachments[0]
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