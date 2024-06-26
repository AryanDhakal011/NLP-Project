import os
import fitz 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import streamlit as st
import numpy as np
from dotenv import load_dotenv
import faiss
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()
api_key = os.getenv("HUGGINGFACE_API_KEY")

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# PDF Processing
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Text Preprocessing
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)

    # Stopword removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

# Text Representation
def get_sentence_embeddings(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    return embeddings

# TF-IDF Vectorization
def vectorize_documents(documents):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix

# Retrieve Most Relevant Document
def retrieve_most_relevant_document(query, documents, tfidf_matrix, vectorizer):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix)
    most_similar_index = np.argmax(similarities)
    return documents[most_similar_index]

# Streamlit Application
st.title("Nepali ChatBot")
st.subheader("Ask any question related to Nepali Events, history, laws and Dates!")

# Hardcoded PDF folder path
pdf_folder_path = "data"

texts = []
for pdf_file in os.listdir(pdf_folder_path):
    if pdf_file.endswith(".pdf"):
        texts.append(extract_text_from_pdf(os.path.join(pdf_folder_path, pdf_file)))

if texts:
    processed_texts = [preprocess_text(text) for text in texts]

    vectorizer, tfidf_matrix = vectorize_documents(processed_texts)

    st.write("Ask your question:")
    query = st.text_input("")

    if query:
        relevant_document = retrieve_most_relevant_document(query, texts, tfidf_matrix, vectorizer)
        st.write("Most relevant document:")
        st.write(relevant_document)  # Displaying the relevant document in its original form
else:
    st.write("No PDF files found in the data folder.")
