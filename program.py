import os
import re
import spacy
from collections import Counter
from math import sqrt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

file_paths = [r"C:\Users\Faisal\Desktop\plagiarism checker\document1.txt", r"C:\Users\Faisal\Desktop\plagiarism checker\document2.txt"]

nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
   
    text = text.lower()
    
  
    text = re.sub(r'[^\w\s]', '', text)
    
 
    tokens = word_tokenize(text)
    
  
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
 
    doc = nlp(text)
    entities = [entity.text for entity in doc.ents]
    
 
    preprocessed_text = ' '.join(tokens + entities)
    
    print("Preprocessed text:", preprocessed_text)
    
    return preprocessed_text


def calculate_tfidf_vectors(documents):
 
    vectorizer = TfidfVectorizer()
    
    tfidf_vectors = vectorizer.fit_transform(documents)
    
    return tfidf_vectors

def calculate_similarity(tfidf_vectors):
  
    similarity_matrix = cosine_similarity(tfidf_vectors, tfidf_vectors)
    
    print("Similarity matrix:", similarity_matrix)
    
    return similarity_matrix

def detect_plagiarism(similarity_matrix, threshold=0.8):
  
    plagiarism_pairs = []
    
    for i in range(len(similarity_matrix)):
        for j in range(i+1, len(similarity_matrix)):
            similarity_score = similarity_matrix[i][j]
            print(f"Similarity score between documents {i} and {j}: {similarity_score}")
            if similarity_score > threshold:
                plagiarism_pairs.append((i, j))
    
    return plagiarism_pairs

documents = [open(file_path, 'r').read() for file_path in file_paths]

preprocessed_documents = [preprocess_text(document) for document in documents]

tfidf_vectors = calculate_tfidf_vectors(preprocessed_documents)

similarity_matrix = calculate_similarity(tfidf_vectors)

plagiarism_pairs = detect_plagiarism(similarity_matrix)

if plagiarism_pairs:
    print("Plagiarism detected between the following documents:")
    for pair in plagiarism_pairs:
        print(f"Documents {pair[0]} and {pair[1]}")
else:
    print("No plagiarism detected.")