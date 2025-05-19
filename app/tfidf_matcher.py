# This script uses TF-IDF to match resumes with job descriptions.
# It computes the TF-IDF vectors for the resume and job descriptions,
# and then calculates the cosine similarity between them.
# It returns the top K job descriptions that are most similar to the resume.
# This is a simple and effective method for text matching,
# especially when the dataset is not too large.
# It is less computationally expensive than using deep learning models like Sentence Transformers.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def match_jobs_tfidf(resume_text, job_descriptions, top_k=5):
    docs = [resume_text] + job_descriptions
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)
    sim_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    ranked_indices = sim_scores.argsort()[::-1][:top_k]
    return [(i, sim_scores[i]) for i in ranked_indices]