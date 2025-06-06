# This script uses TF-IDF to match resumes with job descriptions.
# It computes the TF-IDF vectors for the resume and job descriptions,
# and then calculates the cosine similarity between them.
# It returns the top K job descriptions that are most similar to the resume.
# This is a simple and effective method for text matching,
# especially when the dataset is not too large.
# It is less computationally expensive than using deep learning models like Sentence Transformers.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def match_jobs_tfidf(resume_text, job_descriptions, top_k=5):
    documents = [resume_text] + job_descriptions
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(documents)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]
    return [(i, similarities[i]) for i in top_indices]