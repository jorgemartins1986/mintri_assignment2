from rank_bm25 import BM25Okapi
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

def match_jobs_bm25(resume_text, job_descriptions, top_k=5):
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in job_descriptions]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_resume = word_tokenize(resume_text.lower())
    scores = bm25.get_scores(tokenized_resume)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(i, scores[i]) for i in top_indices]