
"""
This script uses the Sentence Transformers library to embed a resume and job descriptions
and then calculates the cosine similarity between them.
It returns the top K job descriptions that are most similar to the resume.
This is a more advanced method for text matching,
leveraging pre-trained models to capture semantic meaning.
This method is more computationally expensive than TF-IDF,
but it can provide better results, especially for larger datasets or more complex text.
"""

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight, fast

def match_jobs_embed(resume_text, job_descriptions, top_k=5):
    emb_resume = model.encode(resume_text, convert_to_tensor=True)
    emb_jobs = model.encode(job_descriptions, convert_to_tensor=True)
    sim_scores = util.cos_sim(emb_resume, emb_jobs)[0]
    top_indices = sim_scores.argsort(descending=True)[:top_k]
    return [(i.item(), sim_scores[i].item()) for i in top_indices]