import os
import pandas as pd

from utils import extract_text_from_pdf
from tfidf_matcher import match_jobs_tfidf
from bm25_matcher import match_jobs_bm25
from transformers_extractor import match_jobs_by_entities
from embed_matcher import match_jobs_embed

# --- Configuration ---
RESUME_PATH = "../data/resumes/INFORMATION-TECHNOLOGY/10089434.pdf"
JOB_CSV_PATH = "../dataset_preprocessing/merged_job_data.csv"
NUM_SAMPLES = 2000
TOP_K = 5

def normalize_scores(matches):
    scores = [score for _, score in matches]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [(i, 0.0) for i, _ in matches]  # edge case
    return [(i, (score - min_s) / (max_s - min_s)) for i, score in matches]

def print_matches(title, matches, job_texts):
    print(f"\n🔎 Top matches using {title}:\n")
    for idx, score in matches:
        preview = job_texts[idx][:150].replace("\n", " ")
        print(f"[Score: {score:.2f}] Job #{idx} – {preview}...")

def main():
    # 1. Extract resume text
    print(f"📄 Extracting resume from: {RESUME_PATH}")
    resume_text = extract_text_from_pdf(RESUME_PATH)
    print(f"✅ Extracted {len(resume_text)} characters.")

    # 2. Load and prepare job data
    print(f"📊 Loading job data from: {JOB_CSV_PATH}")
    df = pd.read_csv(JOB_CSV_PATH).dropna(subset=["job_title", "job_skills"])
    df_sample = df.sample(n=NUM_SAMPLES, random_state=42)
    job_texts = (df_sample["job_title"].fillna("") + " " +
                 df_sample["job_skills"].fillna("")).tolist()

    # 3. TF-IDF matching
    tfidf_matches = match_jobs_tfidf(resume_text, job_texts, top_k=TOP_K)
    print_matches("TF-IDF", normalize_scores(tfidf_matches), job_texts)

    # 4. BM25 matching
    bm25_matches = match_jobs_bm25(resume_text, job_texts, top_k=TOP_K)
    print_matches("BM25", normalize_scores(bm25_matches), job_texts)

    # 5. Sentence-BERT (semantic similarity)
    embed_matches = match_jobs_embed(resume_text, job_texts, top_k=TOP_K)
    print_matches("Sentence-BERT", normalize_scores(embed_matches), job_texts)

    # 6. Transformer-based entity matcher
    transformer_matches = match_jobs_by_entities(resume_text, job_texts, top_k=TOP_K)
    print_matches("Transformer (NER)", normalize_scores(transformer_matches), job_texts)

if __name__ == "__main__":
    main()