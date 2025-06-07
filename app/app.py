from flask import Flask, render_template, request
import os
import pandas as pd
import time
from werkzeug.utils import secure_filename

from utils import extract_text_from_pdf
from matchers.tfidf_matcher import match_jobs_tfidf
from matchers.bm25_matcher import match_jobs_bm25
from matchers.embed_matcher import match_jobs_embed
from matchers.transformers_extractor import match_jobs_by_entities

from flask import jsonify

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
TOP_K = 5
NUM_SAMPLES = 2000
JOB_CSV_PATH = "../dataset_preprocessing/merged_job_data.csv"

def normalize_scores(matches):
    scores = [score for _, score in matches]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [(int(i), 0.0) for i, _ in matches]
    return [(int(i), float((score - min_s) / (max_s - min_s))) for i, score in matches]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_resume():
    file = request.files.get('resume')
    if not file or not file.filename.endswith('.pdf'):
        return jsonify({"error": "Invalid file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)

    resume_text = extract_text_from_pdf(filepath)
    return jsonify({"resume_text": resume_text})

def load_sample_jobs():
    df = pd.read_csv(JOB_CSV_PATH).dropna(subset=["job_title", "job_skills"])
    df_sample = df.sample(n=NUM_SAMPLES, random_state=42)
    job_texts = (df_sample["job_title"].fillna("") + " " + df_sample["job_skills"].fillna("")).tolist()
    return job_texts

@app.route('/match/tfidf', methods=['POST'])
def match_tfidf():
    resume_text = request.json.get("resume_text")
    if not resume_text:
        return jsonify({"error": "Missing resume text"}), 400

    job_texts = load_sample_jobs()
    start = time.time()
    matches = match_jobs_tfidf(resume_text, job_texts, top_k=TOP_K)
    elapsed = time.time() - start

    return jsonify({
        "matches": [(int(i), float(s), str(job_texts[i][:150])) for i, s in normalize_scores(matches)],
        "time": elapsed
    })

@app.route('/match/bm25', methods=['POST'])
def match_bm25():
    resume_text = request.json.get("resume_text")
    if not resume_text:
        return jsonify({"error": "Missing resume text"}), 400

    job_texts = load_sample_jobs()
    start = time.time()
    matches = match_jobs_bm25(resume_text, job_texts, top_k=TOP_K)
    elapsed = time.time() - start

    return jsonify({
        "matches": [(int(i), float(s), str(job_texts[i][:150])) for i, s in normalize_scores(matches)],
        "time": elapsed
    })

@app.route('/match/bert', methods=['POST'])
def match_bert():
    resume_text = request.json.get("resume_text")
    if not resume_text:
        return jsonify({"error": "Missing resume text"}), 400

    job_texts = load_sample_jobs()
    start = time.time()
    matches = match_jobs_embed(resume_text, job_texts, top_k=TOP_K)
    elapsed = time.time() - start

    return jsonify({
        "matches": [(int(i), float(s), str(job_texts[i][:150])) for i, s in normalize_scores(matches)],
        "time": elapsed
    })

@app.route('/match/ner', methods=['POST'])
def match_ner():
    resume_text = request.json.get("resume_text")
    if not resume_text:
        return jsonify({"error": "Missing resume text"}), 400

    job_texts = load_sample_jobs()
    start = time.time()
    matches = match_jobs_by_entities(resume_text, job_texts, top_k=TOP_K)
    elapsed = time.time() - start

    return jsonify({
        "matches": [(int(i), float(s), str(job_texts[i][:150])) for i, s in normalize_scores(matches)],
        "time": elapsed
    })

if __name__ == "__main__":
    app.run(debug=True)