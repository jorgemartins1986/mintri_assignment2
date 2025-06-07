# 🧠 Intelligent Job Description Matching & Skill Extraction

This project builds a Flask-based intelligent matching system that extracts skills and experience from uploaded resumes (PDFs) and matches them to relevant job postings using both TF-IDF, BM25 and transformer-based similarity models.

---

## 🚀 Getting Started

### 🔧 Prerequisites

You need the following tools installed:

- [Docker] (https://www.docker.com/)
- [VSCode] (https://code.visualstudio.com/)
- [Dev Containers Extension for VSCode] (https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- [Git] (https://git-scm.com/)
- [Jobs Dataset] (https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024/)
- [Resumes Dataset] (https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset/)

---

### 📦 Recommended VSCode Extensions

Inside `.devcontainer/devcontainer.json`, the following extensions are auto-installed:

- Python (`ms-python.python`)
- Jupyter (`ms-toolsai.jupyter`)
- Docker (`ms-azuretools.vscode-docker`)

---

### 🐳 Running in a Dev Container

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/intelligent-matching.git
   cd intelligent-matching
   ```

2. Copy Jobs dataset, csv files, to folder data/jobs/..

3. Copy Resumes dataset, folders with pdf files, to folder data/resumes/..  

2. Open the folder in **VSCode**.

3. Press `F1` → **“Dev Containers: Reopen in Container”**

This will:
- Build the container
- Install all Python dependencies
- Auto-download NLTK stopwords
- Set up the Jupyter kernel

---

## 📂 Project Structure

- `app/` — core Python logic (PDF parsing, preprocessing, matching)
- `dataset_preprocessing/` — notebooks for cleaning datasets
- `uploads/` — uploaded resumes (PDF)
- `Dockerfile` / `docker-compose.yml` — container config
- `.devcontainer/` — VSCode dev environment

---

## 🧪 Features Implemented

- ✅ Resume text extraction from PDFs using PyMuPDF
- ✅ Skill extraction using Hugging Face transformer-based NER models
- ✅ Job-resume similarity matching using:
  - TF-IDF + cosine similarity
  - BM25 (via rank_bm25)
  - Sentence-BERT (MiniLM)
  - Transformer-based entity matching (NER + Jaccard)
- ✅ Score normalization for fair comparison across all methods

---

## 🛠️ To Do Next

- [ ] Add Flask-based UI for resume uploads and job matches
- [ ] Integrate result ranking display
- [ ] Evaluate system using known match labels or qualitative review

---

## 🤝 Contributions

This repo is shared among academic collaborators. Feel free to fork or clone for learning and improvements.

---

## 📄 License

MIT License (or your preferred one)
