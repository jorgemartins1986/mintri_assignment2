# 🧠 Intelligent Job Description Matching & Skill Extraction

This project is an intelligent job matching system that extracts skills and experience from resumes (PDFs) and matches them to job listings using four different strategies:
- ✅ TF-IDF + Cosine Similarity
- ✅ BM25 Ranking
- ✅ Sentence-BERT Embeddings
- ✅ Transformer-based Named Entity Recognition (NER)

It is powered by **Flask**, supports **step-by-step asynchronous matching**, and runs fully containerized with **VSCode Dev Containers**.

---

## 🚀 Getting Started

### 🔧 Prerequisites

Ensure the following are installed:

- [Docker](https://www.docker.com/)
- [VSCode](https://code.visualstudio.com/)
- [VSCode Dev Containers Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- [Git](https://git-scm.com/)

Also required:

- [Jobs Dataset (CSV)](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024/)
- [Resumes Dataset (PDF)](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset/)

---

### 📦 Recommended VSCode Extensions (Auto-installed)

- Python (`ms-python.python`)
- Jupyter (`ms-toolsai.jupyter`)
- Docker (`ms-azuretools.vscode-docker`)

---

## 🐳 Run in Dev Container

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/mintri_assignment2.git
   cd mintri_assignment2
   ```

2. Add data:
   - Place CSV job listings in `data/jobs/`
   - Place resume PDFs in `data/resumes/`

3. Open the folder in **VSCode**.

4. Press `F1` → “Dev Containers: Reopen in Container”

This will:
- Build and start the dev environment
- Install Python dependencies
- Enable Jupyter, Flask, and NLTK

---

## 📂 Project Structure

```
.
├── app/                       # Flask app logic and matchers
│   ├── tfidf_matcher.py
│   ├── bm25_matcher.py
│   ├── embed_matcher.py
│   ├── transformers_extractor.py
│   └── utils.py
├── dataset_preprocessing/    # Cleaning and exploration notebooks
├── templates/                # index.html with Bootstrap + JS
├── uploads/                  # Temporarily stores uploaded PDFs
├── .devcontainer/            # VSCode container config
├── requirements.txt
└── app.py                    # Main Flask application
```

---

## 🎯 Features

- ✅ PDF Resume Upload via Web Interface
- ✅ Text Extraction (PyMuPDF)
- ✅ Real-time progress updates using JavaScript + Fetch API
- ✅ Step-by-step matcher progress: TF-IDF, BM25, BERT, NER
- ✅ Score normalization across matchers
- ✅ Bootstrap UI with progress bars and result cards

---

## 📈 Matching Methods

| Method             | Description                                     |
|--------------------|-------------------------------------------------|
| TF-IDF             | Term Frequency-Inverse Document Frequency       |
| BM25               | Probabilistic retrieval (via `rank_bm25`)       |
| Sentence-BERT      | Embedding-based similarity (MiniLM)             |
| Transformer (NER)  | Jaccard similarity over extracted entities      |

---

## 🧪 How it Works

1. User uploads a resume (PDF)
2. Text is extracted and displayed as “Extracting resume...”
3. Each matcher runs in sequence via AJAX:
   - TF-IDF
   - BM25
   - Sentence-BERT
   - NER
4. UI displays:
   - Top 5 job matches per method
   - Normalized scores
   - Execution time per method

---

## ✅ Example Output

```
🔹 TF-IDF:      [Score: 0.32] Job #5 - Data Analyst, SQL, Tableau, Python...
🔹 BM25:        [Score: 0.89] Job #32 - ML Engineer, scikit-learn, PyTorch...
🔹 Sentence-BERT: [Score: 0.76] Job #18 - AI Researcher, BERT, TensorFlow...
🔹 Transformer (NER): [Score: 0.55] Job #23 - DevOps, AWS, Docker, Kubernetes...
```

---

## 🛠️ Future Improvements

- [ ] Add authentication and save history per user
- [ ] Enable manual skill feedback and fine-tuning
- [ ] Integrate more advanced transformer models

---

## 📄 License

MIT License
