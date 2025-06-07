"""
This module uses a Hugging Face transformer-based Named Entity Recognition (NER) model
to extract entities (skills, job titles, orgs, etc.) from resumes and job descriptions.
It matches resumes to job postings by comparing extracted entities using Jaccard similarity.
"""

# Import Hugging Face's tokenizer and model loader, along with regex
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re

MODEL_NAME = "Jean-Baptiste/roberta-large-ner-english"  # Optional alternative NER model
# MODEL_NAME = "dslim/bert-base-NER" # Pre-trained BERT model fine-tuned for Named Entity Recognition (NER)
# MODEL_NAME = "ml6team/keyphrase-extraction-distilbert-inspec"

# Load the tokenizer and model using the model name
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

# Create a Hugging Face NER pipeline with the loaded model and tokenizer
# aggregation_strategy="simple" merges tokens that are part of the same entity (like "machine" + "learning")
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def extract_entities(text, entity_types=None):
    """
    Extract named entities from text using the pipeline
    If `entity_types` is provided, filter the results to include only those types
    """
    entities = ner_pipeline(text)
    if entity_types:
        return [e["word"] for e in entities if e["entity_group"] in entity_types]
    return [e["word"] for e in entities]

def extract_skills_and_roles(text):
    """
    Extract only specific types of entities (custom logic)
    For our use case, we assume "ORG", "JOB", "MISC", and "SKILL" are likely relevant
    """
    return extract_entities(text, entity_types=["ORG", "JOB", "MISC", "SKILL", "TECH", "TOOL"])
    # return extract_entities(text, entity_types=["KEYPHRASE"])

def estimate_experience_years(text):
    """
    Use regex to search for patterns like "5 years of experience" in the text
    Returns the **maximum** number found (assumes the most experience stated is the total)
    """
    matches = re.findall(r"(\d+)\s+(?:years?|yrs?)\s+of\s+experience", text.lower())
    return max(map(int, matches), default=0)

def match_jobs_by_entities(resume_text, job_texts, top_k=5):
    """
    Match a resume to a list of job descriptions based on extracted entities.
    """

    # Loop through job descriptions and calculate similarity
    resume_entities = set(extract_skills_and_roles(resume_text))
    matches = []

    for i, job in enumerate(job_texts):
        job_entities = set(extract_skills_and_roles(job))
        # job_entities = set(skill.strip().lower() for skill in job.split(",") if skill.strip())  # Simple split by comma

        # Compute Jaccard similarity: intersection over union
        if not resume_entities and not job_entities:
            score = 0 # Avoid divide-by-zero if both are empty
        else:
            score = len(resume_entities & job_entities) / len(resume_entities | job_entities)
        matches.append((i, score))

    # Sort by similarity descending and return top K matches
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:top_k]