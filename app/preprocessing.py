# This module contains functions for preprocessing text data.
# The purpose is to clean the text to remove noise before feeding it to TF-IDF or Hugging Face models.

import re
import spacy
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

def clean_and_lemmatize(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)  # remove special chars
    doc = nlp(text.lower())
    return " ".join([
        token.lemma_ for token in doc
        if token.lemma_ not in stop_words and token.is_alpha
    ])