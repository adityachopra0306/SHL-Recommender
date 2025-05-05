import os
import re
import json
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
from dotenv import load_dotenv
import functools

nltk.data.path.append("nltk_data")
load_dotenv()

DF_PATH = os.getenv("DF_PATH", "data/df_clean.pkl")
BM25_PICKLE_PATH = os.getenv("BM25_PICKLE_PATH", "data/df_bm25_tokenized.pkl")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index.bin")
SBERT_MODEL_NAME = os.getenv("SBERT_MODEL_NAME", "all-mpnet-base-v2")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

PROMPT_TEMPLATE = """
For the following job description, extract a structured query suitable for retrieving relevant candidate assessments. The output should be optimized for semantic and lexical similarity matching in retrieval systems. Pay primary attention to technical skills used in the given query and include keywords, if any.

Include:
All relevant job duties and selection criteria.
Seniority level (e.g., entry level, mid, senior).
Whether it's technical or non-technical based on job responsibilities.
Test duration in minutes (a single integer) if stated; otherwise, set to -1.
Most relevant assessment categories from the following list:
  ['Ability & Aptitude', 'Biodata & Situational Judgement', 'Competencies', 'Development & 360', 'Assessment Exercises', 'Knowledge & Skills', 'Personality & Behavior', 'Simulations'].

Format: Return only a valid JSON object:
{{
  "duration": 0,
  "type": "technical" or "non-technical",
  "test_types": [],
  "query": "..."  // Detailed and precise query in natural language suitable for semantic (SBERT) retrieval and with keywords for lexical (BM25)
}}

Only return the JSON. No explanation or extra text.

QUERY:
{query}
"""

''' TOO MUCH RAM USAGE, now performed in start.sh
def ensure_nltk_resources():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

ensure_nltk_resources()
'''

STOP_WORDS = set(stopwords.words('english'))

@functools.lru_cache(maxsize=1)
def get_df():
    return pd.read_pickle(DF_PATH)

@functools.lru_cache(maxsize=1)
def get_bm25():
    df = pd.read_pickle(BM25_PICKLE_PATH)
    return BM25Okapi(df['bm_tokens'].tolist())

@functools.lru_cache(maxsize=1)
def get_faiss_index():
    return faiss.read_index(FAISS_INDEX_PATH)

@functools.lru_cache(maxsize=1)
def get_sbert_model():
    return SentenceTransformer(SBERT_MODEL_NAME)

@functools.lru_cache(maxsize=1)
def get_gemini_model():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-1.5-flash")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = word_tokenize(text)
    return [word for word in tokens if word.isalnum() and word not in STOP_WORDS]

def generate_and_parse_query(job_text, model):
    prompt = PROMPT_TEMPLATE.format(query=job_text)
    try:
        response = model.generate_content(prompt)
        content = response.text.strip()
        cleaned = re.sub(r"```json|```", "", content).strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print("JSON parse error. Content:\n", content)
    except Exception as e:
        print("Gemini error:", e)

    return {
        "duration": -1,
        "type": "non-technical",
        "test_types": [],
        "query": job_text
    }

def bm25_search(query, bm25, df, top_n=60):
    tokens = preprocess_text(query)
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return df.iloc[top_indices]

def semantic_search(query, model, index, df, top_k=60):
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(np.array(query_embedding).astype('float32'), top_k)
    filtered_indices = [i for i, d in zip(indices[0], distances[0]) if d > 0.35]
    return df.iloc[filtered_indices]

def rrf_fusion(df1, df2, rrf_k=20):
    if df1.empty and df2.empty:
        return pd.DataFrame()

    scores = {}
    for rank, url in enumerate(df1['url']):
        scores[url] = scores.get(url, 0) + 1 / (rrf_k + rank + 1)
    for rank, url in enumerate(df2['url']):
        scores[url] = scores.get(url, 0) + 1 / (rrf_k + rank + 1)

    ranked_urls = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_urls = [url for url, _ in ranked_urls[:10]]

    df = get_df()
    return df[df['url'].isin(top_urls)]

def filter_by_duration(df_subset, max_duration):
    try:
        max_duration = int(max_duration)
        if max_duration <= 0:
            return df_subset[~df_subset['duration'].isna()]
        return df_subset[(~df_subset['duration'].isna()) & (df_subset['duration'] <= max_duration)]
    except:
        return df_subset[~df_subset['duration'].isna()]

def filter_by_test_type(df_subset, target_types):
    def overlap(row_types, target):
        if isinstance(row_types, str):
            row_types = [t.strip() for t in row_types.split(",")]
        return len(set(row_types).intersection(set(target))) > 0
    if len(target_types) <= 0:
        return df_subset
    return df_subset[df_subset['test_types'].apply(lambda x: overlap(x, target_types))]

def get_recommendations(job_text):
    df = get_df()
    bm25 = get_bm25()
    sbert_model = get_sbert_model()
    faiss_index = get_faiss_index()
    gemini_model = get_gemini_model()

    parsed_query = generate_and_parse_query(job_text, gemini_model)
    if not parsed_query:
        return pd.DataFrame()

    bm_df = bm25_search(parsed_query['query'], bm25, df, top_n=60)
    bm_df = filter_by_duration(bm_df, parsed_query['duration'])
    bm_df = filter_by_test_type(bm_df, parsed_query['test_types'])

    sm_df = semantic_search(parsed_query['query'], sbert_model, faiss_index, df, top_k=60)
    sm_df = filter_by_duration(sm_df, parsed_query['duration'])
    sm_df = filter_by_test_type(sm_df, parsed_query['test_types'])

    return rrf_fusion(bm_df, sm_df)
