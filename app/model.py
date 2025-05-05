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

load_dotenv()

# ---- Constants ----
DF_PATH = os.getenv("DF_PATH")
BM25_PICKLE_PATH = os.getenv("BM25_PICKLE_PATH")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH")

SBERT_MODEL_NAME = "all-mpnet-base-v2"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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

def ensure_nltk_resources():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

ensure_nltk_resources()
STOP_WORDS = set(stopwords.words('english'))

# ---- Globals ----
_df = None
_bm25 = None
_faiss_index = None
_sbert_model = None
_gemini_model = None

# ---- Loaders ----
def load_data(df_path=DF_PATH):
    global _df
    _df = pd.read_pickle(df_path)
    return _df

def load_bm25(df_path=BM25_PICKLE_PATH):
    df = pd.read_pickle(df_path)
    bm25 = BM25Okapi(df['bm_tokens'].tolist())
    return df, bm25

def load_semantic_index(index_path=FAISS_INDEX_PATH, model_name=SBERT_MODEL_NAME):
    index = faiss.read_index(index_path)
    model = SentenceTransformer(model_name)
    return model, index

def init_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

# ---- Utilities ----
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
        print(content)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print("JSON parse error. Content:\n", content)
    except Exception as e:
        print("Gemini error:", e)

    # Fallback defaults
    return {
        "duration": -1,
        "type": "technical",
        "test_types": ['Ability & Aptitude', 'Knowledge & Skills'],
        "query": job_text
    }                       # USED FALLBACK

# ---- Retrieval Methods ----
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
    if not df1.empty:
        for rank, url in enumerate(df1['url']):
            scores[url] = scores.get(url, 0) + 1 / (rrf_k + rank + 1)
    if not df2.empty:
        for rank, url in enumerate(df2['url']):
            scores[url] = scores.get(url, 0) + 1 / (rrf_k + rank + 1)

    ranked_urls = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_urls = [url for url, _ in ranked_urls[:10]]
    
    return _df[_df['url'].isin(top_urls)]


# ---- Filters ----
def filter_by_duration(df_subset, max_duration):
    try:
        max_duration = int(max_duration)
        if max_duration <= 0:
            return df_subset[(~df_subset['duration'].isna())]
        return df_subset[(~df_subset['duration'].isna()) & (df_subset['duration'] <= max_duration)]
    except:
        return df_subset[(~df_subset['duration'].isna())]

def filter_by_test_type(df_subset, target_types):
    def overlap(row_types, target):
        if isinstance(row_types, str):
            row_types = [t.strip() for t in row_types.split(",")]
        return len(set(row_types).intersection(set(target))) > 0
    return df_subset[df_subset['test_types'].apply(lambda x: overlap(x, target_types))]

# ---- Main Recommendation Function ----
def get_recommendations(job_text, df, bm25, sbert_model, faiss_index, gemini_model):
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

# ---- Test Run ----
if __name__ == "__main__":
    result_df = get_recommendations("""ICICI Bank Assistant Admin,
Experience required 0-2 years, test
should be 30-40 mins long""")
    print(result_df[['title', 'url', 'duration', 'test_types']])

    result_df = get_recommendations("""    KEY RESPONSIBITILES:
Manage the sound-scape of the station through appropriate
creative and marketing
interventions to Increase or
Maintain the listenership
Acts as an interface between
Programming & sales team, thereby
supporting the sales team by
providing creative inputs in order to
increase the overall ad spends by
clients
Build brand Mirchi by ideating fresh
programming initiatives on air
campaigns, programming led onground events & new properties to
ensure brand differentiation & thus
increase brand recall at station level
Invest time in local RJs to grow &
develop them as local celebrities
Through strong networking, must
focus on identifying the best of local
talent and ensure to bring the
creative minds from the market on
board with Mirchi
Build radio as a category for both
listeners & advertisers
People Management
Identifying the right talent and
investing time in developing them
by frequent feedback on their
performance
Monitor, Coach and mentor team
members on a regular basis
Development of Jocks as per
guidelines
Must have an eye to spot the local
talent to fill up vacancies locally
TECHNICAL SKILLS &
QUALIFICATION REQUIRED:
Graduation / Post Graduation (Any
specialisation) with 8 -12 years of
relevant experience
Experience in digital content
conceptualisation
Strong branding focus
Must be well-read in variety of
areas and must keep up with the
latest events in the city / cluster /
country
Must know to read, write & speak
English
PERSONAL ATTRIBUTES:
Excellent communication skills
Good interpersonal skills
People management
Suggest me some tests for the
above jd. The duration should be at
most 90 mins""")
    print(result_df[['title', 'url', 'duration', 'remote_support']])