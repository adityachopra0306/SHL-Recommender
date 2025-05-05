from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from .model import (
    get_recommendations,
    load_data,
    load_bm25,
    load_semantic_index,
    init_gemini,
    GEMINI_API_KEY,
)

# Lifespan to preload resources
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Preloading models and data...")
    app.state.df = load_data()
    _, app.state.bm25 = load_bm25()
    app.state.sbert_model, app.state.faiss_index = load_semantic_index()
    app.state.gemini_model = init_gemini(GEMINI_API_KEY)
    print("Backend ready.")
    yield

app = FastAPI(title="SHL Assessment Recommendation API", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schemas
class RecommendationRequest(BaseModel):
    query: str

class AssessmentItem(BaseModel):
    url: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]

class FullAssessmentItem(BaseModel):
    title: str
    url: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]

# Routes
@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/recommend", response_model=List[AssessmentItem])
def recommend_assessments(request: RecommendationRequest):
    job_text = request.query
    df = get_recommendations(
        job_text,
        app.state.df,
        app.state.bm25,
        app.state.sbert_model,
        app.state.faiss_index,
        app.state.gemini_model
    )

    return [
        {
            "url": row["url"],
            "adaptive_support": "yes" if bool(row.get("adaptive_support", False)) else "no",
            "description": row["description"],
            "duration": int(row.get("duration", -1)),
            "remote_support": "yes" if bool(row.get("remote_support", False)) else "no",
            "test_type": row.get("test_types", []),
        }
        for _, row in df.iterrows()
    ]

@app.post("/recommend_full", response_model=List[FullAssessmentItem])
def recommend_full(request: RecommendationRequest):
    job_text = request.query
    df = get_recommendations(
        job_text,
        app.state.df,
        app.state.bm25,
        app.state.sbert_model,
        app.state.faiss_index,
        app.state.gemini_model
    )

    return [
        {
            "title": row.get("title", "Untitled"),
            "url": row["url"],
            "adaptive_support": "yes" if bool(row.get("adaptive_support", False)) else "no",
            "description": row["description"],
            "duration": int(row.get("duration", -1)),
            "remote_support": "yes" if bool(row.get("remote_support", False)) else "no",
            "test_type": row.get("test_types", []),
        }
        for _, row in df.iterrows()
    ]

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
