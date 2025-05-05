from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .model import get_recommendations

app = FastAPI(title="SHL Assessment Recommendation API")

# CORS setup
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

    recommendations = get_recommendations(job_text)

    return [
        {
            "url": row["url"],
            "adaptive_support": "yes" if bool(row.get("adaptive_support", False)) else "no",
            "description": row["description"],
            "duration": int(row.get("duration", -1)),
            "remote_support": "yes" if bool(row.get("remote_support", False)) else "no",
            "test_type": row.get("test_types", []),
        }
        for _, row in recommendations.iterrows()
    ]

@app.post("/recommend_full", response_model=List[FullAssessmentItem])
def recommend_full(request: RecommendationRequest):
    job_text = request.query

    recommendations = get_recommendations(job_text)

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
        for _, row in recommendations.iterrows()
    ]

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)