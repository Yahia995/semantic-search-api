from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class FitLevel(str, Enum):
    STRONG = "Strong Match"
    GOOD = "Good Match"
    PARTIAL = "Partial Match"
    WEAK = "Weak Match"


class JobPostingInput(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="Job title")
    company: str = Field(..., min_length=1, max_length=200, description="Company name")
    description: str = Field(..., min_length=10, max_length=10000, description="Full job description")
    location: Optional[str] = Field(None, description="Job location or 'Remote'")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class IndexJobsRequest(BaseModel):
    jobs: List[JobPostingInput] = Field(..., min_items=1, max_items=200)


class IndexJobsResponse(BaseModel):
    status: str
    jobs_indexed: int
    total_jobs: int
    elapsed_seconds: float


class MatchRequest(BaseModel):
    resume_text: str = Field(..., min_length=50, max_length=20000, description="Full resume text")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top matches to return")
    min_score: float = Field(default=0.2, ge=0.0, le=1.0, description="Minimum similarity score")


class SkillGap(BaseModel):
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)


class JobMatch(BaseModel):
    job_id: int
    title: str
    company: str
    location: Optional[str]
    similarity_score: float
    fit_level: FitLevel
    match_summary: str        
    skill_gap: SkillGap
    recommendation: str       


class MatchResponse(BaseModel):
    total_matches: int
    matches: List[JobMatch]
    elapsed_ms: float


class ListJobsResponse(BaseModel):
    total_jobs: int
    jobs: List[Dict[str, Any]]
