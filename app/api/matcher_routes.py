import time
from fastapi import APIRouter, HTTPException, status

from app.core.matcher_engine import MatcherEngine
from app.models.matcher_schemas import (
    IndexJobsRequest,
    IndexJobsResponse,
    MatchRequest,
    MatchResponse,
    ListJobsResponse,
)

matcher_router = APIRouter()

matcher_engine: MatcherEngine = None


def set_matcher_engine(engine: MatcherEngine):
    global matcher_engine
    matcher_engine = engine


def _require_engine():
    if not matcher_engine:
        raise HTTPException(status_code=500, detail="Matcher engine not initialized")


@matcher_router.post(
    "/jobs",
    response_model=IndexJobsResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Index job postings",
    description="Add one or more job postings to the semantic index.",
)
async def index_jobs(request: IndexJobsRequest):
    _require_engine()
    try:
        jobs = [j.model_dump() for j in request.jobs]
        result = matcher_engine.index_jobs(jobs)
        return IndexJobsResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")


@matcher_router.get(
    "/jobs",
    response_model=ListJobsResponse,
    summary="List indexed jobs",
)
async def list_jobs():
    _require_engine()
    jobs = matcher_engine.list_jobs()
    return ListJobsResponse(total_jobs=len(jobs), jobs=jobs)


@matcher_router.post(
    "/match",
    response_model=MatchResponse,
    summary="Match resume to jobs",
    description=(
        "Submit resume text and receive ranked job matches with "
        "AI-generated fit analysis, matched skills, skill gaps, and recommendations."
    ),
)
async def match_resume(request: MatchRequest):
    _require_engine()

    if not matcher_engine.jobs_meta:
        raise HTTPException(
            status_code=400,
            detail="No jobs indexed yet. POST to /matcher/jobs first.",
        )

    try:
        start = time.time()
        matches = matcher_engine.match_resume(
            resume_text=request.resume_text,
            top_k=request.top_k,
            min_score=request.min_score,
        )
        elapsed_ms = round((time.time() - start) * 1000, 2)

        return MatchResponse(
            total_matches=len(matches),
            matches=matches,
            elapsed_ms=elapsed_ms,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Matching failed: {e}")


@matcher_router.delete(
    "/jobs",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Clear all indexed jobs",
)
async def clear_jobs():
    _require_engine()
    matcher_engine.clear()
