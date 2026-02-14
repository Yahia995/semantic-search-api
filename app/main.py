import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes import router, set_search_engine
from app.api.matcher_routes import matcher_router, set_matcher_engine
from app.core.search_engine import SemanticSearchEngine
from app.core.matcher_engine import MatcherEngine

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 50)
    print("Starting Semantic Search API")
    print("=" * 50)

    engine = SemanticSearchEngine(
        model_name="all-MiniLM-L6-v2",
        index_path="data/faiss_index",
    )
    set_search_engine(engine)

    matcher = MatcherEngine(index_path="data/jobs_index")
    set_matcher_engine(matcher)

    print("=" * 50)
    print("API Ready")
    print("=" * 50)

    yield

    print("\n" + "=" * 50)
    print("Saving indices before shutdown...")
    engine.save_index()
    matcher.engine.save_index()
    print("Shutdown complete")
    print("=" * 50)


app = FastAPI(
    title="Semantic Search & Job Matcher API",
    description=(
        "NLP-powered semantic document search and resume–job matching "
        "using HuggingFace transformers, FAISS, and Claude AI."
    ),
    version="1.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["Semantic Search"])

app.include_router(matcher_router, prefix="/api/v1/matcher", tags=["Job Matcher"])


@app.get("/")
async def root():
    return {
        "name": "Semantic Search & Job Matcher API",
        "version": "1.2.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/api/v1/health",
            "search": "/api/v1/search",
            "index": "/api/v1/index",
            "stats": "/api/v1/stats",
            "matcher_index_jobs": "/api/v1/matcher/jobs",
            "matcher_match": "/api/v1/matcher/match",
        },
    }

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

@app.get("/demo", include_in_schema=False)
async def demo():
    absolute_html_path = os.path.join(FRONTEND_DIR, "index.html")
    return FileResponse(absolute_html_path)
