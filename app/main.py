from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes import router, set_search_engine
from app.core.search_engine import SemanticSearchEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize search engine
    print("=" * 50)
    print("Starting Semantic Search API")
    print("=" * 50)
    
    engine = SemanticSearchEngine(
        model_name="all-MiniLM-L6-v2",
        index_path="data/faiss_index"
    )
    set_search_engine(engine)
    
    print("=" * 50)
    print("API Ready")
    print("=" * 50)
    
    yield
    
    # Shutdown: Save index
    print("\n" + "=" * 50)
    print("Saving index before shutdown...")
    engine.save_index()
    print("Shutdown complete")
    print("=" * 50)


app = FastAPI(
    title="Semantic Search API",
    description="NLP-powered semantic document search using HuggingFace transformers and FAISS",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["Semantic Search"])


@app.get("/")
async def root():
    return {
        "name": "Semantic Search API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/api/v1/health",
            "index": "/api/v1/index",
            "search": "/api/v1/search",
            "stats": "/api/v1/stats"
        }
    }
