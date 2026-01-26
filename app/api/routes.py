from fastapi import APIRouter, HTTPException, status
from typing import List
import time

from app.models.schemas import (
    IndexRequest,
    SearchRequest,
    SearchResponse,
    IndexResponse,
    StatsResponse,
    HealthResponse
)
from app.core.search_engine import SemanticSearchEngine

router = APIRouter()

# Global search engine instance (initialized in main.py)
search_engine: SemanticSearchEngine = None


def set_search_engine(engine: SemanticSearchEngine):
    global search_engine
    search_engine = engine


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=search_engine is not None,
        documents_indexed=len(search_engine.documents) if search_engine else 0
    )


@router.post("/index", response_model=IndexResponse, status_code=status.HTTP_201_CREATED)
async def index_documents(request: IndexRequest):
    if not search_engine:
        raise HTTPException(status_code=500, detail="Search engine not initialized")
    
    try:
        texts = [doc.text for doc in request.documents]
        metadata = [doc.metadata for doc in request.documents]
        
        result = search_engine.index_documents(texts, metadata)
        
        search_engine.save_index()
        
        return IndexResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    if not search_engine:
        raise HTTPException(status_code=500, detail="Search engine not initialized")
    
    if not search_engine.documents:
        raise HTTPException(status_code=400, detail="No documents indexed yet")
    
    try:
        start_time = time.time()
        
        results = search_engine.search(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            elapsed_ms=round(elapsed_ms, 2)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    if not search_engine:
        raise HTTPException(status_code=500, detail="Search engine not initialized")
    
    stats = search_engine.get_stats()
    return StatsResponse(**stats)


@router.delete("/index", status_code=status.HTTP_204_NO_CONTENT)
async def clear_index():
    if not search_engine:
        raise HTTPException(status_code=500, detail="Search engine not initialized")
    
    search_engine.clear_index()
    search_engine.save_index()
