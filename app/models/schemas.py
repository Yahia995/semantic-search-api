from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator

class DocumentInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Document text")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata")


class IndexRequest(BaseModel):
    documents: List[DocumentInput] = Field(..., min_items=1, description="List of documents to index")
    
    @validator('documents')
    def validate_documents(cls, v):
        if len(v) > 1000:
            raise ValueError("Maximum 1000 documents per request")
        return v


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results")
    score_threshold: Optional[float] = Field(None, ge=0, le=1, description="Minimum similarity score")


class SearchResult(BaseModel):
    id: int
    text: str
    metadata: Dict[str, Any]
    score: float
    distance: float


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    elapsed_ms: float


class IndexResponse(BaseModel):
    status: str
    documents_indexed: int
    total_documents: int
    elapsed_seconds: float


class StatsResponse(BaseModel):
    total_documents: int
    index_dimension: int
    model_name: str
    index_type: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    documents_indexed: int
