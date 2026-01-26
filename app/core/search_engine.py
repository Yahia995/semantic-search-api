from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from datetime import datetime


class SemanticSearchEngine:
   
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_path: Optional[str] = None
    ):
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        self.index: Optional[faiss.Index] = None
        self.documents: List[Dict] = []
        self.index_path = index_path or "data/faiss_index"
        
        if os.path.exists(f"{self.index_path}.index"):
            self.load_index()
        else:
            self._create_new_index()
        
        print(f"Search engine ready (dimension: {self.dimension})")
    
    def _create_new_index(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
    
    def index_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None) -> Dict:
        if not documents:
            return {"status": "error", "message": "No documents provided"}
        
        start_time = datetime.now()
        
        print(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        self.index.add(embeddings.astype('float32'))
        
        for i, doc in enumerate(documents):
            doc_data = {
                "id": len(self.documents) + i,
                "text": doc,
                "metadata": metadata[i] if metadata and i < len(metadata) else {}
            }
            self.documents.append(doc_data)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        return {
            "status": "success",
            "documents_indexed": len(documents),
            "total_documents": len(self.documents),
            "elapsed_seconds": round(elapsed, 2)
        }
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        if not self.documents:
            return []
        
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            min(top_k, len(self.documents))
        )
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                similarity = 1 / (1 + distance)
                
                if score_threshold is None or similarity >= score_threshold:
                    doc = self.documents[idx]
                    results.append({
                        "id": doc["id"],
                        "text": doc["text"],
                        "metadata": doc["metadata"],
                        "score": round(float(similarity), 4),
                        "distance": round(float(distance), 4)
                    })
        
        return results
    
    def save_index(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        faiss.write_index(self.index, f"{self.index_path}.index")
        
        with open(f"{self.index_path}.docs", "wb") as f:
            pickle.dump(self.documents, f)
        
        print(f"Index saved to {self.index_path}")
    
    def load_index(self):
        try:
            self.index = faiss.read_index(f"{self.index_path}.index")
            
            with open(f"{self.index_path}.docs", "rb") as f:
                self.documents = pickle.load(f)
            
            print(f"Loaded index with {len(self.documents)} documents")
        except Exception as e:
            print(f"Could not load index: {e}")
            self._create_new_index()
    
    def get_stats(self) -> Dict:
        return {
            "total_documents": len(self.documents),
            "index_dimension": self.dimension,
            "model_name": self.model_name,
            "index_type": type(self.index).__name__
        }
    
    def clear_index(self):
        self._create_new_index()
        print("Index cleared")
