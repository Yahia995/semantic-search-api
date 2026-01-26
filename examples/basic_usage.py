import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

def main():
    # Sample documents
    documents = [
        {"text": "Python is excellent for data science and machine learning applications."},
        {"text": "FastAPI provides high-performance web APIs with automatic documentation."},
        {"text": "Docker containers ensure consistent application deployment across environments."},
        {"text": "Kubernetes manages containerized workloads and services at scale."},
        {"text": "Natural language processing enables machines to understand human language."},
    ]
    
    print("=" * 60)
    print("Semantic Search API - Basic Usage Example")
    print("=" * 60)
    
    # 1. Index documents
    print("\nIndexing documents...")
    response = requests.post(
        f"{BASE_URL}/index",
        json={"documents": documents}
    )
    result = response.json()
    print(f"Indexed {result['documents_indexed']} documents in {result['elapsed_seconds']}s")
    
    # 2. Semantic search
    queries = [
        "artificial intelligence and deep learning",
        "deploying applications in production",
        "building REST APIs"
    ]
    
    print("\n2️⃣  Performing semantic searches...")
    for query in queries:
        print(f"\nQuery: '{query}'")
        response = requests.post(
            f"{BASE_URL}/search",
            json={"query": query, "top_k": 2}
        )
        result = response.json()
        
        for i, doc in enumerate(result['results'], 1):
            print(f"   {i}. [{doc['score']:.3f}] {doc['text'][:60]}...")
    
    # 3. Get statistics
    print("\nIndex statistics...")
    response = requests.get(f"{BASE_URL}/stats")
    stats = response.json()
    print(f"Total documents: {stats['total_documents']}")
    print(f"Embedding dimension: {stats['index_dimension']}")
    print(f"Index type: {stats['index_type']}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
