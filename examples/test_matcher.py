import json
import requests

BASE = "http://localhost:8000/api/v1/matcher"

SAMPLE_RESUME = """
Mohamed Ben Ali
Software Engineer

SKILLS
Python, FastAPI, REST APIs, Docker, PostgreSQL, Redis, Git, Linux
Machine Learning basics, NumPy, scikit-learn
React (beginner), TypeScript (beginner)

EXPERIENCE
Backend Developer Intern — TechCorp (2024)
- Built REST APIs with FastAPI and PostgreSQL
- Containerized services with Docker and docker-compose
- Wrote unit tests with pytest, achieving 85% coverage
- Optimized SQL queries reducing response time by 40%

PROJECTS
Semantic Search API (github.com/Yahia995/semantic-search-api)
- NLP-powered search using sentence-transformers and FAISS
- Deployed on Render with Docker, serving sub-20ms queries

EDUCATION
B.Sc. Computer Science — University of Sfax (2022–2025)
Relevant coursework: Algorithms, Databases, Networks, ML Fundamentals
"""


def run():
    print("=" * 60)
    print("Job Matcher — Integration Test")
    print("=" * 60)

    print("\n[1] Indexing sample jobs...")
    with open("tests/sample_jobs.json") as f:
        payload = json.load(f)

    r = requests.post(f"{BASE}/jobs", json=payload)
    r.raise_for_status()
    data = r.json()
    print(f"    ✓ Indexed {data['jobs_indexed']} jobs ({data['elapsed_seconds']}s)")

    print("\n[2] Listing indexed jobs...")
    r = requests.get(f"{BASE}/jobs")
    jobs = r.json()["jobs"]
    for j in jobs:
        print(f"    • [{j['id']}] {j['title']} @ {j['company']}")

    print("\n[3] Matching resume...")
    r = requests.post(
        f"{BASE}/match",
        json={"resume_text": SAMPLE_RESUME, "top_k": 3, "min_score": 0.15},
    )
    r.raise_for_status()
    result = r.json()

    print(f"\n    Found {result['total_matches']} matches in {result['elapsed_ms']}ms\n")
    print("=" * 60)

    for i, m in enumerate(result["matches"], 1):
        print(f"\n  #{i}  {m['title']} @ {m['company']}")
        print(f"       Fit: {m['fit_level']}  |  Score: {m['similarity_score']:.3f}")
        print(f"       {m['match_summary']}")
        if m["skill_gap"]["matched_skills"]:
            print(f"       ✅ Matched: {', '.join(m['skill_gap']['matched_skills'])}")
        if m["skill_gap"]["missing_skills"]:
            print(f"       ❌ Missing: {', '.join(m['skill_gap']['missing_skills'])}")
        print(f"       💡 {m['recommendation']}")

    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    run()
