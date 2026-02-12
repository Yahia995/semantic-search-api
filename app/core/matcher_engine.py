import os
import json
import time
import pickle
import httpx
from typing import List, Dict, Optional, Any

from app.core.search_engine import SemanticSearchEngine
from app.models.matcher_schemas import FitLevel, JobMatch, SkillGap


ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

FIT_THRESHOLDS = [
    (0.55, FitLevel.STRONG),
    (0.40, FitLevel.GOOD),
    (0.28, FitLevel.PARTIAL),
    (0.0,  FitLevel.WEAK),
]


def _score_to_fit(score: float) -> FitLevel:
    for threshold, level in FIT_THRESHOLDS:
        if score >= threshold:
            return level
    return FitLevel.WEAK


class MatcherEngine:

    def __init__(self, index_path: str = "data/jobs_index"):
        self.engine = SemanticSearchEngine(
            model_name="all-MiniLM-L6-v2",
            index_path=index_path,
        )
        self.jobs_meta: List[Dict[str, Any]] = []
        self._meta_path = f"{index_path}.jobs_meta"
        self._load_meta()


    def index_jobs(self, jobs: List[Dict]) -> Dict:
        start = time.time()

        texts = [
            f"{j['title']} at {j['company']}. {j['description']}"
            for j in jobs
        ]
        meta_list = [
            {k: j.get(k) for k in ("title", "company", "location", "metadata")}
            for j in jobs
        ]

        result = self.engine.index_documents(texts, meta_list)
        self.engine.save_index()

        base_id = len(self.jobs_meta)
        for i, j in enumerate(jobs):
            self.jobs_meta.append({
                "id": base_id + i,
                "title": j["title"],
                "company": j["company"],
                "description": j["description"],
                "location": j.get("location"),
                "metadata": j.get("metadata", {}),
            })
        self._save_meta()

        elapsed = round(time.time() - start, 2)
        return {
            "status": "success",
            "jobs_indexed": len(jobs),
            "total_jobs": len(self.jobs_meta),
            "elapsed_seconds": elapsed,
        }

    def match_resume(
        self,
        resume_text: str,
        top_k: int = 5,
        min_score: float = 0.2,
    ) -> List[JobMatch]:
        raw_results = self.engine.search(
            query=resume_text,
            top_k=top_k,
            score_threshold=min_score,
        )

        matches: List[JobMatch] = []
        for r in raw_results:
            job = self.jobs_meta[r["id"]] if r["id"] < len(self.jobs_meta) else None
            if job is None:
                continue

            fit_level = _score_to_fit(r["score"])
            analysis = self._analyze_match(resume_text, job, r["score"])

            matches.append(
                JobMatch(
                    job_id=r["id"],
                    title=job["title"],
                    company=job["company"],
                    location=job.get("location"),
                    similarity_score=r["score"],
                    fit_level=fit_level,
                    match_summary=analysis["summary"],
                    skill_gap=SkillGap(
                        matched_skills=analysis["matched_skills"],
                        missing_skills=analysis["missing_skills"],
                    ),
                    recommendation=analysis["recommendation"],
                )
            )

        return matches

    def _analyze_match(
        self, resume_text: str, job: Dict, score: float
    ) -> Dict:
        prompt = f"""You are a technical recruiter analyzing a resume against a job posting.

RESUME:
{resume_text[:3000]}

JOB POSTING:
Title: {job['title']}
Company: {job['company']}
Description: {job['description'][:2000]}

Semantic similarity score: {score:.2f} (0=no match, 1=perfect match)

Respond ONLY with a JSON object — no preamble, no markdown fences — with exactly these keys:
{{
  "summary": "<1-2 sentences explaining why this candidate is or isn't a good fit>",
  "matched_skills": ["<skill1>", "<skill2>", ...],
  "missing_skills": ["<skill1>", "<skill2>", ...],
  "recommendation": "<one concrete action the candidate can take to improve this match>"
}}

Keep matched_skills and missing_skills to at most 5 items each. Be specific and honest."""

        try:
            response = httpx.post(
                ANTHROPIC_API_URL,
                headers={"Content-Type": "application/json"},
                json={
                    "model": ANTHROPIC_MODEL,
                    "max_tokens": 500,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=15.0,
            )
            response.raise_for_status()
            content = response.json()["content"][0]["text"].strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)

        except Exception as e:
            print(f"[MatcherEngine] AI analysis failed: {e}. Using fallback.")
            return self._fallback_analysis(score, job)

    @staticmethod
    def _fallback_analysis(score: float, job: Dict) -> Dict:
        fit = _score_to_fit(score)
        return {
            "summary": (
                f"Your profile shows a {fit.value.lower()} for the {job['title']} role "
                f"at {job['company']} based on semantic similarity."
            ),
            "matched_skills": [],
            "missing_skills": [],
            "recommendation": (
                "Add more specific skills and technologies from the job description to your resume."
            ),
        }

    def _save_meta(self):
        os.makedirs(os.path.dirname(self._meta_path) or ".", exist_ok=True)
        with open(self._meta_path, "wb") as f:
            pickle.dump(self.jobs_meta, f)

    def _load_meta(self):
        if os.path.exists(self._meta_path):
            try:
                with open(self._meta_path, "rb") as f:
                    self.jobs_meta = pickle.load(f)
                print(f"[MatcherEngine] Loaded {len(self.jobs_meta)} job postings.")
            except Exception as e:
                print(f"[MatcherEngine] Could not load job meta: {e}")
                self.jobs_meta = []

    def clear(self):
        self.engine.clear_index()
        self.engine.save_index()
        self.jobs_meta = []
        self._save_meta()

    def list_jobs(self) -> List[Dict]:
        return [
            {k: j[k] for k in ("id", "title", "company", "location", "metadata")}
            for j in self.jobs_meta
        ]
