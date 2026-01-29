#!/usr/bin/env python3
"""
Research Tracker: Fetches new interpretability papers from arXiv and Semantic Scholar.
Scores them for relevance and stores in a local database.

Usage:
    python research_tracker/tracker.py [--days N] [--verbose]
"""

import argparse
import json
import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
import yaml
import feedparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"


# Reference abstracts representing core interp topics
REFERENCE_ABSTRACTS = [
    "We present a mathematical framework for reverse engineering transformer language models. We identify attention heads as the key unit of analysis and show that they can be understood as performing two operations: determining where to move information (QK circuit) and what information to move (OV circuit).",
    "We study the phenomenon of superposition in neural networks, where models represent more features than dimensions. Using toy models, we show that superposition occurs when features are sparse and demonstrate phase transitions in feature representation.",
    "We train sparse autoencoders on language model activations to extract interpretable features. The resulting features are monosemantic - each corresponding to a single interpretable concept. We find that approximately 70% of extracted features are clearly interpretable.",
    "We use activation patching to identify circuits in transformer language models. By systematically intervening on model components, we can determine which attention heads and MLP layers are causally responsible for specific behaviors.",
    "We introduce circuit tracing, a method for revealing the computational graphs within language models. By combining sparse autoencoders with attribution methods, we can trace how information flows through the model to produce specific outputs.",
    "We present representation engineering, a method for controlling language model behavior by adding steering vectors to internal activations. By computing the difference between positive and negative example activations, we obtain directions that can be used to steer generation.",
]


def compute_reference_similarity(title: str, abstract: str) -> float:
    """Score paper relevance using TF-IDF similarity to reference abstracts."""
    try:
        texts = REFERENCE_ABSTRACTS + [f"{title}. {abstract}"]
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(texts)

        # Similarity of the paper to each reference
        paper_vector = tfidf_matrix[-1]
        ref_vectors = tfidf_matrix[:-1]
        similarities = cosine_similarity(paper_vector, ref_vectors)[0]

        # Return max similarity (best match to any reference topic)
        return float(similarities.max())
    except Exception:
        return 0.0


def load_config() -> dict:
    """Load configuration from YAML file."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_papers_db(path: Path) -> dict:
    """Load existing papers database."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"papers": {}, "last_updated": None}


def save_papers_db(db: dict, path: Path):
    """Save papers database."""
    path.parent.mkdir(parents=True, exist_ok=True)
    db["last_updated"] = datetime.now().isoformat()
    with open(path, "w") as f:
        json.dump(db, f, indent=2)


def paper_id(title: str, authors: list[str]) -> str:
    """Generate a unique ID for a paper based on title and authors."""
    key = f"{title.lower().strip()}|{'|'.join(sorted(a.lower().strip() for a in authors[:3]))}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def compute_relevance_score(title: str, abstract: str, config: dict) -> tuple[float, list[str]]:
    """Compute relevance score based on keyword matching and TF-IDF similarity."""
    text = f"{title} {abstract}".lower()
    keyword_score = 0
    matched_keywords = []

    weights = {"high": 3, "medium": 2, "low": 1}
    for level, weight in weights.items():
        for keyword in config["relevance_keywords"].get(level, []):
            if keyword.lower() in text:
                keyword_score += weight
                matched_keywords.append(keyword)

    # TF-IDF similarity score (0-1, scaled to 0-10)
    tfidf_score = compute_reference_similarity(title, abstract) * 10

    # Combined score
    combined_score = keyword_score + tfidf_score

    return combined_score, matched_keywords


def categorize_paper(title: str, abstract: str, config: dict) -> list[str]:
    """Assign topic categories to a paper."""
    text = f"{title} {abstract}".lower()
    categories = []
    for category, keywords in config["topic_categories"].items():
        for keyword in keywords:
            if keyword.lower() in text:
                categories.append(category)
                break
    return categories if categories else ["General"]


def fetch_arxiv(config: dict, days_lookback: int, verbose: bool = False) -> list[dict]:
    """Fetch papers from arXiv API."""
    papers = []
    base_url = "http://export.arxiv.org/api/query"
    cutoff_date = datetime.now() - timedelta(days=days_lookback)

    categories = "+OR+".join(f"cat:{c}" for c in config["arxiv"]["categories"])

    for query in config["arxiv"]["queries"]:
        search_query = f'all:"{query}"+AND+({categories})'
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": config["arxiv"]["max_results_per_query"],
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        if verbose:
            print(f"  arXiv query: {query}")

        try:
            response = requests.get(base_url, params=params, timeout=30)
            feed = feedparser.parse(response.content)

            for entry in feed.entries:
                # Parse date
                published = datetime(*entry.published_parsed[:6])
                if published < cutoff_date:
                    continue

                authors = [a.get("name", "") for a in entry.get("authors", [])]
                arxiv_id = entry.id.split("/abs/")[-1].split("v")[0] if "/abs/" in entry.id else entry.id

                paper = {
                    "title": entry.title.replace("\n", " ").strip(),
                    "authors": authors,
                    "abstract": entry.summary.replace("\n", " ").strip(),
                    "date": published.strftime("%Y-%m-%d"),
                    "url": f"https://arxiv.org/abs/{arxiv_id}",
                    "source": "arxiv",
                    "arxiv_id": arxiv_id,
                }
                papers.append(paper)

            # Rate limiting
            time.sleep(1)

        except Exception as e:
            if verbose:
                print(f"    Error fetching arXiv for '{query}': {e}")

    return papers


def fetch_semantic_scholar(config: dict, days_lookback: int, verbose: bool = False) -> list[dict]:
    """Fetch papers from Semantic Scholar API."""
    papers = []
    base_url = "https://api.semanticscholar.org/graph/v1"
    cutoff_date = (datetime.now() - timedelta(days=days_lookback)).strftime("%Y-%m-%d")

    # Search by queries
    for query in config["semantic_scholar"].get("queries", []):
        if verbose:
            print(f"  S2 query: {query}")

        try:
            params = {
                "query": query,
                "limit": 20,
                "fields": "title,authors,abstract,url,publicationDate,externalIds",
                "publicationDateOrYear": f"{cutoff_date}:",
            }
            response = requests.get(f"{base_url}/paper/search", params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                for item in data.get("data", []):
                    if not item.get("title") or not item.get("abstract"):
                        continue

                    authors = [a.get("name", "") for a in item.get("authors", [])]
                    ext_ids = item.get("externalIds", {})
                    arxiv_id = ext_ids.get("ArXiv")
                    url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else (item.get("url") or "")

                    paper = {
                        "title": item["title"],
                        "authors": authors,
                        "abstract": item["abstract"],
                        "date": item.get("publicationDate", ""),
                        "url": url,
                        "source": "semantic_scholar",
                        "arxiv_id": arxiv_id,
                    }
                    papers.append(paper)

            time.sleep(1)  # Rate limiting

        except Exception as e:
            if verbose:
                print(f"    Error fetching S2 for '{query}': {e}")

    return papers


def deduplicate_papers(papers: list[dict]) -> list[dict]:
    """Remove duplicate papers based on title similarity."""
    seen = {}
    unique = []
    for paper in papers:
        pid = paper_id(paper["title"], paper["authors"])
        if pid not in seen:
            seen[pid] = True
            unique.append(paper)
    return unique


def run_tracker(days_lookback: int = 30, verbose: bool = False) -> dict:
    """Main tracker function. Fetches, scores, and stores papers."""
    config = load_config()
    db_path = PROJECT_DIR / config["output"]["papers_db"]
    db = load_papers_db(db_path)

    print(f"Fetching papers from the last {days_lookback} days...")

    # Fetch from sources
    if verbose:
        print("\nFetching from arXiv...")
    arxiv_papers = fetch_arxiv(config, days_lookback, verbose)
    if verbose:
        print(f"  Found {len(arxiv_papers)} papers from arXiv")

    if verbose:
        print("\nFetching from Semantic Scholar...")
    s2_papers = fetch_semantic_scholar(config, days_lookback, verbose)
    if verbose:
        print(f"  Found {len(s2_papers)} papers from Semantic Scholar")

    # Combine and deduplicate
    all_papers = deduplicate_papers(arxiv_papers + s2_papers)
    print(f"Total unique papers found: {len(all_papers)}")

    # Score and categorize
    new_count = 0
    for paper in all_papers:
        pid = paper_id(paper["title"], paper["authors"])

        if pid in db["papers"]:
            continue  # Already tracked

        score, keywords = compute_relevance_score(paper["title"], paper["abstract"], config)
        categories = categorize_paper(paper["title"], paper["abstract"], config)

        paper["relevance_score"] = score
        paper["matched_keywords"] = keywords
        paper["categories"] = categories
        paper["added_date"] = datetime.now().isoformat()

        db["papers"][pid] = paper
        new_count += 1

    # Save
    save_papers_db(db, db_path)
    print(f"New papers added: {new_count}")
    print(f"Total papers in database: {len(db['papers'])}")

    # Print top new papers
    if new_count > 0:
        new_papers = sorted(
            [p for p in all_papers if paper_id(p["title"], p["authors"]) in db["papers"]
             and db["papers"][paper_id(p["title"], p["authors"])].get("added_date", "")
             >= (datetime.now() - timedelta(seconds=60)).isoformat()],
            key=lambda p: db["papers"][paper_id(p["title"], p["authors"])].get("relevance_score", 0),
            reverse=True,
        )

        print(f"\nTop new papers by relevance:")
        for paper in new_papers[:10]:
            pid = paper_id(paper["title"], paper["authors"])
            entry = db["papers"][pid]
            cats = ", ".join(entry.get("categories", []))
            print(f"  [{entry['relevance_score']:5.1f}] [{cats}] {entry['title'][:80]}")
            print(f"       {entry['url']}")

    return {"new_count": new_count, "total": len(db["papers"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch new interpretability papers")
    parser.add_argument("--days", type=int, default=30, help="Days to look back")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    run_tracker(days_lookback=args.days, verbose=args.verbose)
