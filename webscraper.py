import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

DEFAULT_TIMEOUT = 20

OPENROUTER_API_URL = os.getenv(
    "OPENROUTER_API_URL",
    "https://openrouter.ai/api/v1/chat/completions"
)
OPENROUTER_DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL")


@dataclass
class ScrapeResult:
    url: str
    title: str
    text: str
    links: List[str]
    summary: str
    chunks: List[str]


# ---------------- FETCH HTML (403 FIXED) ----------------

def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    session = requests.Session()
    session.headers.update(headers)

    resp = session.get(url, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()

    ctype = resp.headers.get("Content-Type", "")
    if "text/html" not in ctype:
        raise ValueError(f"Unsupported content type: {ctype}")

    return resp.text


# ---------------- CLEAN + PARSE ----------------

def parse_links_and_text(base_url: str, html: str) -> Tuple[str, List[str], str]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
        tag.decompose()

    title = (soup.title.string or "").strip() if soup.title else ""

    main_content = soup.find("main") or soup.find("article") or soup.body

    texts = []
    seen_blocks = set()

    for node in main_content.find_all(["p", "li", "h1", "h2", "h3", "h4"]):
        t = node.get_text(separator=" ", strip=True)

        if not t:
            continue
        if len(t) < 40:
            continue
        if t in seen_blocks:
            continue

        seen_blocks.add(t)
        texts.append(t)

    text = " ".join(texts)
    text = re.sub(r"\s+", " ", text).strip()

    seen: Set[str] = set()
    links: List[str] = []

    for a in soup.find_all("a", href=True):
        href = a.get("href").strip()
        if not href or href.startswith("#"):
            continue

        abs_url = urljoin(base_url, href)
        parsed = urlparse(abs_url)._replace(fragment="")
        normalized = parsed.geturl()

        if normalized.startswith(("mailto:", "tel:", "javascript:")):
            continue

        if normalized not in seen:
            seen.add(normalized)
            links.append(normalized)

    return title, links, text


# ---------------- TEXT UTILITIES ----------------

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[a-zA-Z0-9]+", text)]


def summarize_text(text: str, max_sentences: int = 5) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= max_sentences:
        return text

    tokens = tokenize(text)
    stopwords = {
        "the","a","an","and","or","but","if","then","else","for","to","of","in","on","at","by","with",
        "is","are","was","were","be","been","being","it","this","that","as","from","not","can","will",
        "we","you","they","he","she","them","his","her","their","our","your"
    }

    freq = Counter(t for t in tokens if t not in stopwords)

    def score_sentence(s: str) -> float:
        toks = tokenize(s)
        return sum(freq.get(tok, 0) for tok in toks) / (len(toks) + 1)

    ranked = sorted(
        ((i, s, score_sentence(s)) for i, s in enumerate(sentences)),
        key=lambda x: (-x[2], x[0])
    )

    selected = sorted(ranked[:max_sentences], key=lambda x: x[0])
    return " ".join(s for _, s, _ in selected).strip()


def chunk_text(text: str, max_tokens: int = 1200, overlap: int = 150) -> List[str]:
    words = text.split()
    step = max_tokens - overlap if max_tokens > overlap else max_tokens

    chunks = []
    seen_chunks = set()

    i = 0
    while i < len(words):
        end = min(i + max_tokens, len(words))
        chunk = " ".join(words[i:end]).strip()

        if chunk and chunk not in seen_chunks:
            chunks.append(chunk)
            seen_chunks.add(chunk)

        if end >= len(words):
            break
        i += step

    return chunks


# ---------------- SCRAPER ----------------

def scrape(url: str, same_domain_only: bool = False) -> ScrapeResult:
    html = fetch_html(url)
    title, links, text = parse_links_and_text(url, html)

    if same_domain_only:
        domain = urlparse(url).netloc
        links = [l for l in links if urlparse(l).netloc == domain]

    summary = summarize_text(text)
    chunks = chunk_text(text)

    return ScrapeResult(
        url=url,
        title=title,
        text=text,
        links=links,
        summary=summary,
        chunks=chunks,
    )


def is_valid_content(text: str) -> bool:
    if len(text) < 300:
        return False

    junk_patterns = [
        "privacy policy",
        "cookie",
        "full name",
        "submit",
        "email address"
    ]

    lower = text.lower()
    return not any(p in lower for p in junk_patterns)


# ---------------- OPENROUTER ----------------

def call_openrouter(messages, model: str = None, temperature: float = 0.2) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY")

    use_model = model or OPENROUTER_DEFAULT_MODEL
    if not use_model:
        raise RuntimeError("Missing OPENROUTER_MODEL")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": use_model,
        "messages": messages,
        "temperature": temperature
    }

    resp = requests.post(
        OPENROUTER_API_URL,
        headers=headers,
        json=payload,
        timeout=120
    )

    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def summarize_chunk_openrouter(url, title, chunk, idx, total, model=None):
    system_prompt = (
        "You are a pharma industry analyst summarizing webpages into structured intelligence."
    )

    user_prompt = f"""
URL: {url}
Title: {title}
Chunk {idx} of {total}

Summarize in this format:

Overview:
Key Insights:
Important Entities:
Numbers & Metrics:
Business Impact:
Signals to Monitor:

Text:
{chunk}
"""

    return call_openrouter(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
    )


def process_link_with_openrouter(url, same_domain_only, model, chunk_size, overlap):
    res = scrape(url, same_domain_only=same_domain_only)

    if not is_valid_content(res.text):
        print(f"Skipping low-value page: {url}")
        return None

    res.chunks = chunk_text(res.text, max_tokens=chunk_size, overlap=overlap)

    summaries = []

    for i, chunk in enumerate(res.chunks, 1):
        print(f"[{i}/{len(res.chunks)}] Processing chunk from {url}")

        try:
            summary = summarize_chunk_openrouter(
                res.url, res.title, chunk, i, len(res.chunks), model
            )
        except Exception as e:
            summary = f"Error: {e}"

        summaries.append(summary)
        print(summary)
        print()

        time.sleep(1.5)  # rate limiting

    return {
        "url": res.url,
        "title": res.title,
        "links": res.links,
        "chunk_summaries": summaries,
    }


# ---------------- MAIN ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--same-domain-only", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--openrouter", action="store_true")
    parser.add_argument("--model", type=str, default=OPENROUTER_DEFAULT_MODEL)
    parser.add_argument("--max-links", type=int, default=5)

    args = parser.parse_args()

    results = {}

    root = scrape(args.url, same_domain_only=args.same_domain_only)

    if args.openrouter:
        r = process_link_with_openrouter(
            root.url,
            args.same_domain_only,
            args.model,
            args.chunk_size,
            args.overlap
        )
        if r:
            results[root.url] = r

    links = root.links[: args.max_links]

    for link in links:
        try:
            time.sleep(2)
            r = process_link_with_openrouter(
                link,
                args.same_domain_only,
                args.model,
                args.chunk_size,
                args.overlap
            )
            if r:
                results[link] = r
        except Exception as e:
            print(f"Error processing {link}: {e}")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
