import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

USER_AGENT = "Mozilla/5.0 (Macintosh; Mac OS X) WebscraperAI/1.0"
DEFAULT_TIMEOUT = 20

# OpenRouter configuration
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL")  # must be set in .env or passed via CLI


@dataclass
class ScrapeResult:
    url: str
    title: str
    text: str
    links: List[str]
    summary: str
    chunks: List[str]


def fetch_html(url: str) -> str:
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    resp = requests.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    ctype = resp.headers.get("Content-Type", "")
    if "text/html" not in ctype:
        raise ValueError(f"Unsupported content type: {ctype}")
    return resp.text


def parse_links_and_text(base_url: str, html: str) -> Tuple[str, List[str], str]:
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = (soup.title.string or "").strip() if soup.title else ""

    # Extract text: prioritize main content tags
    content_selectors = ["article", "main", "section", "div", "p", "li", "h1", "h2", "h3", "h4"]
    texts: List[str] = []
    for sel in content_selectors:
        for node in soup.select(sel):
            t = node.get_text(separator=" ", strip=True)
            if t:
                texts.append(t)
    raw_text = " ".join(texts) or soup.get_text(separator=" ", strip=True)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", raw_text).strip()

    # Extract and normalize links
    seen: Set[str] = set()
    links: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a.get("href").strip()
        if not href or href.startswith("#"):
            continue
        abs_url = urljoin(base_url, href)
        parsed = urlparse(abs_url)._replace(fragment="")
        normalized = parsed.geturl()
        if normalized.startswith(("mailto:", "tel:")):
            continue
        if normalized not in seen:
            seen.add(normalized)
            links.append(normalized)

    return title, links, text


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

    ranked = sorted(((i, s, score_sentence(s)) for i, s in enumerate(sentences)), key=lambda x: (-x[2], x[0]))
    selected = sorted(ranked[:max_sentences], key=lambda x: x[0])
    return " ".join(s for _, s, _ in selected).strip()


def chunk_text(text: str, max_tokens: int = 1200, overlap: int = 150) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    i = 0
    while i < len(words):
        end = min(i + max_tokens, len(words))
        chunk = " ".join(words[i:end])
        chunks.append(chunk)
        if end == len(words):
            break
        i = max(0, end - overlap)
    return chunks


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


def to_json(result: ScrapeResult) -> Dict:
    return {
        "url": result.url,
        "title": result.title,
        "links": result.links,
        "summary": result.summary,
        "chunks": result.chunks,
    }


def call_openrouter(messages, model: str = None, temperature: float = 0.2) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY in environment.")
    use_model = model or OPENROUTER_DEFAULT_MODEL
    if not use_model:
        raise RuntimeError("Missing OPENROUTER_MODEL in environment or pass --model.")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # Optional identification headers
    ref = os.getenv("OPENROUTER_REFERRER")
    app = os.getenv("OPENROUTER_APP_TITLE")
    if ref:
        headers["HTTP-Referer"] = ref
    if app:
        headers["X-Title"] = app

    payload = {"model": use_model, "messages": messages, "temperature": temperature}
    resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def summarize_chunk_openrouter(url: str, title: str, chunk: str, idx: int, total: int, model: str = None) -> str:
    system_prompt = "You are a concise analyst. Summarize the chunk into 5-10 bullet points. Be specific and avoid repetition."
    user_prompt = (
        f"Source: {url}\nTitle: {title}\nChunk {idx}/{total}:\n{chunk}\n\n"
        "Return a concise bulleted summary of this chunk."
    )
    return call_openrouter(
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        model=model,
    )


def process_chunks_with_openrouter(url: str, title: str, chunks: List[str], model: str = None) -> Dict[str, List[str]]:
    per_chunk_summaries: List[str] = []
    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n=== Chunk {i}/{len(chunks)} ===")
        print(chunk)
        summary = summarize_chunk_openrouter(url, title, chunk, i, len(chunks), model=model)
        per_chunk_summaries.append(summary)
        print(f"\n--- Summary for Chunk {i} ---")
        print(summary)
    synthesis_prompt = (
        f"Source: {url}\nTitle: {title}\nCombine the following per-chunk summaries into a single concise overall summary "
        f"with sections: Overview, Key Points, Entities, Metrics, Risks, Actions.\n\n" + "\n\n".join(per_chunk_summaries)
    )
    final_summary = call_openrouter(
        messages=[{"role": "system", "content": "You are a concise analyst."}, {"role": "user", "content": synthesis_prompt}],
        model=model,
    )
    return {"per_chunk": per_chunk_summaries, "final": final_summary}


def main():
    parser = argparse.ArgumentParser(description="Scrape a webpage, extract links, and summarize content.")
    parser.add_argument("url", help="URL to scrape")
    parser.add_argument("--same-domain-only", action="store_true", help="Return only links from the same domain")
    parser.add_argument("--max-sentences", type=int, default=5, help="Max sentences in summary")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Approx words per chunk")
    parser.add_argument("--overlap", type=int, default=150, help="Approx words overlap between chunks")
    parser.add_argument("--output", "-o", type=str, default="", help="Output JSON file path")
    parser.add_argument("--openrouter", action="store_true", help="Send each chunk to OpenRouter and print summaries")
    parser.add_argument("--model", type=str, default=OPENROUTER_DEFAULT_MODEL, help="OpenRouter model id (overrides env)")
    parser.add_argument("--print-chunks", action="store_true", help="Print each chunk to stdout")
    args = parser.parse_args()

    try:
        res = scrape(args.url, same_domain_only=args.same_domain_only)
        res.summary = summarize_text(res.text, max_sentences=args.max_sentences)
        res.chunks = chunk_text(res.text, max_tokens=args.chunk_size, overlap=args.overlap)

        # Print raw chunks if requested
        if args.print_chunks:
            for i, c in enumerate(res.chunks, 1):
                print(f"\n=== Chunk {i}/{len(res.chunks)} ===")
                print(c)

        payload = to_json(res)

        # Process chunks with OpenRouter if requested
        if args.openrouter:
            results = process_chunks_with_openrouter(res.url, res.title, res.chunks, model=args.model)
            payload["openrouter_chunk_summaries"] = results["per_chunk"]
            payload["openrouter_final_summary"] = results["final"]

        out = json.dumps(payload, ensure_ascii=False, indent=2)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(out)
        else:
            print(out)
    except requests.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()