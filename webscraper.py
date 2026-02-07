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


def print_section(title: str):
    bar = "=" * 100
    print(f"\n{bar}\n{title}\n{bar}")

def print_subsection(title: str):
    bar = "-" * 100
    print(f"\n{bar}\n{title}\n{bar}")

def safe_print_block(text: str, width: int = 120):
    for line in text.splitlines():
        if len(line) > width:
            for i in range(0, len(line), width):
                print(line[i:i+width])
        else:
            print(line)


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

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = (soup.title.string or "").strip() if soup.title else ""

    content_selectors = ["article", "main", "section", "div", "p", "li", "h1", "h2", "h3", "h4"]
    texts: List[str] = []
    for sel in content_selectors:
        for node in soup.select(sel):
            t = node.get_text(separator=" ", strip=True)
            if t:
                texts.append(t)
    raw_text = " ".join(texts) or soup.get_text(separator=" ", strip=True)

    text = re.sub(r"\s+", " ", raw_text).strip()

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
    ref = os.getenv("OPENROUTER_REFERRER")
    app = os.getenv("OPENROUTER_APP_TITLE")
    if ref:
        headers["HTTP-Referer"] = ref
    if app:
        headers["X-Title"] = app

    payload = {"model": use_model, "messages": messages, "temperature": temperature}
    resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def summarize_chunk_openrouter(url: str, title: str, chunk: str, idx: int, total: int, model: str = None) -> str:
    # Strong prompt for detailed but concise, structured bullets
    system_prompt = (
        "You are a precise analyst. Summarize the provided webpage chunk with accurate, detailed insights. "
        "Respond ONLY with plain text headings and bullet lists (no code blocks)."
    )
    user_prompt = (
        f"Current URL: {url}\n"
        f"Page Title: {title}\n"
        f"Chunk: {idx} of {total}\n\n"
        "Instructions:\n"
        "- Produce a concise, information-dense summary.\n"
        "- Use short headings and bullet points only.\n"
        "- Sections: Overview, Key Points, Entities, Metrics, Risks, Actions.\n"
        "- Cite concrete facts, numbers, names, and definitions when present.\n"
        "- Avoid repetition across bullets and avoid generic phrasing.\n\n"
        "Chunk Content:\n"
        f"{chunk}"
    )
    return call_openrouter(
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        model=model,
    )


def process_link_with_openrouter(url: str, same_domain_only: bool, model: str, print_chunks: bool, chunk_size: int, overlap: int) -> Dict:
    # Scrape and chunk for this URL
    res = scrape(url, same_domain_only=same_domain_only)
    res.chunks = chunk_text(res.text, max_tokens=chunk_size, overlap=overlap)

    per_chunk_summaries: List[str] = []
    per_chunk_durations: List[float] = []

    print_section(f"Processing URL: {url}")
    print_subsection(f"Title: {res.title or '(no title)'}")
    print_subsection(f"Total chunks: {len(res.chunks)}")

    for i, chunk in enumerate(res.chunks, 1):
        print_section(f"[{url}] Chunk {i}/{len(res.chunks)}")
        if print_chunks:
            safe_print_block(chunk)

        start = time.perf_counter()
        try:
            summary = summarize_chunk_openrouter(res.url, res.title, chunk, i, len(res.chunks), model=model)
        except Exception as e:
            summary = f"Error summarizing chunk {i}: {e}"
        elapsed = time.perf_counter() - start
        per_chunk_summaries.append(summary)
        per_chunk_durations.append(elapsed)

        print_subsection(f"[{url}] Summary for Chunk {i} (elapsed: {elapsed:.2f}s)")
        safe_print_block(summary)

    return {
        "url": res.url,
        "title": res.title,
        "links": res.links,
        "chunks": res.chunks,
        "chunk_summaries": per_chunk_summaries,
        "chunk_durations_sec": per_chunk_durations,
    }


def main():
    parser = argparse.ArgumentParser(description="Scrape a webpage, visit all links, chunk text, and get per-chunk LLM insights.")
    parser.add_argument("url", help="URL to scrape")
    parser.add_argument("--same-domain-only", action="store_true", help="Visit only links from the same domain")
    parser.add_argument("--max-sentences", type=int, default=5, help="Max sentences in extractive summary (non-LLM)")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Approx words per chunk")
    parser.add_argument("--overlap", type=int, default=150, help="Approx words overlap between chunks")
    parser.add_argument("--output", "-o", type=str, default="", help="Output JSON file path")
    parser.add_argument("--openrouter", action="store_true", help="Send each chunk to OpenRouter and print summaries")
    parser.add_argument("--model", type=str, default=OPENROUTER_DEFAULT_MODEL, help="OpenRouter model id (overrides env)")
    parser.add_argument("--print-chunks", action="store_true", help="Print each chunk content to stdout")
    parser.add_argument("--max-links", type=int, default=0, help="Limit number of discovered links to visit (0 = no limit)")
    args = parser.parse_args()

    try:
        # Scrape root and compute its chunks
        root = scrape(args.url, same_domain_only=args.same_domain_only)
        root.chunks = chunk_text(root.text, max_tokens=args.chunk_size, overlap=args.overlap)
        root.summary = summarize_text(root.text, max_sentences=args.max_sentences)

        results: Dict[str, Dict] = {}

        # Process root page
        if args.openrouter:
            results[root.url] = process_link_with_openrouter(
                root.url, args.same_domain_only, args.model, args.print_chunks, args.chunk_size, args.overlap
            )
        else:
            results[root.url] = {
                "url": root.url,
                "title": root.title,
                "links": root.links,
                "chunks": root.chunks,
                "extractive_summary": root.summary,
            }
            print_section(f"Chunks for URL: {root.url}")
            for i, c in enumerate(root.chunks, 1):
                print_section(f"[{root.url}] Chunk {i}/{len(root.chunks)}")
                safe_print_block(c)

        # Prepare discovered links list with optional limit
        discovered_links = root.links
        if args.max_links and args.max_links > 0:
            discovered_links = discovered_links[: args.max_links]

        # Visit and process every discovered link
        for idx, link in enumerate(discovered_links, 1):
            print_section(f"Visiting link {idx}/{len(discovered_links)}: {link}")
            try:
                if args.openrouter:
                    results[link] = process_link_with_openrouter(
                        link, args.same_domain_only, args.model, args.print_chunks, args.chunk_size, args.overlap
                    )
                else:
                    lr = scrape(link, same_domain_only=args.same_domain_only)
                    lr.chunks = chunk_text(lr.text, max_tokens=args.chunk_size, overlap=args.overlap)
                    lr.summary = summarize_text(lr.text, max_sentences=args.max_sentences)
                    results[link] = {
                        "url": lr.url,
                        "title": lr.title,
                        "links": lr.links,
                        "chunks": lr.chunks,
                        "extractive_summary": lr.summary,
                    }
                    print_section(f"Chunks for URL: {lr.url}")
                    for i, c in enumerate(lr.chunks, 1):
                        print_section(f"[{lr.url}] Chunk {i}/{len(lr.chunks)}")
                        safe_print_block(c)
            except requests.HTTPError as e:
                print_subsection(f"HTTP error on {link}: {e}")
            except Exception as e:
                print_subsection(f"Error processing {link}: {e}")

        payload = {"root": root.url, "results": results}
        out = json.dumps(payload, ensure_ascii=False, indent=2)

        print_section("JSON Output")
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(out)
            print_subsection(f"Wrote JSON to {args.output}")
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