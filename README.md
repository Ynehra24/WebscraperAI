# WebscraperAI


Here’s a **project-submission quality README** — written the way evaluators expect (clear architecture, novelty, evaluation, and future scope). You can paste this directly into `README.md`.

---

# Pharma Intelligence Agent

**AI-Powered Monitoring and Analysis of Pharma & Life Sciences Intelligence**

---

## 1. Problem Statement

The pharmaceutical and life sciences industry produces a massive amount of information daily:

* Press releases
* Regulatory announcements
* Clinical trial updates
* Product launches
* Partnerships and acquisitions

Manually tracking and analyzing this information is:

* Time consuming
* Fragmented
* Prone to missing critical signals

This project builds an **AI-powered intelligence agent** that automatically:

* Scrapes websites
* Extracts meaningful content
* Generates structured insights
* Identifies business signals

---

## 2. Solution Overview

The system acts as an **Always-On Pharma Intelligence Agent** that converts raw web content into structured, actionable insights.

Pipeline:

```
Websites / News Sources
        ↓
Scraper & Cleaner
        ↓
Chunking Engine
        ↓
LLM Analysis
        ↓
Structured Intelligence
        ↓
Dashboard / Alerts / Reports
```

---

## 3. Features Implemented

### Always-On Monitoring

Continuously scrapes company websites and relevant pages.

### Agentic Intelligence

AI analyzes content and extracts:

* Market signals
* Entities
* Business implications

### Marketing Brief Generator

Automatically produces structured summaries that can be used for:

* GTM analysis
* Competitive intelligence
* Market research

### Entity Tracking

Identifies:

* Companies
* Drugs
* Technologies
* Partnerships

### Smart Alerts (Design-Level)

The system identifies signals that could trigger alerts:

* Acquisitions
* Product launches
* Clinical trial updates

### Full-Text Search (Design-Level)

Prepared for integration with vector databases.

---

## 4. System Architecture

### High-Level Architecture

```
                 ┌──────────────┐
                 │ Web Sources  │
                 └──────┬───────┘
                        │
                ┌───────▼────────┐
                │ Web Scraper     │
                │ (Requests + BS4)│
                └───────┬────────┘
                        │
                ┌───────▼────────┐
                │ Text Cleaner    │
                │ & Parser        │
                └───────┬────────┘
                        │
                ┌───────▼────────┐
                │ Chunking Engine │
                └───────┬────────┘
                        │
                ┌───────▼────────┐
                │ LLM Analyzer    │
                │ (OpenRouter)    │
                └───────┬────────┘
                        │
                ┌───────▼────────┐
                │ Structured JSON │
                └─────────────────┘
```

---

## 5. Tech Stack

| Component              | Technology     |
| ---------------------- | -------------- |
| Scraping               | Requests       |
| Parsing                | BeautifulSoup  |
| Text Processing        | Regex, Python  |
| AI Analysis            | OpenRouter API |
| Environment Management | python-dotenv  |
| Language               | Python         |

---

## 6. How It Works (Flow)

1. URL is provided
2. HTML is fetched with headers to bypass blocking
3. Scripts, forms, and noise are removed
4. Content is extracted and cleaned
5. Text is chunked
6. Each chunk is analyzed by an LLM
7. Structured insights are returned

Output format:

* Overview
* Key Insights
* Entities
* Metrics
* Business Impact
* Signals to Monitor

---

## 7. Installation

### Clone the repository

```
git clone <repo-url>
cd <repo>
```

### Install dependencies

```
pip install -r requirements.txt
```

Example requirements:

```
requests
beautifulsoup4
python-dotenv
```

---

## 8. Environment Setup

Create a `.env` file:

```
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct
```

Add to `.gitignore`:

```
.env
```

---

## 9. Usage

Run the scraper:

```
python webscraper.py https://www.indegene.com --openrouter
```

Optional parameters:

| Argument             | Purpose                    |
| -------------------- | -------------------------- |
| `--same-domain-only` | Restrict crawling          |
| `--chunk-size`       | Control chunk size         |
| `--overlap`          | Chunk overlap              |
| `--max-links`        | Number of pages to process |
| `--model`            | Override LLM model         |

---

## 10. Example Output

```
Overview:
Indegene provides commercialization solutions for life sciences companies.

Key Insights:
Expansion into digital commercialization and AI-driven marketing.

Entities:
Indegene, Biopharma companies

Metrics:
Revenue growth figures, acquisition announcements

Business Impact:
Faster go-to-market for pharma companies.

Signals to Monitor:
Partnerships, product launches, acquisitions
```

---

## 11. Novelty of the Project

Unlike traditional scrapers, this system:

* Converts unstructured web data into business intelligence
* Identifies strategic signals automatically
* Structures insights for decision-making
* Designed for pharma-specific intelligence workflows

Key innovation:
**Turning scraping + LLMs into a decision-support intelligence agent**

---

## 12. Evaluation Criteria

The system is evaluated on:

### Accuracy

* Relevant insights extracted
* Noise removed effectively

### Coverage

* Number of meaningful signals detected

### Performance

* Time per page processed
* Chunk handling efficiency

### Usability

* Readable structured output
* Ease of running and extending

---

## 13. Limitations

* JavaScript-heavy websites may not load fully
* API rate limits may slow processing
* LLM output may vary slightly

---

## 14. Future Scope

Planned improvements:

* RSS feed ingestion
* Vector database search
* Real-time alerting
* Trend detection across weeks/months
* Dashboard visualization
* Competitive intelligence scoring
* GTM signal detection

---

## 15. Potential Real-World Applications

* Pharma market intelligence teams
* Competitive research
* Investment analysis
* Product strategy teams
* Healthcare consulting

---

## 16. Team Contribution (Edit This)

| Member | Contribution |
| ------ | ------------ |
| Name   | Scraper      |
| Name   | AI Pipeline  |
| Name   | Research     |

---

## 17. License

MIT License

---

If you want to **push this project from “good” to “top-tier project”**, I can show you **one small feature to add that will massively impress evaluators** (takes ~45 minutes to implement).
