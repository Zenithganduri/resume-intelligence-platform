# Resume Intelligence Platform

An AI-powered resume-to-job-description matching system built entirely on AWS. Combines a custom NLP scoring pipeline with AWS Bedrock LLM integration and a two-stage FAISS retrieval architecture to score, store, and rank resumes against job descriptions.

---

## What it does

The platform supports three modes of operation:

| Mode | Input | Output |
|------|-------|--------|
| **Mode 1** | Resume + Job Description | Explainable 0-100 role-fit score, five-dimension breakdown, AI-generated improvement suggestions |
| **Mode 2** | Resume only | Parses, embeds, and stores the resume for future matching |
| **Mode 3** | Job Description only | Searches all stored resumes, returns top 3 candidates with scores and an AI-generated ranking explanation |

Each mode is accessible two ways: directly via REST API, or by simply uploading a file to the corresponding S3 folder. The system detects the mode automatically and processes it end to end with no manual API call required.

---

## Architecture
Postman
                         |
                   direct API calls
                         v
**Two-stage retrieval (Mode 3):** FAISS performs fast kNN vector search over stored resume embeddings to narrow the candidate pool from hundreds down to the top 10 in milliseconds. The full NLP scoring pipeline then runs detailed analysis on those 10, and Bedrock reranks the final top 3 with a plain-language explanation for the recruiter.

**Event-driven pipeline:** Uploading a file to S3 triggers a Lambda function that reads the folder prefix to determine the mode, coordinates Mode 1's resume/JD pairing using SQS, and calls the appropriate Flask endpoint on EC2. Fully automated, no manual trigger needed.

---

## Scoring methodology

Each resume is scored against a job description using a weighted five-dimension rubric:
| Dimension | Weight | What it measures |
|-----------|--------|-------------------|
| S - Skills Coverage | 35% | Semantic similarity between resume content and JD required/preferred lines |
| E - Evidence Strength | 25% | Presence of quantified outcomes (numbers, money, outcome verbs) in experience bullets |
| X - Experience Alignment | 20% | Matched years vs. required years, with a squared penalty for shortfalls |
| K - Keyword/ATS Alignment | 10% | Percentage of significant JD keywords present in the resume |
| L - Logistical Fit | 10% | Location, remote, and timezone signal matching |

Experience matching uses a three-gate detection system to correctly attribute years only to genuine work-experience subsections, distinguishing real roles from education, projects, and certifications even when section headings are inconsistent.

---

## Tech stack

**Document parsing & NLP**
- PyMuPDF, pymupdf4llm - column-aware PDF text extraction
- docx2python - Word document parsing
- SpaCy (en_core_web_md) - named entity recognition, POS tagging, noun chunk extraction
- dateparser - natural language date range parsing for experience duration

**Machine learning**
- Sentence-Transformers (all-MiniLM-L6-v2) - 384-dimensional semantic embeddings
- FAISS - vector similarity search (two-stage retrieval)
- AWS Bedrock (Amazon Nova Lite) - LLM-generated suggestions and ranking explanations

**Backend & infrastructure**
- Flask - REST API
- Docker - containerization
- Amazon ECR - container registry
- Amazon EC2 - compute
- Amazon S3 - document storage
- Amazon DynamoDB - metadata and vector storage
- AWS Lambda - event-driven routing
- Amazon SQS - Mode 1 resume/JD pairing coordination
- AWS IAM - least-privilege role-based access

**CI/CD**
- GitHub to AWS CodePipeline to AWS CodeBuild to Amazon ECR
- Automated build and push on every push to main

---

## API Reference

### GET /health
Health check.
```json
{ "service": "resume-intelligence", "status": "ok" }
```

### POST /upload-resume
Mode 2 - ingest a resume.
### POST /upload-both
Mode 1 - score a resume against a job description.
Returns overall score, five-dimension breakdown, and AI-generated improvement suggestions.

### POST /match-jd
Mode 3 - find top matching resumes for a job description.
Returns top 3 candidates with scores and an AI-generated ranking explanation.

---

## Sample output

```json
{
  "candidate_name": "Jesse Jayant",
  "overall_role_fit_score": 71.2,
  "matched_experience_years": 15.0,
  "required_experience_years": 7.0,
  "score_breakdown": {
    "skills_score": 55.0,
    "evidence_score": 85.0,
    "experience_score": 100.0,
    "keyword_score": 37.3,
    "logistical_score": 70.0
  },
  "llm_suggestions": {
    "summary": "Strong experience alignment; needs keyword optimization.",
    "top_strength": "Experience alignment",
    "ats_pass_probability": "medium"
  }
}
```

---

## Engineering decisions

**FAISS over OpenSearch.** At this scale (under 500 resumes), FAISS running in-process delivers millisecond search with zero infrastructure cost. OpenSearch would add managed-cluster overhead that isn't justified until concurrent-write volume or candidate-pool size grows significantly.

**Two-stage retrieval over single-pass LLM scoring.** Sending every resume to an LLM for every job description doesn't scale in cost or latency. FAISS narrows the field cheaply; the LLM is reserved for the step where holistic reasoning adds real value: generating suggestions and explaining the final ranking.

**SQS for Mode 1 pairing over re-listing S3.** Two Lambda invocations (one per uploaded file) can run concurrently. Re-checking S3's object listing on each invocation is vulnerable to consistency lag and race conditions. SQS provides a durable, atomic record of pairing state that's reliable regardless of upload timing.

**EC2 over Lambda for the Flask API.** The processing pipeline depends on SpaCy and Sentence-Transformers, which are too large and slow-starting for Lambda's container size and cold-start expectations. EC2 running a long-lived Docker container is the simpler, more reliable fit for this workload.

---

## Setup

```bash
git clone https://github.com/Zenithganduri/resume-intelligence-platform.git
cd resume-intelligence-platform

pip install -r requirements.txt

aws configure

python app.py
```

Or run via Docker:

```bash
docker build -t resume-intelligence .
docker run -p 8080:8080 -e AWS_REGION=us-east-1 resume-intelligence
```

---

## Author

Built by Zenith Ganduri - AWS AI Practitioner (December 2025), AWS ML Engineer Associate (May 2026)
