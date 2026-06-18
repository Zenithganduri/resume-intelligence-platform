
# In[9]:


import os
import fitz  # PyMuPDF
import pymupdf4llm
from docx2python import docx2python
from sentence_transformers import SentenceTransformer, util
import spacy
import re
import datetime
from dateparser.search import search_dates

# ================================
# UNIVERSAL TEXT EXTRACTION
# ================================
def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_pdf_blocks(file_path)
    elif ext in [".docx", ".doc"]:
        return extract_docx_exact_layout(file_path)
    else:
        raise ValueError("Unsupported file type. Use PDF or DOCX.")

# ================================
# DOCX Exact Layout Extraction
# ================================
def extract_docx_exact_layout(filepath):
    doc_result = docx2python(filepath)
    main_content = doc_result.body
    all_text = []
    for section in main_content:
        for row in section:
            for cell in row:
                for para in cell:
                    para_str = para.strip()
                    if para_str:
                        all_text.append(para_str)
    return all_text

# ================================
# PDF Column-Aware Block Extraction
# ================================
def extract_column_aware_blocks(pdf_path, column_gap=50):
    doc = fitz.open(pdf_path)
    all_blocks = []
    for page in doc:
        blocks = page.get_text("blocks", sort=True)
        blocks = sorted(blocks, key=lambda b: (b[0], b[1]))
        left_col, right_col = [], []
        if blocks:
            page_width = page.rect.width
            center_line = page_width / 2
            for b in blocks:
                x0, y0, x1, y1, text, *_ = b
                t = text.strip()
                if not t:
                    continue
                if x1 < center_line - column_gap:
                    left_col.append((y0, t))
                else:
                    right_col.append((y0, t))
            left_col_sorted = [t for _, t in sorted(left_col)]
            right_col_sorted = [t for _, t in sorted(right_col)]
            combined = left_col_sorted + right_col_sorted
            all_blocks.extend(combined)
    doc.close()
    return all_blocks

# ================================
# PDF Block Extraction and Merging
# ================================
def extract_pdf_blocks(pdf_path):
    blocks_code1 = extract_column_aware_blocks(pdf_path)

    md_text = pymupdf4llm.to_markdown(pdf_path)
    blocks_code2 = [
        block.strip()
        for block in md_text.split("\n\n")
        if block.strip() and len(block.strip()) > 5
    ]

    md_chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    page_chunk_blocks = [
        chunk["text"].strip()
        for chunk in md_chunks
        if isinstance(chunk, dict) and "text" in chunk and chunk["text"].strip()
    ]

    def jaccard_similarity(s1, s2, threshold=0.7):
        set1, set2 = set(s1.lower().strip()), set(s2.lower().strip())
        intersection = set1 & set2
        union = set1 | set2
        if not union:
            return False
        return len(intersection) / len(union) > threshold

    missing_from_code2 = []
    for block1 in blocks_code1:
        if not any(jaccard_similarity(block1, block2) for block2 in blocks_code2):
            missing_from_code2.append(block1)

    final_blocks = blocks_code2 + missing_from_code2
    return final_blocks, page_chunk_blocks

# ================================
# SEMANTIC HEADING DETECTION + SECTION GROUPING
# ================================
resume_headings = [
    "experience", "work experience", "education", "academic history",
    "projects", "technical projects", "skills", "key skills",
    "certifications", "achievements", "publications", "personal details", "contact",
    "technical skills", "professional experience", "core competencies", "community service",
    "portfolio management project", "project director"
]

jd_headings = [
    "job position", "role title", "job responsibilities", "responsibilities",
    "duties", "minimum qualifications", "requirements", "qualifications",
    "preferred qualifications", "desired skills", "job role", "key responsibilities"
]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
resume_embeddings = model.encode(resume_headings, convert_to_tensor=True)
jd_embeddings = model.encode(jd_headings, convert_to_tensor=True)

def semantic_heading_detection(lines, heading_list, embeddings, threshold=0.6):
    detected_headings = []
    for line in lines:
        line_clean = line.strip().lower()
        if not line_clean or len(line_clean) < 3:
            continue
        emb = model.encode(line_clean, convert_to_tensor=True)
        cosine_scores = util.cos_sim(emb, embeddings)[0]
        max_score = float(cosine_scores.max())
        if max_score >= threshold:
            detected_headings.append(line.strip())
    return detected_headings

def semantic_sectioning(lines, for_jd=False, threshold=0.55):
    headings = jd_headings if for_jd else resume_headings
    embeddings = jd_embeddings if for_jd else resume_embeddings
    detected_headings = semantic_heading_detection(lines, headings, embeddings, threshold)

    sections = {}
    current_section = None
    buffer = []
    section_counter = {}

    for line in lines:
        if line.strip() in detected_headings:
            if current_section and buffer:
                key = current_section
                count = section_counter.get(key, 0) + 1
                section_counter[key] = count
                section_key = f"{key} ({count})"
                sections[section_key] = "\n".join(buffer).strip()
            current_section = line.strip()
            buffer = []
        else:
            buffer.append(line.strip())

    if current_section and buffer:
        key = current_section
        count = section_counter.get(key, 0) + 1
        section_counter[key] = count
        section_key = f"{key} ({count})"
        sections[section_key] = "\n".join(buffer).strip()

    return sections

# ================================
# SPA-CY FEATURE EXTRACTION (with numbers)
# ================================
nlp = spacy.load("en_core_web_md")

def filter_and_clean_noun_chunks(doc):
    seen = set()
    clean_chunks = []
    for chunk in doc.noun_chunks:
        text = chunk.text.strip().lower()
        if not text or len(text) < 2:
            continue
        if all(token.is_stop for token in chunk):
            continue
        if text in seen:
            continue
        seen.add(text)
        clean_chunks.append(text)
    return clean_chunks

def extract_section_spacy_features(sections):
    section_features = {}
    for heading, text in sections.items():
        flat = " ".join(line.strip() for line in text.splitlines() if line.strip())
        doc = nlp(flat)

        noun_chunks = filter_and_clean_noun_chunks(doc)
        verbs = sorted(set(
            t.lemma_ for t in doc
            if t.pos_ == "VERB" and not t.is_stop and len(t.lemma_) > 1
        ))

        compounds = []
        for chunk in doc.noun_chunks:
            if any(t.dep_ == "compound" for t in chunk):
                compound_text = chunk.text.strip().lower()
                if compound_text not in compounds:
                    compounds.append(compound_text)

        verbal_nouns = sorted(set(
            t.text for t in doc
            if t.tag_ == "VBG" and t.pos_ == "NOUN"
        ))
        dates = sorted(set(ent.text for ent in doc.ents if ent.label_ == "DATE"))
        proper_nouns = sorted(set(
            t.text for t in doc
            if t.pos_ == "PROPN" and len(t.text.strip()) > 1
        ))

        numbers = set(t.text for t in doc if t.pos_ == "NUM")
        numbers = numbers.union({
            ent.text for ent in doc.ents
            if ent.label_ in ["CARDINAL", "QUANTITY", "ORDINAL", "MONEY"]
        })

        section_features[heading] = {
            "noun_chunks": noun_chunks,
            "compounds": compounds,
            "verbal_nouns": verbal_nouns,
            "verbs": verbs,
            "dates": dates,
            "proper_nouns": proper_nouns,
            "numbers": sorted(numbers)
        }
    return section_features

def print_section_parts_of_speech(sections):
    for heading, text in sections.items():
        print(f"\n### PARTS OF SPEECH IN SECTION: {heading.upper()} ###\n")
        doc = nlp(text)
        for token in doc:
            print(f"{token.text}\t{token.pos_}")
        print("=" * 60)

def tag_section_features_in_order(sections, section_features):
    for heading, text in sections.items():
        print(f"\n### {heading.upper()} ###\n")
        lines = text.splitlines()
        features = section_features[heading]

        feat_types = [
            'noun_chunks', 'compounds', 'verbal_nouns',
            'verbs', 'dates', 'proper_nouns', 'numbers'
        ]
        feat_map = {}
        for ft in feat_types:
            for val in features.get(ft, []):
                val_low = val.lower().strip()
                if val_low not in feat_map:
                    feat_map[val_low] = []
                feat_map[val_low].append(ft)

        phrases_sorted = sorted(feat_map.keys(), key=lambda x: -len(x))
        for line in lines:
            line_str = line.strip()
            output = line_str
            for phrase in phrases_sorted:
                if phrase and phrase in line_str.lower():
                    tag = ','.join(feat_map[phrase])
                    pat = r'(?i)\b({})\b'.format(re.escape(phrase))
                    output = re.sub(pat, r'[\1|{}]'.format(tag), output)
            print(output)
        print("=" * 60)

# ================================
# SUBSECTION SPLITTING
# ================================
BOLD_LINE_PATTERN = re.compile(r"\*\*(.+?)\*\*")
BULLET_CHARS = "•\u2022\u2023\u25E6*+-"

def is_all_caps_heading(line: str) -> bool:
    text = line.strip()
    if not text:
        return False
    alpha = "".join(ch for ch in text if ch.isalpha())
    return bool(alpha) and alpha.isupper()

def is_bold_line(line: str) -> bool:
    return bool(BOLD_LINE_PATTERN.search(line))

def is_sentence_start(line: str) -> bool:
    stripped = line.lstrip()
    return bool(stripped) and stripped[0].isupper()

def line_has_bullet(line: str) -> bool:
    return any(ch in line for ch in BULLET_CHARS)

def split_section_by_start_and_last_bullet(section_text, seen_lines):
    lines = section_text.splitlines()
    subsections = []
    current_start = None
    current_end = None

    def commit_block():
        nonlocal current_start, current_end
        if current_start is None or current_end is None:
            current_start = None
            current_end = None
            return
        block = lines[current_start:current_end + 1]
        if not block:
            current_start = None
            current_end = None
            return

        filtered = []
        for raw_line in block:
            line = raw_line.rstrip()
            if not line:
                filtered.append(raw_line)
                continue
            if line in seen_lines:
                continue
            seen_lines.add(line)
            filtered.append(raw_line)

        if filtered and any(l.strip() for l in filtered):
            subsections.append("\n".join(filtered).strip())

        current_start = None
        current_end = None

    for i, raw in enumerate(lines):
        line = raw.rstrip()
        if not line.strip():
            if current_start is not None:
                current_end = i
            continue

        start_like = is_bold_line(line) or is_all_caps_heading(line) or is_sentence_start(line)

        if start_like:
            if current_start is not None:
                commit_block()
            current_start = i
            current_end = i
        else:
            if current_start is not None:
                current_end = i

    if current_start is not None:
        commit_block()

    result = {}
    for idx, block_text in enumerate(subsections, start=1):
        key = f"Subsection {idx}"
        result[key] = block_text

    had_any_subsections = bool(result)
    return result, had_any_subsections

# ================================
# DATE & DURATION HELPERS
# ================================
def years_between_dates(start_dt, end_dt):
    months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
    return round(months / 12.0, 2)

MONTH_NAME = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
MONTH_YEAR = rf"{MONTH_NAME}\s+\d{{4}}"
YEAR_ONLY = r"\d{4}"
DATE_TOKEN = rf"(?:{MONTH_YEAR}|{YEAR_ONLY})"
PRESENT_WORD = r"(?:Present|Current|Ongoing|To\s+date)"
DATE_RANGE_PATTERN = re.compile(
    rf"({DATE_TOKEN})\s*[–\-]\s*({DATE_TOKEN}|{PRESENT_WORD})",
    re.IGNORECASE,
)

def find_date_ranges_in_text(text):
    matches = list(DATE_RANGE_PATTERN.finditer(text))
    if not matches:
        return []

    now = datetime.datetime.now()
    ranges = []

    for m in matches:
        start_str = m.group(1)
        end_str = m.group(2)

        start_parsed = search_dates(
            start_str,
            languages=['en'],
            settings={'PREFER_DATES_FROM': 'past'}
        )
        if not start_parsed:
            continue
        start_dt = start_parsed[0][1]

        if re.fullmatch(PRESENT_WORD, end_str, flags=re.IGNORECASE):
            end_dt = now
        else:
            end_parsed = search_dates(
                end_str,
                languages=['en'],
                settings={'PREFER_DATES_FROM': 'past'}
            )
            if not end_parsed:
                continue
            end_dt = end_parsed[0][1]

        if end_dt < start_dt:
            start_dt, end_dt = end_dt, start_dt
        elif end_dt == start_dt:
            continue

        ranges.append((start_dt, end_dt, m.start(), m.end()))

    return ranges

def append_multiple_durations_to_text(text):
    ranges = find_date_ranges_in_text(text)
    if not ranges:
        return text

    lines = [text]
    for idx, (start_dt, end_dt, _s, _e) in enumerate(ranges, start=1):
        years = years_between_dates(start_dt, end_dt)
        if years <= 0:
            continue
        start_str = start_dt.strftime("%b %Y")
        end_str = end_dt.strftime("%b %Y")
        lines.append(
            f"[Duration {idx}: {years} years | Range: {start_str} to {end_str}]"
        )

    return "\n".join(lines)

def section_has_date_range(text: str) -> bool:
    return bool(find_date_ranges_in_text(text))

def merge_subsections_attach_to_previous_with_dates(expanded_sections):
    by_parent = {}
    for full_key in expanded_sections.keys():
        parent = full_key.split(" – ")[0] if " – " in full_key else full_key
        by_parent.setdefault(parent, []).append(full_key)

    new_sections = dict(expanded_sections)

    for parent, keys in by_parent.items():
        keys_sorted = sorted(keys, key=lambda k: list(expanded_sections.keys()).index(k))

        last_with_dates_key = None
        for k in keys_sorted:
            text = new_sections.get(k, "")
            if section_has_date_range(text):
                last_with_dates_key = k
                continue

            if last_with_dates_key is not None and k != last_with_dates_key:
                prev_text = new_sections[last_with_dates_key]
                merged = prev_text.rstrip() + "\n" + text.lstrip()
                new_sections[last_with_dates_key] = merged
                del new_sections[k]
            else:
                continue

    return new_sections

def add_bold_subsections_to_all_sections(resume_sections):
    expanded = {}
    seen_lines = set()

    for heading, content in resume_sections.items():
        expanded[heading] = content
        subsections, had_any = split_section_by_start_and_last_bullet(content, seen_lines)
        if subsections:
            for sub_name, sub_text in subsections.items():
                if not sub_text:
                    continue
                new_key = f"{heading} – {sub_name}"
                expanded[new_key] = sub_text

    merged = merge_subsections_attach_to_previous_with_dates(expanded)

    final = {}
    for heading, text in merged.items():
        if section_has_date_range(text):
            final[heading] = append_multiple_durations_to_text(text)
        else:
            final[heading] = text

    return final

# ================================
# EXPERIENCE VS JD YEARS MATCHING
# ================================
EXPERIENCE_REQ_HEADING_KEYWORDS = [
    "requirements", "required", "qualifications", "skills", "what we are looking for",
]

def heading_looks_like_requirements(heading: str) -> bool:
    h = heading.lower()
    return any(k in h for k in EXPERIENCE_REQ_HEADING_KEYWORDS)

YEARS_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)", re.IGNORECASE)

def extract_years_number_from_text(text: str):
    m = YEARS_PATTERN.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None

def split_into_bullets(text: str):
    bullets = []
    current = []
    for line in text.splitlines():
        if line_has_bullet(line):
            if current:
                bullets.append(" ".join(current).strip())
                current = []
            current.append(re.sub(r"^\s*[" + re.escape(BULLET_CHARS) + r"]\s*", "", line).strip())
        else:
            if current:
                current.append(line.strip())
    if current:
        bullets.append(" ".join(current).strip())
    return [b for b in bullets if b]

def find_jd_experience_requirement_line(jd_sections):
    """
    Find the JD years-of-experience line and concatenate with the next line
    if it appears truncated (ends with a conjunction/preposition).
    """
    TRUNCATION_WORDS = {"including", "and", "or", "with", "such", "as", "for", "of"}

    def is_truncated(line: str) -> bool:
        last_word = line.strip().rstrip(".").split()[-1].lower() if line.strip() else ""
        return last_word in TRUNCATION_WORDS

    for heading, content in jd_sections.items():
        if not heading_looks_like_requirements(heading):
            continue
        bullets = split_into_bullets(content)
        for idx, b in enumerate(bullets):
            if YEARS_PATTERN.search(b):
                if is_truncated(b) and idx + 1 < len(bullets):
                    b = b + " " + bullets[idx + 1]
                return heading, b

    for heading, content in jd_sections.items():
        bullets = split_into_bullets(content)
        for idx, b in enumerate(bullets):
            if YEARS_PATTERN.search(b):
                if is_truncated(b) and idx + 1 < len(bullets):
                    b = b + " " + bullets[idx + 1]
                return heading, b

    return None, None

def extract_duration_years_from_augmented_text(text: str):
    total = 0.0
    pattern = re.compile(r"\[Duration\s+\d+:\s+([\d\.]+)\s+years", re.IGNORECASE)
    for line in text.splitlines():
        m = pattern.search(line)
        if m:
            try:
                total += float(m.group(1))
            except ValueError:
                continue
    return total if total > 0 else None

# ================================
# IMPROVED EXPERIENCE SUBSECTION DETECTION
# ================================

EXCLUDE_SECTION_KEYWORDS = [
    "education", "academic", "certification", "certificate",
    "license", "skill", "summary", "objective", "profile",
    "award", "publication", "language", "interest", "hobby"
]

EXPERIENCE_SECTION_KEYWORDS = [
    "experience", "professional experience", "work experience",
    "employment", "career", "professional background"
]

JOB_TITLE_WORDS = {
    "analyst", "manager", "director", "coordinator", "executive",
    "associate", "lead", "head", "officer", "consultant", "specialist",
    "engineer", "developer", "advisor", "supervisor", "administrator",
    "president", "founder", "co-founder", "partner", "principal",
    "strategist", "planner", "designer", "architect", "scientist",
    "researcher", "intern", "trainee", "representative", "agent",
    "controller", "accountant", "recruiter", "producer", "editor"
}

def subsection_looks_like_job_role(content: str) -> bool:
    """
    Checks if subsection content has signals of a real job role,
    regardless of what the parent section heading says.

    Returns True if duration exists AND at least one of:
    - Job title word found in content
    - 2+ bullet points present
    - Quantitative language (numbers, %, $) present
    """
    has_duration = extract_duration_years_from_augmented_text(content) is not None
    if not has_duration:
        return False

    content_lower = content.lower()

    has_title_word = any(word in content_lower for word in JOB_TITLE_WORDS)

    bullet_count = sum(1 for line in content.splitlines() if line_has_bullet(line))
    has_bullets = bullet_count >= 2

    has_quantitative = bool(re.search(r'(\$[\d,]+|\d+%|\d+[MBK]\b|\d{4,})', content))

    return has_title_word or has_bullets or has_quantitative


def build_resume_experience_subsections(resume_sections):
    """
    Collect experience subsections using a three-gate approach:

    Gate 1 — Must be a subsection (has ' – Subsection' in key)
    Gate 2 — Explicit exclusion: skip if parent heading matches
              education/skills/certifications/etc.
    Gate 3 — Inclusion via two paths:
              (a) Explicit: parent heading matches experience keywords
              (b) Implicit: subsection content itself looks like a job role
                  (has duration + job title word OR bullets OR quantitative language)
    """
    items = []

    for heading, content in resume_sections.items():

        if " – Subsection" not in heading:
            continue

        parent = heading.split(" – ")[0].lower()

        is_excluded = any(kw in parent for kw in EXCLUDE_SECTION_KEYWORDS)
        if is_excluded:
            continue

        is_explicit_experience = any(kw in parent for kw in EXPERIENCE_SECTION_KEYWORDS)
        is_implicit_experience = subsection_looks_like_job_role(content)

        if not is_explicit_experience and not is_implicit_experience:
            continue

        if extract_duration_years_from_augmented_text(content) is None:
            continue

        items.append({
            "full_key": heading,
            "parent_heading": parent,
            "text": content,
        })

    return items


def attach_adjacent_duration_if_missing(resume_items, index):
    this = resume_items[index]
    parent = this["parent_heading"]
    this_dur = extract_duration_years_from_augmented_text(this["text"])
    if this_dur is not None:
        return this_dur

    for i in range(index - 1, -1, -1):
        if resume_items[i]["parent_heading"] != parent:
            continue
        d = extract_duration_years_from_augmented_text(resume_items[i]["text"])
        if d is not None:
            return d

    for i in range(index + 1, len(resume_items)):
        if resume_items[i]["parent_heading"] != parent:
            continue
        d = extract_duration_years_from_augmented_text(resume_items[i]["text"])
        if d is not None:
            return d

    return None

def dedupe_items_by_text(resume_items):
    seen = {}
    deduped = []
    for item in resume_items:
        key = " ".join(item["text"].split()).strip().lower()
        if not key:
            continue
        if key in seen:
            continue
        seen[key] = True
        deduped.append(item)
    return deduped

def compute_matched_experience_years(jd_sections, resume_sections, similarity_threshold=0.25):
    jd_heading, jd_line = find_jd_experience_requirement_line(jd_sections)
    if jd_line is None:
        return {
            "jd_heading": None,
            "jd_line": None,
            "required_years": None,
            "matched_years": 0.0,
            "matched_subsections": [],
        }

    required_years = extract_years_number_from_text(jd_line)

    resume_items = build_resume_experience_subsections(resume_sections)
    if not resume_items:
        return {
            "jd_heading": jd_heading,
            "jd_line": jd_line,
            "required_years": required_years,
            "matched_years": 0.0,
            "matched_subsections": [],
        }

    resume_items = dedupe_items_by_text(resume_items)

    jd_emb = model.encode([jd_line], convert_to_tensor=True)
    resume_texts = [item["text"] for item in resume_items]
    resume_embs = model.encode(resume_texts, convert_to_tensor=True)

    sims = util.cos_sim(jd_emb, resume_embs).cpu().numpy()[0]

    matched_subsections = []
    total_years = 0.0

    for idx, sim_val in enumerate(sims):
        if sim_val < similarity_threshold:
            continue
        dur = attach_adjacent_duration_if_missing(resume_items, idx)
        if dur is None:
            continue
        matched_subsections.append(
            (resume_items[idx]["full_key"], dur, float(sim_val))
        )
        total_years += dur

    return {
        "jd_heading": jd_heading,
        "jd_line": jd_line,
        "required_years": required_years,
        "matched_years": round(total_years, 2),
        "matched_subsections": matched_subsections,
    }

def print_matched_experience_summary(match_info):
    print("\n================ MATCHED EXPERIENCE SUMMARY ================\n")
    if match_info["jd_line"] is None:
        print("No explicit years-of-experience requirement line found in the job description.")
        return

    print(f"JD Experience Requirement (from section '{match_info['jd_heading']}'):\n- {match_info['jd_line']}")
    if match_info["required_years"] is not None:
        print(f"Parsed required years of experience: {match_info['required_years']} years")
    else:
        print("Could not parse a numeric years-of-experience value from the JD line.")

    print(f"\nTotal matched experience (sum of relevant subsections): {match_info['matched_years']} years")

    if not match_info["matched_subsections"]:
        print("No resume subsections found that both semantically match and have usable durations.")
        return

    print("\nMatched resume subsections contributing to this total:")
    for key, dur, sim_val in match_info["matched_subsections"]:
        print(f"- {key} | Duration used: {dur} years | Similarity: {sim_val:.3f}")

# ================================
# LINE-LEVEL MATCHES (grouped + no duration lines)
# ================================
SUBSECTION_LABEL_RE = re.compile(r" – (Subsection \d+)", re.IGNORECASE)

def classify_section_type(base_heading: str) -> str:
    h = base_heading.lower()
    if any(w in h for w in ["experience", "professional experience", "work experience", "services"]):
        return "experience"
    if any(w in h for w in ["education", "academic history"]):
        return "education"
    if any(w in h for w in ["certification", "certifications", "licenses", "license"]):
        return "certifications"
    if any(w in h for w in ["technical skills", "technical skill", "skills", "core skills", "core competencies", "core competence", "core competency", "competencies"]):
        return "skills"
    return "other"

def build_resume_lines_for_matching(resume_sections):
    items = []
    for heading, content in resume_sections.items():
        base_heading = heading.split(" – ")[0] if " – " in heading else heading
        sub_label_match = SUBSECTION_LABEL_RE.search(heading)
        subsection_label = sub_label_match.group(1) if sub_label_match else None
        section_type = classify_section_type(base_heading)

        lines = [l.strip() for l in content.splitlines() if l.strip()]
        for idx, line in enumerate(lines, start=1):
            norm = " ".join(line.split()).strip()
            if not norm:
                continue
            if norm.startswith("[Duration") and "years | Range:" in norm:
                continue
            items.append({
                "section_heading": base_heading,
                "full_heading": heading,
                "subsection_label": subsection_label,
                "section_type": section_type,
                "line_index": idx,
                "line_text": norm,
            })
    return items

def build_jd_lines_for_matching(jd_sections):
    items = []
    for heading, content in jd_sections.items():
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        for idx, line in enumerate(lines, start=1):
            norm = " ".join(line.split()).strip()
            if not norm:
                continue
            items.append({
                "section_heading": heading,
                "line_index": idx,
                "line_text": norm,
            })
    return items

def dedupe_lines_by_text(line_items):
    seen = set()
    out = []
    for item in line_items:
        key = item["line_text"].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out

def compute_and_print_line_level_matches(resume_sections, jd_sections, line_sim_threshold=0.25, max_matches_per_resume_line=2):
    resume_lines = build_resume_lines_for_matching(resume_sections)
    resume_lines = dedupe_lines_by_text(resume_lines)

    jd_lines = build_jd_lines_for_matching(jd_sections)
    jd_lines = dedupe_lines_by_text(jd_lines)

    if not resume_lines or not jd_lines:
        return

    resume_texts = [r["line_text"] for r in resume_lines]
    jd_texts = [j["line_text"] for j in jd_lines]

    resume_embs = model.encode(resume_texts, convert_to_tensor=True)
    jd_embs = model.encode(jd_texts, convert_to_tensor=True)

    sim_matrix = util.cos_sim(resume_embs, jd_embs).cpu().numpy()

    grouped_matches = {
        "experience": [],
        "education": [],
        "certifications": [],
        "skills": [],
        "other": [],
    }

    for i, r_item in enumerate(resume_lines):
        sims = sim_matrix[i]
        indices = sims.argsort()[::-1]
        best = []
        for j in indices:
            if sims[j] < line_sim_threshold:
                break
            best.append((j, sims[j]))
            if len(best) >= max_matches_per_resume_line:
                break
        if not best:
            continue

        grouped_matches[r_item["section_type"]].append((r_item, best))

    print("\n================ LINE-LEVEL RESUME ⇄ JD MATCHES ================\n")

    def print_group(title, items):
        if not items:
            return
        print(f"\n----- {title} -----\n")
        for r_item, best in items:
            sub_label = f" ({r_item['subsection_label']})" if r_item["subsection_label"] else ""
            print(f"RESUME HEADING: **{r_item['section_heading']}**{sub_label} (Full: {r_item['full_heading']})")
            print(f"  Resume line [{r_item['line_index']}]: {r_item['line_text']}")
            print("  Matches:")
            for j, score in best:
                jd_item = jd_lines[j]
                print(f"    - JD line [{jd_item['line_index']}]: {jd_item['line_text']} (Section: **{jd_item['section_heading']}**, Similarity: {score:.3f})")
            print("-" * 80)

    print_group("EXPERIENCE SECTION MATCHES", grouped_matches["experience"])
    print_group("EDUCATION SECTION MATCHES", grouped_matches["education"])
    print_group("CERTIFICATIONS SECTION MATCHES", grouped_matches["certifications"])
    print_group("SKILLS / CORE COMPETENCIES SECTION MATCHES", grouped_matches["skills"])
    print_group("OTHER SECTION MATCHES", grouped_matches["other"])

# ================================
# EDUCATION REQUIREMENT MATCHING
# ================================
EDU_DEGREE_KEYWORDS = [
    "bachelor", "bachelors", "bachelor's", "ba ", "bs ",
    "master", "masters", "master's", "ma ", "ms ",
    "phd", "ph.d", "doctorate", "mba"
]
EDU_SECTION_HINTS = ["education", "academic history", "degree", "qualification"]

def heading_looks_like_education_requirements(heading: str) -> bool:
    h = heading.lower()
    if any(k in h for k in ["education", "qualifications", "requirements", "minimum qualifications", "basic qualifications"]):
        return True
    return False

def jd_line_contains_degree(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in EDU_DEGREE_KEYWORDS)

def find_jd_education_requirement_lines(jd_sections):
    results = []
    for heading, content in jd_sections.items():
        if not heading_looks_like_education_requirements(heading):
            continue
        bullets = split_into_bullets(content)
        if not bullets:
            bullets = [content]
        for b in bullets:
            if jd_line_contains_degree(b):
                results.append((heading, b))

    if not results:
        for heading, content in jd_sections.items():
            bullets = split_into_bullets(content)
            if not bullets:
                bullets = [content]
            for b in bullets:
                if jd_line_contains_degree(b):
                    results.append((heading, b))

    return results

def resume_line_is_education(line_item) -> bool:
    base = line_item["section_heading"].lower()
    if any(h in base for h in EDU_SECTION_HINTS):
        return True
    text = line_item["line_text"].lower()
    if any(k in text for k in EDU_DEGREE_KEYWORDS):
        return True
    return False

def education_degree_similarity(jd_line: str, resume_line: str) -> float:
    jd_emb = model.encode([jd_line], convert_to_tensor=True)
    res_emb = model.encode([resume_line], convert_to_tensor=True)
    sim = util.cos_sim(jd_emb, res_emb).cpu().numpy()[0][0]
    return float(sim)

def compute_matched_education_summary(jd_sections, resume_sections, sim_threshold=0.30):
    jd_edu_requirements = find_jd_education_requirement_lines(jd_sections)
    if not jd_edu_requirements:
        return {
            "jd_requirements": [],
            "matches": [],
        }

    resume_lines = build_resume_lines_for_matching(resume_sections)
    resume_lines = dedupe_lines_by_text(resume_lines)

    edu_resume_lines = [r for r in resume_lines if resume_line_is_education(r)]

    matches = []
    for jd_heading, jd_line in jd_edu_requirements:
        for r_item in edu_resume_lines:
            sim = education_degree_similarity(jd_line, r_item["line_text"])
            if sim >= sim_threshold:
                matches.append({
                    "jd_heading": jd_heading,
                    "jd_line": jd_line,
                    "resume_heading": r_item["section_heading"],
                    "resume_full_heading": r_item["full_heading"],
                    "resume_line_index": r_item["line_index"],
                    "resume_line_text": r_item["line_text"],
                    "similarity": round(sim, 3),
                })

    return {
        "jd_requirements": jd_edu_requirements,
        "matches": matches,
    }

def print_matched_education_summary(edu_info):
    print("\n================ MATCHED EDUCATION SUMMARY ================\n")

    if not edu_info["jd_requirements"]:
        print("No explicit education or degree requirements found in the job description.")
        return

    print("JD Education Requirements:")
    for heading, line in edu_info["jd_requirements"]:
        print(f"- Section '{heading}': {line}")
    print("")

    if not edu_info["matches"]:
        print("No matching education entries found in the resume for these requirements.")
        return

    print("Matched resume education lines:")
    for m in edu_info["matches"]:
        print(f"- JD requirement (Section '{m['jd_heading']}'): {m['jd_line']}")
        print(f"  Resume: [{m['resume_line_index']}] {m['resume_line_text']} (Section: '{m['resume_full_heading']}', Similarity: {m['similarity']})")
        print("-" * 80)

# ================================
# SHARED SCORING HELPERS
# ================================

def detect_outcome_strength(text: str):
    doc = nlp(text)
    has_number = any(t.pos_ == "NUM" for t in doc)
    has_money = any(ent.label_ == "MONEY" for ent in doc.ents)
    outcome_verbs = {"increase", "improve", "grow", "optimize", "reduce", "save", "deliver", "achieve", "launch", "ship", "close"}
    has_outcome_verb = any(t.lemma_.lower() in outcome_verbs for t in doc if t.pos_ == "VERB")

    score = 0.0
    if has_number:
        score += 0.4
    if has_money:
        score += 0.3
    if has_outcome_verb:
        score += 0.4
    return min(score, 1.0)

def detect_keyword_and_ats_features(text: str):
    doc = nlp(text)
    tokens = [t for t in doc if not t.is_space]
    length_factor = min(len(tokens) / 40.0, 1.0)
    has_date = any(ent.label_ == "DATE" for ent in doc.ents)
    keywords = {"experience", "skill", "project", "lead", "managed", "responsible", "delivered"}
    has_keywords = any(t.lemma_.lower() in keywords for t in doc)
    score = 0.3 * length_factor + (0.35 if has_date else 0.0) + (0.35 if has_keywords else 0.0)
    return min(score, 1.0)

def detect_logistical_fit(text: str):
    t = text.lower()
    remote_signals = any(k in t for k in ["remote", "hybrid"])
    timezone_signals = any(k in t for k in ["est", "eastern time", "cst", "pst"])
    location_signals = any(k in t for k in ["usa", "united states", "atlanta", "georgia"])
    score = 0.0
    if remote_signals:
        score += 0.4
    if timezone_signals:
        score += 0.3
    if location_signals:
        score += 0.4
    return min(score, 1.0)

# ================================
# JD REQUIREMENT vs PREFERRED LINE CLASSIFICATION
# ================================

def classify_jd_requirement_lines(jd_sections):
    """
    Split JD bullet lines into 'required' and 'preferred' groups based on
    section heading. Used for weighted skills coverage (required weighted
    higher than preferred).
    """
    required_lines = []
    preferred_lines = []

    for heading, content in jd_sections.items():
        h = heading.lower()
        bullets = split_into_bullets(content)
        if not bullets:
            bullets = [content]
        bullets = [b for b in bullets if b.strip()]

        if "preferred" in h:
            preferred_lines.extend(bullets)
        elif any(k in h for k in [
            "required", "requirement", "qualification",
            "responsibilit", "duties", "key responsibilities", "job role"
        ]):
            required_lines.extend(bullets)

    # Dedupe while preserving order
    def dedupe(lines):
        seen = set()
        out = []
        for l in lines:
            key = l.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(l.strip())
        return out

    return dedupe(required_lines), dedupe(preferred_lines)

# ================================
# GLOBAL (WHOLE-RESUME) ROLE FIT SCORE
# ================================

def compute_skills_coverage_global(resume_sections, jd_sections):
    """
    S = Skills coverage (0-100), weighted by required vs preferred JD lines.
    For each JD requirement/preferred line, find its best semantic match
    anywhere in the resume. Average those best-matches, weighting required
    lines more heavily (0.7) than preferred lines (0.3).
    """
    required_lines, preferred_lines = classify_jd_requirement_lines(jd_sections)

    resume_lines = build_resume_lines_for_matching(resume_sections)
    resume_lines = dedupe_lines_by_text(resume_lines)
    resume_texts = [r["line_text"] for r in resume_lines]

    if not resume_texts:
        return 0.0

    resume_embs = model.encode(resume_texts, convert_to_tensor=True)

    def avg_best_match(jd_lines):
        if not jd_lines:
            return None
        jd_embs = model.encode(jd_lines, convert_to_tensor=True)
        sims = util.cos_sim(jd_embs, resume_embs).cpu().numpy()
        best_per_line = sims.max(axis=1)
        return float(best_per_line.mean())

    req_score = avg_best_match(required_lines)
    pref_score = avg_best_match(preferred_lines)

    if req_score is not None and pref_score is not None:
        combined = 0.7 * req_score + 0.3 * pref_score
    elif req_score is not None:
        combined = req_score
    elif pref_score is not None:
        combined = pref_score
    else:
        combined = 0.0

    return max(0.0, min(combined * 100.0, 100.0))


def compute_evidence_strength_global(resume_sections):
    """
    E = Evidence strength (0-100). Average outcome-strength signal
    (numbers, money, outcome verbs) across all detected experience
    subsections. Falls back to all subsections if none qualify.
    """
    items = build_resume_experience_subsections(resume_sections)
    if items:
        texts = [it["text"] for it in items]
    else:
        texts = [c for h, c in resume_sections.items() if " – Subsection" in h]

    if not texts:
        return 0.0

    scores = [detect_outcome_strength(t) for t in texts]
    return max(0.0, min((sum(scores) / len(scores)) * 100.0, 100.0))


def compute_experience_alignment_global(match_info):
    """
    X = Experience alignment (0-100). Ratio of total matched years to
    required years, capped at 100. Returns (score, applicable).
    """
    required_years = match_info.get("required_years")
    matched_years = match_info.get("matched_years", 0.0)

    if required_years is None or required_years <= 0:
        return 0.0, False

    ratio = min(matched_years / required_years, 1.0)
    # Non-linear penalty: shortfalls in years hurt more than a linear
    # scaling would suggest (e.g. 70% of required years -> 49, not 70).
    # Ratios >= 1.0 (met or exceeded requirement) still map to 100.
    score = max(0.0, min((ratio ** 2) * 100.0, 100.0))
    return score, True


def extract_jd_keywords(jd_sections):
    """
    Extract a set of significant keywords/terms from the JD to be checked
    for presence in the resume. Includes:
      - Noun chunks (multi-word terms like "data warehouse", "machine learning")
      - Proper nouns and acronyms/tech terms (e.g. "SQL", "C#", "ETL", "Java")
    Stopword-only chunks and very short/common words are filtered out.
    """
    full_text = "\n".join(jd_sections.values())
    if not full_text.strip():
        return set()

    doc = nlp(full_text)

    keywords = set()

    # Multi-word noun chunks (skip pure-stopword chunks, keep meaningful terms)
    for chunk in doc.noun_chunks:
        text = chunk.text.strip().lower()
        text = re.sub(r"^(the|a|an|this|that|these|those)\s+", "", text)
        if not text or len(text) < 3:
            continue
        if all(t.is_stop for t in chunk):
            continue
        keywords.add(text)

    # Individual tokens that look like technical terms / proper nouns /
    # acronyms (e.g. SQL, ETL, C#, Java, Python, Spark)
    for t in doc:
        tok = t.text.strip()
        if len(tok) < 2:
            continue
        if t.is_stop or t.is_punct:
            continue
        # Acronyms / tech terms: all-caps, contains digit, or proper noun
        if tok.isupper() or any(ch.isdigit() for ch in tok) or t.pos_ == "PROPN":
            keywords.add(tok.lower())

    return keywords


def compute_ats_alignment_global(resume_sections, jd_sections):
    """
    K = Keyword/ATS alignment (0-100).

    Compares the JD's significant keywords/terms (noun chunks, tech terms,
    acronyms, proper nouns) against the resume text. Score is the
    percentage of JD keywords that appear (as substrings) anywhere in the
    resume — i.e. how many of the JD's important terms show up on the
    candidate's resume, which is what an ATS keyword scan checks for.
    """
    jd_keywords = extract_jd_keywords(jd_sections)
    if not jd_keywords:
        return 0.0

    resume_text = "\n".join(resume_sections.values()).lower()
    if not resume_text.strip():
        return 0.0

    matched = sum(1 for kw in jd_keywords if kw in resume_text)
    score = (matched / len(jd_keywords)) * 100.0
    return max(0.0, min(score, 100.0))


def compute_logistical_fit_global(resume_sections):
    """
    L = Logistical fit (0-100), computed once over the resume as a whole
    (location/remote/timezone cues).
    """
    full_text = "\n".join(resume_sections.values())
    if not full_text.strip():
        return 0.0
    return max(0.0, min(detect_logistical_fit(full_text) * 100.0, 100.0))


def compute_overall_role_fit_score(resume_sections, jd_sections, match_info):
    """
    Single overall Role Fit Score (0-100) for the whole resume vs JD,
    using the rubric:

        Score = 0.35*S + 0.25*E + 0.20*X + 0.10*K + 0.10*L

    where S, E, K, L are computed once across the whole resume/JD, and
    X is the experience-years alignment from compute_matched_experience_years.

    If X is not applicable (JD has no parseable years-of-experience
    requirement), its 0.20 weight is excluded from both numerator and
    denominator and the score is renormalized over the remaining weights.
    """
    S = compute_skills_coverage_global(resume_sections, jd_sections)
    E = compute_evidence_strength_global(resume_sections)
    X, experience_applicable = compute_experience_alignment_global(match_info)
    K = compute_ats_alignment_global(resume_sections, jd_sections)
    L = compute_logistical_fit_global(resume_sections)

    S_W, E_W, X_W, K_W, L_W = 0.35, 0.25, 0.20, 0.10, 0.10

    raw_score = S_W * S + E_W * E + K_W * K + L_W * L
    total_weight = S_W + E_W + K_W + L_W

    if experience_applicable:
        raw_score += X_W * X
        total_weight += X_W

    overall = (raw_score / total_weight) if total_weight > 0 else 0.0

    return {
        "overall_score": round(overall, 1),
        "skills_score": round(S, 1),
        "evidence_score": round(E, 1),
        "experience_score": round(X, 1),
        "experience_applicable": experience_applicable,
        "keyword_score": round(K, 1),
        "logistical_score": round(L, 1),
        "required_years": match_info.get("required_years"),
        "matched_years": match_info.get("matched_years"),
    }


def print_overall_role_fit_score(result):
    print("\n================ OVERALL ROLE FIT SCORE ================\n")
    print(f"Overall Role Fit Score (0–100): {result['overall_score']}")
    print("\nFormula: Score = 0.35*S + 0.25*E + 0.20*X + 0.10*K + 0.10*L")
    print("  where S = Skills coverage, E = Evidence strength, X = Experience alignment,")
    print("        K = Keyword/ATS alignment, L = Logistical fit.")
    print("\nComponent breakdown (each 0-100, computed once for the whole resume vs JD):")
    print(f"  - Skills coverage (S, weight 0.35): {result['skills_score']}")
    print(f"      Weighted average of best semantic match between each JD")
    print(f"      required/preferred line and the resume (required lines")
    print(f"      weighted 0.7, preferred lines weighted 0.3).")
    print(f"  - Evidence strength (E, weight 0.25): {result['evidence_score']}")
    print(f"      Average presence of numbers, money amounts, and outcome")
    print(f"      verbs across the resume's experience subsections.")

    if result["experience_applicable"]:
        print(f"  - Experience alignment (X, weight 0.20): {result['experience_score']}")
        print(f"      {result['matched_years']} matched years vs "
              f"{result['required_years']} years required in the JD.")
        print(f"      X = (matched/required)^2 * 100, capped at 100 — shortfalls")
        print(f"      are penalized more steeply than a straight ratio.")
    else:
        print(f"  - Experience alignment (X, weight 0.20): not applicable")
        print(f"      No parseable years-of-experience requirement found in the JD,")
        print(f"      so this dimension was excluded and the score was renormalized")
        print(f"      over the remaining weights.")

    print(f"  - Keyword / ATS alignment (K, weight 0.10): {result['keyword_score']}")
    print(f"      Percentage of significant JD keywords/terms (noun phrases,")
    print(f"      tech terms, acronyms, proper nouns) that also appear")
    print(f"      anywhere in the resume.")
    print(f"  - Logistical fit (L, weight 0.10): {result['logistical_score']}")
    print(f"      Based on location/remote/timezone cues found anywhere")
    print(f"      in the resume.")
    print("-" * 80)

# ================================
# PER-SUBSECTION RELEVANCE (informational only)
# ================================

def compute_subsection_relevance_scores(resume_sections, jd_sections):
    """
    Lightweight, informational per-subsection relevance score.

    Unlike the overall Role Fit Score, this is NOT the weighted
    5-dimension rubric (skills/evidence/experience/keyword/logistical) —
    several of those dimensions (experience alignment, logistical fit,
    ATS alignment) are properties of the resume/candidate as a whole and
    don't meaningfully apply to a single subsection.

    Instead, each subsection gets:
      - relevance_score: best semantic match (0-100) against any JD
        required/preferred line — "how relevant is this entry to the role"
      - evidence_score: outcome-strength signal (0-100) for that entry
      - duration_years: parsed duration, if any, for context
    """
    required_lines, preferred_lines = classify_jd_requirement_lines(jd_sections)
    all_jd_lines = required_lines + preferred_lines

    if all_jd_lines:
        jd_embs = model.encode(all_jd_lines, convert_to_tensor=True)
    else:
        jd_embs = None

    results = []
    for heading, content in resume_sections.items():
        if " – Subsection" not in heading:
            continue

        if jd_embs is not None:
            res_emb = model.encode([content], convert_to_tensor=True)
            sims = util.cos_sim(res_emb, jd_embs).cpu().numpy()[0]
            relevance = float(sims.max()) * 100.0
        else:
            relevance = 0.0

        evidence = detect_outcome_strength(content) * 100.0
        duration_years = extract_duration_years_from_augmented_text(content)

        results.append({
            "heading": heading,
            "relevance_score": round(max(0.0, min(relevance, 100.0)), 1),
            "evidence_score": round(max(0.0, min(evidence, 100.0)), 1),
            "duration_years": duration_years,
        })

    return results


def print_subsection_relevance_scores(subsection_scores):
    print("\n================ SUBSECTION RELEVANCE (INFORMATIONAL) ================\n")
    if not subsection_scores:
        print("No subsections found.")
        return

    print("Note: these are informational relevance scores, not the overall")
    print("Role Fit Score. The overall Role Fit Score (printed separately)")
    print("is computed once for the whole resume vs the whole JD.\n")

    for item in subsection_scores:
        dur_str = f"{item['duration_years']} years" if item["duration_years"] is not None else "n/a"
        print(f"Subsection: {item['heading']}")
        print(f"  Relevance to JD (best match): {item['relevance_score']} / 100")
        print(f"  Evidence strength: {item['evidence_score']} / 100")
        print(f"  Duration: {dur_str}")
        print("-" * 80)

# ================================
# MAIN APPLICATION
# ================================
def process_files(resume_path, jd_path):
    resume_extracted = extract_text(resume_path)
    if isinstance(resume_extracted, tuple):
        resume_blocks, _ = resume_extracted
    else:
        resume_blocks = resume_extracted

    jd_extracted = extract_text(jd_path)
    if isinstance(jd_extracted, tuple):
        jd_blocks, _ = jd_extracted
    else:
        jd_blocks = jd_extracted

    resume_sections = semantic_sectioning(resume_blocks, for_jd=False)
    jd_sections = semantic_sectioning(jd_blocks, for_jd=True)

    resume_sections = add_bold_subsections_to_all_sections(resume_sections)

    resume_features = extract_section_spacy_features(resume_sections)
    jd_features = extract_section_spacy_features(jd_sections)

    print("\n================ RESUME SECTIONS (WITH SUBSECTIONS) ================\n")
    for heading, content in resume_sections.items():
        print(f"\n### {heading.upper()} ###\n{content}\n")
        print("=" * 60)

    print("\n================ JOB DESCRIPTION SECTIONS ================\n")
    for heading, content in jd_sections.items():
        print(f"\n### {heading.upper()} ###\n{content}\n")
        print("=" * 60)

    # ----- MATCHED EXPERIENCE SUMMARY -----
    match_info = compute_matched_experience_years(jd_sections, resume_sections)
    print_matched_experience_summary(match_info)

    # ----- MATCHED EDUCATION SUMMARY -----
    edu_info = compute_matched_education_summary(jd_sections, resume_sections)
    print_matched_education_summary(edu_info)

    # ----- LINE-LEVEL MATCHES -----
    compute_and_print_line_level_matches(resume_sections, jd_sections)

    # ----- PER-SUBSECTION RELEVANCE (informational) -----
    subsection_scores = compute_subsection_relevance_scores(resume_sections, jd_sections)
    print_subsection_relevance_scores(subsection_scores)

    # ----- OVERALL ROLE FIT SCORE (single, whole-resume vs whole-JD) -----
    overall_result = compute_overall_role_fit_score(resume_sections, jd_sections, match_info)
    print_overall_role_fit_score(overall_result)



# In[10]:


# ================================
# CELL 9 — BEDROCK LLM SUGGESTIONS (Amazon Nova Lite) — UPDATED
# ================================
import boto3
import json

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)
import boto3 as _boto3_s3
S3_BUCKET = "resume-intelligence-zenith"
s3_client = _boto3_s3.client("s3", region_name="us-east-1")

def upload_to_s3(local_path, resume_id):
    filename = os.path.basename(local_path)
    s3_key = f"resumes/{resume_id}_{filename}"
    s3_client.upload_file(local_path, S3_BUCKET, s3_key)
    return f"s3://{S3_BUCKET}/{s3_key}"

def download_from_s3_to_temp(s3_uri):
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    suffix = os.path.splitext(key)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    s3_client.download_file(bucket, key, tmp.name)
    return tmp.name

def get_llm_suggestions(overall_result, subsection_scores, resume_sections, jd_sections):
    """
    overall_result: dict returned by compute_overall_role_fit_score()
    subsection_scores: list returned by compute_subsection_relevance_scores()
    """

    # Lowest-relevance subsections
    sorted_subs = sorted(subsection_scores, key=lambda x: x["relevance_score"])
    weak_subsections = sorted_subs[:3]

    # JD requirement/preferred lines (your Cell 8 function)
    required_lines, preferred_lines = classify_jd_requirement_lines(jd_sections)
    jd_skills_text = "\n".join(
        f"- {l}" for l in (required_lines + preferred_lines)[:15]
    )

    # Resume skills section text
    resume_skills_text = ""
    for heading, content in resume_sections.items():
        if "skill" in heading.lower() and " – Subsection" not in heading:
            resume_skills_text = content[:500]
            break

    # Weak subsections summary
    weak_summary = ""
    for sub in weak_subsections:
        dur = f"{sub['duration_years']} years" if sub['duration_years'] is not None else "n/a"
        weak_summary += f"""
Subsection: {sub['heading']}
Relevance to JD: {sub['relevance_score']}/100
Evidence strength: {sub['evidence_score']}/100
Duration: {dur}
"""

    required_years = overall_result.get("required_years")
    matched_years = overall_result.get("matched_years", 0)
    if overall_result.get("experience_applicable") and required_years:
        exp_gap = f"JD requires {required_years} years. Resume shows {matched_years} years matched."
    else:
        exp_gap = "No explicit years-of-experience requirement detected in JD."

    overall_score = overall_result["overall_score"]

    prompt = f"""You are an expert resume coach and ATS optimization specialist.

A candidate has submitted their resume for a job. Here is the analysis:

OVERALL ROLE FIT SCORE: {overall_score}/100
  - Skills coverage (S): {overall_result['skills_score']}/100
  - Evidence strength (E): {overall_result['evidence_score']}/100
  - Experience alignment (X): {overall_result['experience_score']}/100
  - Keyword/ATS alignment (K): {overall_result['keyword_score']}/100
  - Logistical fit (L): {overall_result['logistical_score']}/100

EXPERIENCE: {exp_gap}

JOB DESCRIPTION KEY REQUIREMENTS:
{jd_skills_text}

CANDIDATE RESUME SKILLS SECTION:
{resume_skills_text}

LOWEST-RELEVANCE RESUME SUBSECTIONS (least aligned to JD):
{weak_summary}

Based on this analysis, provide SPECIFIC improvement suggestions.
Your response must be valid JSON only. No extra text before or after.
Use exactly this structure:

{{
  "summary": "2 sentence overall assessment",
  "overall_score": {overall_score},
  "improvements": [
    {{
      "section": "section name",
      "issue": "specific problem identified",
      "suggestion": "specific actionable fix",
      "jd_requirement": "the JD requirement this addresses"
    }}
  ],
  "missing_keywords": ["keyword1", "keyword2", "keyword3"],
  "ats_pass_probability": "low/medium/high",
  "top_strength": "the strongest matching area"
}}

Provide 3-5 specific improvements. Be concrete, not generic.
Return valid JSON only. No markdown. No explanation. Just the JSON."""

    body = json.dumps({
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": 1500, "temperature": 0.3}
    })

    try:
        response = bedrock.invoke_model(
            modelId="amazon.nova-lite-v1:0",
            body=body,
            contentType="application/json",
            accept="application/json"
        )
        response_body = json.loads(response["body"].read())
        raw_text = response_body["output"]["message"]["content"][0]["text"]

        clean_text = raw_text.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.split("```")[1]
            if clean_text.startswith("json"):
                clean_text = clean_text[4:]
        clean_text = clean_text.strip()

        return json.loads(clean_text)

    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Raw response: {raw_text}")
        return {"error": "Failed to parse LLM response", "raw": raw_text}
    except Exception as e:
        print(f"Bedrock API error: {e}")
        return {"error": str(e)}


def print_llm_suggestions(suggestions):
    print("\n================ LLM IMPROVEMENT SUGGESTIONS ================\n")
    if "error" in suggestions:
        print(f"Error: {suggestions['error']}")
        return
    print(f"Overall Score: {suggestions.get('overall_score')}/100")
    print(f"\nSummary: {suggestions.get('summary')}")
    print(f"\nATS Pass Probability: {suggestions.get('ats_pass_probability', 'unknown').upper()}")
    print(f"Top Strength: {suggestions.get('top_strength', 'N/A')}")
    print("\nMissing Keywords:")
    for kw in suggestions.get("missing_keywords", []):
        print(f"  • {kw}")
    print("\nSpecific Improvements:")
    for i, imp in enumerate(suggestions.get("improvements", []), 1):
        print(f"\n  {i}. Section: {imp.get('section')}")
        print(f"     Issue: {imp.get('issue')}")
        print(f"     Fix: {imp.get('suggestion')}")
        print(f"     JD Requirement: {imp.get('jd_requirement')}")
        print("-" * 60)


# In[11]:


# ================================
# CELL 10 — UPDATED MAIN WITH LLM
# ================================

def process_files_with_llm(resume_path, jd_path):
    resume_extracted = extract_text(resume_path)
    resume_blocks = resume_extracted[0] if isinstance(resume_extracted, tuple) else resume_extracted

    jd_extracted = extract_text(jd_path)
    jd_blocks = jd_extracted[0] if isinstance(jd_extracted, tuple) else jd_extracted

    resume_sections = semantic_sectioning(resume_blocks, for_jd=False)
    jd_sections = semantic_sectioning(jd_blocks, for_jd=True)
    resume_sections = add_bold_subsections_to_all_sections(resume_sections)

    match_info = compute_matched_experience_years(jd_sections, resume_sections)
    edu_info = compute_matched_education_summary(jd_sections, resume_sections)
    subsection_scores = compute_subsection_relevance_scores(resume_sections, jd_sections)
    overall_result = compute_overall_role_fit_score(resume_sections, jd_sections, match_info)

    print_matched_experience_summary(match_info)
    print_matched_education_summary(edu_info)
    compute_and_print_line_level_matches(resume_sections, jd_sections)
    print_subsection_relevance_scores(subsection_scores)
    print_overall_role_fit_score(overall_result)

    print("\nCalling Bedrock Nova Lite for improvement suggestions...")
    suggestions = get_llm_suggestions(overall_result, subsection_scores, resume_sections, jd_sections)
    print_llm_suggestions(suggestions)

    return {
        "overall_role_fit_score": overall_result["overall_score"],
        "score_breakdown": overall_result,
        "matched_experience_years": match_info["matched_years"],
        "required_experience_years": match_info["required_years"],
        "education_match": len(edu_info["matches"]) > 0,
        "subsection_scores": subsection_scores,
        "llm_suggestions": suggestions
    }







# In[12]:


# ================================
# CELL 11 — FAISS STORE AND RETRIEVE
# ================================
import faiss
import numpy as np
import uuid
import boto3
import json
from decimal import Decimal

# Initialize DynamoDB
dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
table = dynamodb.Table("resume_intelligence")

# ================================
# HELPER — convert floats for DynamoDB
# DynamoDB does not accept Python floats
# so we convert everything to Decimal
# ================================
def float_to_decimal(obj):
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, list):
        return [float_to_decimal(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: float_to_decimal(v) for k, v in obj.items()}
    return obj

def decimal_to_float(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, list):
        return [decimal_to_float(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: decimal_to_float(v) for k, v in obj.items()}
    return obj

# ================================
# MODE 2 — STORE RESUME
# Called after your pipeline runs on a resume
# ================================
def store_resume(
    resume_path,
    resume_sections,
    overall_score,
    match_info,
    edu_info,
    subsection_scores,
    suggestions
):
    """
    Embeds the resume text and stores everything
    in DynamoDB — text, scores, vector, suggestions.
    """

    # ── Step 1: Build full resume text for embedding ──
    full_text = " ".join([
        content for heading, content in resume_sections.items()
        if " – Subsection" not in heading
    ])

    # ── Step 2: Generate embedding using your existing model ──
    # model is already loaded in Cell 1 as:
    # model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embedding = model.encode([full_text])[0]
    vector_list = embedding.tolist()

    # ── Step 3: Extract candidate name from resume sections ──
    candidate_name = "Unknown"
    for heading, content in resume_sections.items():
        h = heading.lower()
        if "contact" in h or "personal" in h:
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            if lines:
                candidate_name = lines[0]
            break

    # ── Step 4: Extract skills ──
    skills = []
    for heading, content in resume_sections.items():
        if "skill" in heading.lower() and " – Subsection" not in heading:
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            skills = lines[:20]
            break

    # ── Step 5: Generate unique resume ID ──
    resume_id = str(uuid.uuid4())
    # ── Upload original file to S3 ──
    if os.path.exists(resume_path) and not resume_path.startswith("s3://"):
        try:
            resume_path = upload_to_s3(resume_path, resume_id)
        except Exception as e:
            print(f"  ⚠ S3 upload failed: {e}")

    # ── Step 6: Build DynamoDB item ──
    item = {
        "resume_id": resume_id,
        "candidate_name": candidate_name,
        "s3_path": resume_path,
        "skills": skills,
        "overall_score": overall_score if overall_score else 0.0,
        "matched_experience_years": match_info.get("matched_years", 0.0),
        "required_experience_years": match_info.get("required_years") or 0.0,
        "education_match": len(edu_info.get("matches", [])) > 0,
        "vector": vector_list,
        "llm_suggestions": suggestions,
        "subsection_scores": subsection_scores
    }

    # Convert all floats to Decimal for DynamoDB
    item = float_to_decimal(item)

    # ── Step 7: Save to DynamoDB ──
    table.put_item(Item=item)

    print(f"\n✓ Resume stored successfully")
    print(f"  Resume ID: {resume_id}")
    print(f"  Candidate: {candidate_name}")
    print(f"  Overall Score: {overall_score}")
    print(f"  Vector dimensions: {len(vector_list)}")

    return resume_id


# ================================
# MODE 3 — SEARCH RESUMES BY JD
# Called when only a JD is uploaded
# ================================
def search_resumes_by_jd(jd_sections, top_k=10):
    """
    Embeds the JD, loads all resume vectors from DynamoDB,
    builds a FAISS index in memory, searches for top matches,
    returns top 10 resume IDs for detailed scoring.
    """

    print("\nSearching stored resumes...")

    # ── Step 1: Embed the JD ──
    jd_text = " ".join([
        content for heading, content in jd_sections.items()
    ])
    jd_embedding = model.encode([jd_text])[0]
    jd_vector = np.array([jd_embedding], dtype=np.float32)

    # ── Step 2: Load all resumes from DynamoDB ──
    response = table.scan()
    items = response["Items"]

    # Handle DynamoDB pagination
    while "LastEvaluatedKey" in response:
        response = table.scan(
            ExclusiveStartKey=response["LastEvaluatedKey"]
        )
        items.extend(response["Items"])

    if not items:
        print("No resumes found in database.")
        return []

    print(f"  Found {len(items)} resumes in database")

    # ── Step 3: Build FAISS index in memory ──
    vectors = []
    resume_ids = []

    for item in items:
        vector = item.get("vector")
        if not vector:
            continue
        # Convert Decimal back to float
        vector_floats = [float(v) for v in vector]
        vectors.append(vector_floats)
        resume_ids.append(str(item["resume_id"]))

    if not vectors:
        print("No vectors found in database.")
        return []

    # Build FAISS index
    dimension = len(vectors[0])
    index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity

    # Normalize vectors for cosine similarity
    vectors_array = np.array(vectors, dtype=np.float32)
    faiss.normalize_L2(vectors_array)
    faiss.normalize_L2(jd_vector)

    # Add all resume vectors to index
    index.add(vectors_array)

    # ── Step 4: Search for top matches ──
    actual_k = min(top_k, len(vectors))
    distances, indices = index.search(jd_vector, actual_k)

    # ── Step 5: Build results ──
    results = []
    for rank, (dist, idx) in enumerate(
        zip(distances[0], indices[0]), start=1
    ):
        if idx == -1:
            continue
        resume_id = resume_ids[idx]
        item = next(
            (i for i in items if str(i["resume_id"]) == resume_id),
            None
        )
        if not item:
            continue

        results.append({
            "rank": rank,
            "resume_id": resume_id,
            "candidate_name": str(item.get("candidate_name", "Unknown")),
            "similarity_score": round(float(dist) * 100, 2),
            "overall_score": float(item.get("overall_score", 0)),
            "skills": [str(s) for s in item.get("skills", [])],
            "matched_experience_years": float(
                item.get("matched_experience_years", 0)
            ),
            "s3_path": str(item.get("s3_path", ""))
        })

    print(f"  Top {len(results)} matches found")
    return results


# ================================
# PRINT SEARCH RESULTS
# ================================
def print_search_results(results):
    print("\n================ TOP RESUME MATCHES ================\n")

    if not results:
        print("No matches found.")
        return

    for r in results:
        print(f"Rank {r['rank']}: {r['candidate_name']}")
        print(f"  Similarity Score: {r['similarity_score']}%")
        print(f"  Overall Role Fit: {r['overall_score']}/100")
        print(f"  Experience: {r['matched_experience_years']} years")
        print(f"  Skills: {', '.join(r['skills'][:5])}")
        print("-" * 60)


# In[ ]:

# In[7]:


# ================================
# CELL 12 — MODE 2: BULK STORE ALL RESUMES
# ================================
import os
import time

# ── Clean file lists ──
notebook_dir = os.getcwd()

# JDs — anything starting with "JD" or "jd"

# Resumes — everything else except known non-resumes


# ================================
# MODE 2 — STORE SINGLE RESUME
# ================================
def mode2_store_resume(resume_path):
    """
    Runs the full pipeline on one resume
    and stores everything in DynamoDB.
    No JD needed — just parsing, embedding, storing.
    """
    print(f"\nProcessing: {os.path.basename(resume_path)}")
    print("-" * 50)

    try:
        # ── Step 1: Extract text ──
        resume_extracted = extract_text(resume_path)
        if isinstance(resume_extracted, tuple):
            resume_blocks, _ = resume_extracted
        else:
            resume_blocks = resume_extracted

        if not resume_blocks:
            print(f"  ✗ No text extracted — skipping")
            return None

        # ── Step 2: Section the resume ──
        resume_sections = semantic_sectioning(
            resume_blocks, for_jd=False
        )
        resume_sections = add_bold_subsections_to_all_sections(
            resume_sections
        )

        if not resume_sections:
            print(f"  ✗ No sections detected — skipping")
            return None

        # ── Step 3: Score without JD ──
        # No JD available so scores will be 0
        # They get updated when Mode 1 runs later
        subsection_scores = []
        overall_score = 0.0
        match_info = {
            "jd_heading": None,
            "jd_line": None,
            "required_years": None,
            "matched_years": 0.0,
            "matched_subsections": []
        }
        edu_info = {
            "jd_requirements": [],
            "matches": []
        }

        # ── Step 4: Store in DynamoDB ──
        resume_id = store_resume(
            resume_path=resume_path,
            resume_sections=resume_sections,
            overall_score=overall_score,
            match_info=match_info,
            edu_info=edu_info,
            subsection_scores=subsection_scores,
            suggestions={}
        )

        return resume_id

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


# ================================
# BULK LOAD ALL RESUMES
# ================================
def bulk_store_all_resumes():
    """
    Processes all resume files in the notebook
    directory and stores them in DynamoDB.
    """
    print("=" * 60)
    print("BULK LOADING ALL RESUMES INTO DYNAMODB")
    print("=" * 60)
    print(f"Total resumes to process: {len(RESUME_FILES)}\n")

    successful = []
    failed = []

    for i, filename in enumerate(RESUME_FILES, start=1):
        full_path = os.path.join(notebook_dir, filename)
        print(f"[{i}/{len(RESUME_FILES)}] {filename}")

        resume_id = mode2_store_resume(full_path)

        if resume_id:
            successful.append(filename)
        else:
            failed.append(filename)

        # Small delay to avoid overwhelming DynamoDB
        time.sleep(0.5)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("BULK LOAD COMPLETE")
    print("=" * 60)
    print(f"✓ Successfully stored: {len(successful)}")
    print(f"✗ Failed: {len(failed)}")

    if failed:
        print("\nFailed files:")
        for f in failed:
            print(f"  - {f}")

    return successful, failed


# ── Run the bulk loader ──



# In[8]:



# In[13]:


# ================================
# CELL 13 — MODE 1: RESUME + JD TOGETHER — UPDATED
# ================================
import re as _re

def mode1_resume_and_jd(resume_path, jd_path):
    print("=" * 60)
    print("MODE 1 — RESUME + JD PROCESSING")
    print("=" * 60)
    print(f"Resume: {os.path.basename(resume_path)}")
    print(f"JD:     {os.path.basename(jd_path)}")
    print()

    print("Step 1: Extracting text...")
    resume_extracted = extract_text(resume_path)
    resume_blocks = resume_extracted[0] if isinstance(resume_extracted, tuple) else resume_extracted

    jd_extracted = extract_text(jd_path)
    jd_blocks = jd_extracted[0] if isinstance(jd_extracted, tuple) else jd_extracted

    print("Step 2: Sectioning documents...")
    resume_sections = semantic_sectioning(resume_blocks, for_jd=False)
    jd_sections = semantic_sectioning(jd_blocks, for_jd=True)
    resume_sections = add_bold_subsections_to_all_sections(resume_sections)

    print("Step 3: Running scoring pipeline...")
    match_info = compute_matched_experience_years(jd_sections, resume_sections)
    edu_info = compute_matched_education_summary(jd_sections, resume_sections)
    subsection_scores = compute_subsection_relevance_scores(resume_sections, jd_sections)
    overall_result = compute_overall_role_fit_score(resume_sections, jd_sections, match_info)
    overall_score = overall_result["overall_score"]

    print("Step 4: Calling Bedrock for LLM suggestions...")
    suggestions = get_llm_suggestions(overall_result, subsection_scores, resume_sections, jd_sections)

    # ── Candidate name extraction ──
    candidate_name = "Unknown"
    for heading, content in resume_sections.items():
        h = heading.lower()
        if "contact" in h or "personal" in h:
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            if lines:
                candidate_name = lines[0]
            break
    if candidate_name == "Unknown":
        for block in resume_blocks:
            line = block.strip()
            if line and len(line) < 50 and not any(
                kw in line.lower() for kw in [
                    "resume", "cv", "curriculum", "page", "http", "@", "phone"
                ]
            ):
                candidate_name = line
                break

    candidate_name = _re.sub(r'[#*_]', '', candidate_name).strip()

    # ── Skills extraction ──
    skills = []
    for heading, content in resume_sections.items():
        if "skill" in heading.lower() and " – Subsection" not in heading:
            skills = [l.strip() for l in content.splitlines() if l.strip()][:20]
            break

    # ── Save to DynamoDB ──
    print("Step 5: Saving to DynamoDB...")
    existing = table.scan(
        FilterExpression="s3_path = :path",
        ExpressionAttributeValues={":path": resume_path}
    )

    if existing["Items"]:
        resume_id = existing["Items"][0]["resume_id"]
        table.update_item(
            Key={"resume_id": resume_id},
            UpdateExpression="""
                SET overall_score = :score,
                    matched_experience_years = :exp,
                    required_experience_years = :req,
                    education_match = :edu,
                    skills = :skills,
                    candidate_name = :name,
                    llm_suggestions = :suggestions,
                    subsection_scores = :subs,
                    score_breakdown = :breakdown
            """,
            ExpressionAttributeValues=float_to_decimal({
                ":score": overall_score or 0.0,
                ":exp": match_info.get("matched_years", 0.0),
                ":req": match_info.get("required_years") or 0.0,
                ":edu": len(edu_info.get("matches", [])) > 0,
                ":skills": skills,
                ":name": candidate_name,
                ":suggestions": suggestions,
                ":subs": subsection_scores,
                ":breakdown": overall_result
            })
        )
        print(f"  ✓ Updated existing resume record")
    else:
        resume_id = store_resume(
            resume_path=resume_path,
            resume_sections=resume_sections,
            overall_score=overall_score,
            match_info=match_info,
            edu_info=edu_info,
            subsection_scores=subsection_scores,
            suggestions=suggestions
        )

    # ── Build final result ──
    result = {
        "mode": 1,
        "resume_id": resume_id,
        "resume_file": os.path.basename(resume_path),
        "jd_file": os.path.basename(jd_path),
        "candidate_name": candidate_name,
        "overall_role_fit_score": overall_score,
        "score_breakdown": overall_result,
        "matched_experience_years": match_info.get("matched_years", 0),
        "required_experience_years": match_info.get("required_years"),
        "education_match": len(edu_info.get("matches", [])) > 0,
        "lowest_relevance_sections": [
            {"section": s["heading"], "relevance_score": s["relevance_score"]}
            for s in sorted(subsection_scores, key=lambda x: x["relevance_score"])[:3]
        ],
        "llm_suggestions": suggestions
    }

    # ── Print results ──
    print("\n" + "=" * 60)
    print("MODE 1 RESULTS")
    print("=" * 60)
    print(f"Candidate:          {result['candidate_name']}")
    print(f"Overall Score:      {result['overall_role_fit_score']}/100")
    print(f"  Skills (S):       {overall_result['skills_score']}/100")
    print(f"  Evidence (E):     {overall_result['evidence_score']}/100")
    print(f"  Experience (X):   {overall_result['experience_score']}/100")
    print(f"  Keyword/ATS (K):  {overall_result['keyword_score']}/100")
    print(f"  Logistical (L):   {overall_result['logistical_score']}/100")
    print(f"Experience Match:   {result['matched_experience_years']} / {result['required_experience_years']} years")
    print(f"Education Match:    {result['education_match']}")

    print("\nLowest-relevance sections:")
    for s in result["lowest_relevance_sections"]:
        print(f"  • {s['section']}: {s['relevance_score']}/100")

    print_llm_suggestions(suggestions)

    return result


# ── Test Mode 1 ──



# In[14]:


# ================================
# CELL 14 — MODE 3: JD ONLY, FIND TOP 3 RESUMES
# ================================

def mode3_match_jd(jd_path, top_k_faiss=10, top_n_final=3):
    """
    Mode 3 — Two-stage retrieval:
    Stage 1: FAISS narrows all stored resumes down to top_k_faiss
             using fast vector similarity (Cell 11 search).
    Stage 2: Run the full Cell 8 scoring pipeline on those
             candidates and return the top_n_final by overall_score.
    """

    print("=" * 60)
    print("MODE 3 — JD ONLY, FINDING TOP MATCHES")
    print("=" * 60)
    print(f"JD: {os.path.basename(jd_path)}")
    print()

    # ── Step 1: Extract and section the JD ──
    print("Step 1: Parsing job description...")
    jd_extracted = extract_text(jd_path)
    jd_blocks = jd_extracted[0] if isinstance(jd_extracted, tuple) else jd_extracted
    jd_sections = semantic_sectioning(jd_blocks, for_jd=True)

    # ── Step 2: FAISS search — narrow to top_k_faiss candidates ──
    print(f"Step 2: FAISS search — narrowing to top {top_k_faiss} candidates...")
    faiss_results = search_resumes_by_jd(jd_sections, top_k=top_k_faiss)

    if not faiss_results:
        print("No stored resumes found.")
        return {"mode": 3, "jd_file": os.path.basename(jd_path), "top_matches": []}

    # ── Step 3: Run full scoring pipeline on each candidate ──
    print(f"\nStep 3: Running detailed scoring on {len(faiss_results)} candidates...")
    scored_candidates = []

    for i, candidate in enumerate(faiss_results, start=1):
        resume_path = candidate["s3_path"]
        print(f"  [{i}/{len(faiss_results)}] Scoring: {os.path.basename(resume_path)}")

        try:
            local_resume_path = resume_path
            if resume_path.startswith("s3://"):
                local_resume_path = download_from_s3_to_temp(resume_path)

            resume_extracted = extract_text(local_resume_path)
            resume_blocks = resume_extracted[0] if isinstance(resume_extracted, tuple) else resume_extracted

            resume_sections = semantic_sectioning(resume_blocks, for_jd=False)
            resume_sections = add_bold_subsections_to_all_sections(resume_sections)

            match_info = compute_matched_experience_years(jd_sections, resume_sections)
            overall_result = compute_overall_role_fit_score(
                resume_sections, jd_sections, match_info
            )

            # Re-extract candidate name and skills (fresh from this resume)
            candidate_name = "Unknown"
            for heading, content in resume_sections.items():
                h = heading.lower()
                if "contact" in h or "personal" in h:
                    lines = [l.strip() for l in content.splitlines() if l.strip()]
                    if lines:
                        candidate_name = lines[0]
                    break
            if candidate_name == "Unknown":
                for block in resume_blocks:
                    line = block.strip()
                    if line and len(line) < 50 and not any(
                        kw in line.lower() for kw in [
                            "resume", "cv", "curriculum", "page", "http", "@", "phone"
                        ]
                    ):
                        candidate_name = line
                        break
            candidate_name = re.sub(r'[#*_]', '', candidate_name).strip()

            SKILLS_HEADING_KEYWORDS = [
                "skill", "competenc", "expertise", "proficienc", "qualification"
            ]
            skills = []
            for heading, content in resume_sections.items():
                h = heading.lower()
                if any(kw in h for kw in SKILLS_HEADING_KEYWORDS) and " – Subsection" not in heading:
                    skills = [l.strip() for l in content.splitlines() if l.strip()][:10]
                    break

            scored_candidates.append({
                "resume_id": candidate["resume_id"],
                "candidate_name": candidate_name,
                "s3_path": resume_path,
                "faiss_similarity": candidate["similarity_score"],
                "overall_score": overall_result["overall_score"],
                "score_breakdown": overall_result,
                "matched_experience_years": match_info.get("matched_years", 0),
                "required_experience_years": match_info.get("required_years"),
                "skills": skills
            })

        except Exception as e:
            print(f"      ✗ Error scoring this resume: {e}")
            continue

    # ── Step 4: Sort by overall_score, take top_n_final ──
    print(f"\nStep 4: Selecting top {top_n_final} by overall role fit score...")
    sorted_candidates = sorted(
        scored_candidates,
        key=lambda x: x["overall_score"],
        reverse=True
    )
    top_matches = sorted_candidates[:top_n_final]

    for rank, c in enumerate(top_matches, start=1):
        c["rank"] = rank

    # ── Step 5: Call Bedrock to explain the ranking ──
    print("Step 5: Calling Bedrock to explain the top matches...")
    ranking_explanation = get_ranking_explanation(jd_sections, top_matches)

    # ── Step 6: Build final result ──
    result = {
        "mode": 3,
        "jd_file": os.path.basename(jd_path),
        "candidates_evaluated": len(scored_candidates),
        "top_matches": top_matches,
        "ranking_explanation": ranking_explanation
    }

    # ── Print results ──
    print("\n" + "=" * 60)
    print("MODE 3 RESULTS — TOP MATCHES")
    print("=" * 60)
    for c in top_matches:
        print(f"\nRank {c['rank']}: {c['candidate_name']}")
        print(f"  Overall Score:      {c['overall_score']}/100")
        print(f"  FAISS Similarity:   {c['faiss_similarity']}%")
        print(f"  Experience:         {c['matched_experience_years']} / {c['required_experience_years']} years")
        print(f"  Skills:             {', '.join(c['skills'][:5])}")
        print("-" * 60)

    print("\n================ RANKING EXPLANATION ================\n")
    print(ranking_explanation.get("explanation", "N/A"))

    return result


# ================================
# BEDROCK — EXPLAIN WHY THESE TOP 3 WERE CHOSEN
# ================================
def get_ranking_explanation(jd_sections, top_matches):
    """
    Calls Bedrock Nova Lite to explain why these candidates
    ranked highest, in plain language for a recruiter.
    """
    required_lines, preferred_lines = classify_jd_requirement_lines(jd_sections)
    jd_summary = "\n".join(f"- {l}" for l in (required_lines + preferred_lines)[:10])

    candidates_summary = ""
    for c in top_matches:
        candidates_summary += f"""
Rank {c['rank']}: {c['candidate_name']}
  Overall Score: {c['overall_score']}/100
  Experience: {c['matched_experience_years']} / {c['required_experience_years']} years
  Skills: {', '.join(c['skills'][:8])}
"""

    prompt = f"""You are a recruiter assistant. A job description was matched against
a database of resumes, and these are the top {len(top_matches)} candidates by role fit score.

JOB REQUIREMENTS:
{jd_summary}

TOP CANDIDATES:
{candidates_summary}

Write a short, plain-language explanation (3-5 sentences) for a recruiter
explaining why these candidates ranked highest and what distinguishes
the #1 candidate from the others.

Return valid JSON only, with this structure:
{{
  "explanation": "your explanation here"
}}
Return valid JSON only. No markdown. No extra text."""

    body = json.dumps({
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": 500, "temperature": 0.3}
    })

    try:
        response = bedrock.invoke_model(
            modelId="amazon.nova-lite-v1:0",
            body=body,
            contentType="application/json",
            accept="application/json"
        )
        response_body = json.loads(response["body"].read())
        raw_text = response_body["output"]["message"]["content"][0]["text"]

        clean_text = raw_text.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.split("```")[1]
            if clean_text.startswith("json"):
                clean_text = clean_text[4:]
        clean_text = clean_text.strip()

        return json.loads(clean_text)

    except Exception as e:
        return {"explanation": f"Could not generate explanation: {e}"}





# In[16]:


# ================================
# CELL 15 — FLASK REST API
# ================================
from flask import Flask, request, jsonify
import tempfile
import threading

app = Flask(__name__)

# ── Helper: save uploaded file to a temp path ──
def save_uploaded_file(file_obj):
    suffix = os.path.splitext(file_obj.filename)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    file_obj.save(tmp.name)
    return tmp.name


# ================================
# ENDPOINT 1 — POST /upload-resume  (Mode 2)
# ================================
@app.route("/upload-resume", methods=["POST"])
def upload_resume():
    if "resume" not in request.files:
        return jsonify({"error": "No 'resume' file provided"}), 400

    resume_file = request.files["resume"]
    resume_path = save_uploaded_file(resume_file)

    try:
        resume_id = mode2_store_resume(resume_path)
        if resume_id is None:
            return jsonify({"error": "Failed to parse/store resume"}), 500

        return jsonify({
            "mode": 2,
            "status": "success",
            "resume_id": resume_id,
            "message": "Resume parsed, embedded, and stored"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================================
# ENDPOINT 2 — POST /upload-both  (Mode 1)
# ================================
@app.route("/upload-both", methods=["POST"])
def upload_both():
    if "resume" not in request.files or "jd" not in request.files:
        return jsonify({"error": "Both 'resume' and 'jd' files are required"}), 400

    resume_file = request.files["resume"]
    jd_file = request.files["jd"]

    resume_path = save_uploaded_file(resume_file)
    jd_path = save_uploaded_file(jd_file)

    try:
        result = mode1_resume_and_jd(resume_path, jd_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================================
# ENDPOINT 3 — POST /match-jd  (Mode 3)
# ================================
@app.route("/match-jd", methods=["POST"])
def match_jd():
    if "jd" not in request.files:
        return jsonify({"error": "No 'jd' file provided"}), 400

    jd_file = request.files["jd"]
    jd_path = save_uploaded_file(jd_file)

    top_k_faiss = int(request.form.get("top_k_faiss", 10))
    top_n_final = int(request.form.get("top_n_final", 3))

    try:
        result = mode3_match_jd(jd_path, top_k_faiss=top_k_faiss, top_n_final=top_n_final)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================================
# HEALTH CHECK
# ================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "resume-intelligence"})


# ================================
# RUN FLASK IN A BACKGROUND THREAD
# (so it doesn't block the Jupyter kernel)
# ================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)

# In[ ]:




