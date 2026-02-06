"""
Batch pipeline for NAVCRUIT high-risk training forms:
- Iterate all PDFs in a folder
- Auto-rotate (0/90/180/270) and deskew scanned PDF pages
- OCR using Google Cloud Vision
- Extract header fields, rating answers, and free-text comments
- Use OpenAI LLM to:
    - summarize comments + decide if human review is needed
    - (optionally) clean up messy scalar answers for Q1–Q4
- Output:
    1) ONE combined JSON with per-document outputs keyed by filename
    2) ONE *compact* analysis JSON:
        - Overall stats per question (mean for Q1, YES-rate for Q2–Q4)
        - Per-course stats (includes "UNKNOWN" for missing course title)
        - Missing/unidentified counts per question
        - Human-review docs list + risk-level breakdown

Assumptions:
- Each form is 2 pages (page 1: header + answers, page 2: comments).
- If a PDF has >2 pages, uses first 2.
- If a PDF has <2 pages, records an error and continues.
"""

import os
import json
import re
from typing import Dict, Any, Optional, List
import difflib
from datetime import datetime

import fitz  # PyMuPDF
import cv2
import numpy as np
from google.cloud import vision
from openai import OpenAI
from dotenv import load_dotenv

from survey_ocr.config import get_config
# -------------------------------------------------------------------------
# USER CONFIG
# -------------------------------------------------------------------------

_CONFIG = get_config()
GCP_VISION_KEY = str(_CONFIG.gcp_vision_key) if _CONFIG.gcp_vision_key else ""

SCANNED_DOCS_DIR = str(_CONFIG.scanned_docs_dir)

OUTPUT_DIR = str(_CONFIG.output_dir)
RAW_OCR_DIR = os.path.join(OUTPUT_DIR, "raw_ocr_pages")

RESULTS_ALL_JSON = str(_CONFIG.results_all_json)
ANALYSIS_JSON = str(_CONFIG.analysis_json)

DOTENV_PATH = str(_CONFIG.dotenv_path)

MAX_HEADER_TOKENS = 200
USE_LLM_FOR_ANSWERS = True
SAVE_RAW_OCR = True

# -------------------------------------------------------------------------
# ROTATION & DESKEW UTILITIES
# -------------------------------------------------------------------------

def auto_rotate_opencv(img_bgr: np.ndarray) -> np.ndarray:
    """
    Score orientation by number of detected horizontal text lines.
    Tries 0°, 90°, 180°, 270° and picks the best.
    """
    def score(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 25, 15
        )
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detect = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        return cv2.countNonZero(detect)

    rotations = [
        img_bgr,
        cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(img_bgr, cv2.ROTATE_180),
        cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ]

    scores = [score(i) for i in rotations]
    best_index = int(np.argmax(scores))
    return rotations[best_index]


def deskew(img_bgr: np.ndarray) -> np.ndarray:
    """
    Estimate skew angle and rotate to straighten text.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )[1]

    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return img_bgr

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    (h, w) = img_bgr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    return cv2.warpAffine(
        img_bgr, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

# -------------------------------------------------------------------------
# OCR UTIL
# -------------------------------------------------------------------------

def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)

def get_pdf_page_count(pdf_path: str) -> int:
    doc = fitz.open(pdf_path)
    n = doc.page_count
    doc.close()
    return n

def pdf_page_to_image_bgr(pdf_path: str, page_index: int, dpi: int = 300) -> np.ndarray:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    doc.close()
    return img_bgr

def make_vision_client() -> vision.ImageAnnotatorClient:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_VISION_KEY
    return vision.ImageAnnotatorClient()

def ocr_full_page(client: vision.ImageAnnotatorClient, img_bgr: np.ndarray) -> str:
    _, encoded = cv2.imencode(".png", img_bgr)
    image = vision.Image(content=encoded.tobytes())
    resp = client.document_text_detection(image=image)
    if resp.error.message:
        raise RuntimeError(resp.error.message)
    return resp.full_text_annotation.text or ""

# -------------------------------------------------------------------------
# HEADER LABEL FUZZY MATCHING
# -------------------------------------------------------------------------

LABEL_CANON = {
    "course": ["course", "cours", "c0urse"],
    "unit": ["unit", "unlt"],
    "division": ["division", "divison", "divis1on", "div"],
    "instructor": ["instructor", "instructors", "instr", "instrctor"],
    "date": ["date", "datc"],
    "cin": ["cin", "c1n", "cm"],
}

def match_label(tok: str) -> Optional[str]:
    t = tok.strip("():,.;").lower()
    best_label = None
    best_score = 0.0
    for canon, variants in LABEL_CANON.items():
        for v in variants:
            score = difflib.SequenceMatcher(None, t, v).ratio()
            if score > best_score:
                best_score = score
                best_label = canon
    return best_label if best_score >= 0.7 else None

# -------------------------------------------------------------------------
# HEADER CLEANUP HELPERS
# -------------------------------------------------------------------------

def clean_division(value: str) -> str:
    if not value:
        return ""
    m = re.search(r"\b[0-9]{1,4}[A-Za-z/]*\b", value)
    if m:
        return m.group(0)
    return value.split()[0]

def clean_instructor(value: str) -> str:
    if not value:
        return ""
    value = re.split(
        r"\b(Write\s+N/A|Write\s+N\/A|Write|Using\s+a\s+1\s+to\s+5|Using\s+à\s+YES|Use\s+the\s+following|1\.)\b",
        value,
        maxsplit=1,
    )[0]
    tokens = value.strip(" ,.;").split()
    clean_tokens = []
    for t in tokens:
        t_clean = re.sub(r"[^A-Za-z\-']", "", t)
        if t_clean:
            clean_tokens.append(t_clean)
    if not clean_tokens:
        return value.strip(" ,.;")
    return " ".join(clean_tokens[:3])

def clean_course(value: str) -> str:
    if not value:
        return ""
    tok = value.split()[0]
    tok = re.sub(r"[^A-Za-z0-9\-]", "", tok)
    if len(tok) > 12:
        tok = tok[:12]
    return tok

def clean_unit(value: str) -> str:
    if not value:
        return ""
    value = re.split(
        r"\b(Division|Instructor|Date|Write|Using|Use|NAVCRUIT)\b",
        value,
        maxsplit=1,
    )[0]
    tokens = value.strip(" ,.;").split()
    return " ".join(tokens[:4]) if len(tokens) > 4 else " ".join(tokens)

def clean_date(value: str) -> str:
    if not value:
        return ""
    value = re.split(
        r"\b(CIN|Unit|Course|NAVCRUIT|For\s+high-risk)\b",
        value,
        maxsplit=1,
    )[0]
    value = value.strip(" ,.;")

    m = re.search(r"\b\d{1,2}\s+[A-Za-z]{3}\s+\d{2}\b", value)
    if m:
        return m.group(0)

    m = re.search(r"\b\d{1,2}\s+[A-Za-z]{3}\s+\d{4}\b", value)
    if m:
        return m.group(0)

    m = re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", value)
    if m:
        return m.group(0)

    tokens = value.split()
    return " ".join(tokens[:3])

# -------------------------------------------------------------------------
# PARSING – HEADER
# -------------------------------------------------------------------------

def parse_header_fields(full_text: str) -> Dict[str, Any]:
    result = {"course": "", "unit": "", "division": "", "instructor": "", "date": ""}

    s = re.sub(r"\s+", " ", full_text)
    if not s.strip():
        return result

    raw_tokens = s.split()

    label_positions = []
    for i, tok in enumerate(raw_tokens):
        if i > MAX_HEADER_TOKENS:
            break
        lab = match_label(tok)
        if lab:
            label_positions.append((i, lab))

    if not label_positions:
        return result

    label_positions.append((len(raw_tokens), None))

    label_to_key = {
        "course": "course",
        "unit": "unit",
        "division": "division",
        "instructor": "instructor",
        "date": "date",
    }

    for idx in range(len(label_positions) - 1):
        start_idx, lab = label_positions[idx]
        end_idx, _ = label_positions[idx + 1]
        if lab not in label_to_key:
            continue

        key = label_to_key[lab]
        value_tokens = raw_tokens[start_idx + 1 : end_idx]
        value = " ".join(value_tokens).strip(" ,.;")
        if value and not result[key]:
            result[key] = value

    result["course"] = clean_course(result["course"])
    result["unit"] = clean_unit(result["unit"])
    result["division"] = clean_division(result["division"])
    result["instructor"] = clean_instructor(result["instructor"])
    result["date"] = clean_date(result["date"])

    return result

# -------------------------------------------------------------------------
# PARSING – ANSWERS (scalar extraction + fuzzy anchors + optional LLM)
# -------------------------------------------------------------------------

def extract_scalar_from_line(line: str, mode: str) -> str:
    tokens = [t.strip(".,:;)") for t in line.strip().split()]
    if not tokens:
        return ""

    upper_tokens = [t.upper() for t in tokens]

    if mode == "scale":
        for t in upper_tokens:
            if t.isdigit() and t in {"1", "2", "3", "4", "5"}:
                return t
        m = re.search(r"\b([1-5])\b", " ".join(upper_tokens))
        return m.group(1) if m else ""

    if mode == "yesno":
        for t in upper_tokens:
            if t.startswith("Y") or "YES" in t or t.replace("/", "").startswith("SE"):
                return "YES"
            if t.startswith("N") or "NO" in t:
                return "NO"
        line_up = " ".join(upper_tokens)
        if "YES" in line_up or "Y ES" in line_up:
            return "YES"
        if "NO" in line_up or "N O" in line_up:
            return "NO"
        return ""

    return ""

def llm_clean_scalar_answer(
    client: OpenAI,
    model: str,
    question_id: str,
    question_text: str,
    raw_line: str,
    mode: str
) -> str:
    if not USE_LLM_FOR_ANSWERS:
        return ""

    allowed_values = "1,2,3,4,5" if mode == "scale" else "YES,NO"

    prompt = f"""
You are cleaning OCR results for a survey.

Each answer must be one of: {allowed_values}.
Do NOT invent new options.

Given:
- question_id: {question_id}
- question_text: {question_text}
- raw_ocr_line: {raw_line}

Decide the most likely intended answer. If you cannot confidently map it,
return "UNKNOWN".

Return ONLY JSON:
{{ "clean": "<one of {allowed_values} or UNKNOWN>" }}
""".strip()

    resp = client.responses.create(model=model, input=prompt)
    text = resp.output_text.strip()
    start, end = text.find("{"), text.rfind("}")
    json_str = text[start:end+1] if start != -1 and end != -1 else text

    try:
        parsed = json.loads(json_str)
        clean = str(parsed.get("clean", "")).upper()
        if mode == "scale" and clean in {"1", "2", "3", "4", "5"}:
            return clean
        if mode == "yesno" and clean in {"YES", "NO"}:
            return clean
        return ""
    except Exception:
        return ""

QUESTION_CONFIG = [
    {"id": "q1_A", "section": "q1", "sub": "A", "anchors": [
        "i felt my safety was always a primary concern of the instructor",
        "i felt my safety was always a primary concem of the instructor",
        "felt my safety was always a primary concern",
        "felt my safety was always a primary concem",
    ]},
    {"id": "q1_B", "section": "q1", "sub": "B", "anchors": [
        "i felt that the training environment was both safe and non-hazardous",
        "training environment was both safe and non-hazardous",
        "training environment was both safe and non hazardous",
    ]},

    {"id": "q2_A", "section": "q2", "sub": "A", "anchors": ["training time out procedures", "training time-out procedures"]},
    {"id": "q2_B", "section": "q2", "sub": "B", "anchors": ["emergency action plan"]},
    {"id": "q2_C", "section": "q2", "sub": "C", "anchors": ["tasks to be performed"]},
    {"id": "q2_D", "section": "q2", "sub": "D", "anchors": ["methods used to determine successful performance", "used to determine successful performance"]},

    {"id": "q3_A", "section": "q3", "sub": "A", "anchors": [
        "safety precautions were reemphasized immediately prior to job performance",
        "safety precautions were re-emphasized immediately prior to job performance",
    ]},
    {"id": "q3_B", "section": "q3", "sub": "B", "anchors": [
        "the instructor evaluated my knowledge of safety precautions prior to job performance",
        "instructor evaluated my knowledge of safety precautions prior to job performance",
    ]},
    {"id": "q3_C", "section": "q3", "sub": "C", "anchors": ["laboratory equipment was safe for use", "laboratory/equipment was safe for use"]},

    {"id": "q4_A", "section": "q4", "sub": "A", "anchors": ["encouraged me to report unsafe or unhealthy conditions", "encouraged me to report unsafe conditions"]},
    {"id": "q4_B", "section": "q4", "sub": "B", "anchors": ["encouraged me to do my best"]},
    {"id": "q4_C", "section": "q4", "sub": "C", "anchors": ["provided a learning environment that was not threatening to me", "learning environment that was not threatening to me"]},
]

def parse_answers_from_page1(full_text: str, oai_client: Optional[OpenAI], oai_model: str) -> Dict[str, Any]:
    data = {"q1": {}, "q2": {}, "q3": {}, "q4": {}}
    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
    best_scores = {cfg["id"]: 0.0 for cfg in QUESTION_CONFIG}
    THRESHOLD = 0.55

    for line in lines:
        l = line.lower()
        if not l:
            continue

        for cfg in QUESTION_CONFIG:
            qid = cfg["id"]
            for phrase in cfg["anchors"]:
                ratio = difflib.SequenceMatcher(None, l, phrase).ratio()
                if ratio > THRESHOLD and ratio > best_scores[qid]:
                    best_scores[qid] = ratio

                    mode = "scale" if cfg["section"] == "q1" else "yesno"
                    ans = extract_scalar_from_line(line, mode)

                    if not ans and USE_LLM_FOR_ANSWERS and oai_client is not None:
                        ans = llm_clean_scalar_answer(
                            client=oai_client,
                            model=oai_model,
                            question_id=qid,
                            question_text=phrase,
                            raw_line=line,
                            mode=mode,
                        )

                    data[cfg["section"]][cfg["sub"]] = ans
                    break

    return data

# -------------------------------------------------------------------------
# PARSING – COMMENTS
# -------------------------------------------------------------------------

def parse_comments_page2(full_text: str) -> str:
    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
    out = []
    for ln in lines:
        if re.fullmatch(r"[_\- ]+", ln):
            break
        out.append(ln)
    return " ".join(out)

# -------------------------------------------------------------------------
# LLM COMMENT ANALYSIS
# -------------------------------------------------------------------------

def analyze_comments_with_openai(oai_client: OpenAI, oai_model: str, comments: str) -> Dict[str, Any]:
    if not comments.strip():
        return {
            "summary": "",
            "requires_human_review": False,
            "risk_level": "none",
            "reason": "No comments provided."
        }

    prompt = f"""
You are analyzing trainee feedback from a high-risk military training form.

Summarize the feedback in 1 sentences.
Then decide if the comments require human review based on:
- reporting feeling unsafe
- harassment, bullying, discrimination
- threats or harm-related statements
- serious equipment/safety failures

Return ONLY raw JSON with:
- summary
- requires_human_review (true/false)
- risk_level ("none" | "low" | "medium" | "high")
- reason

NO code fences. NO markdown.

Comments:
{comments}
""".strip()

    resp = oai_client.responses.create(model=oai_model, input=prompt)
    text = resp.output_text.strip()

    text = re.sub(r"^```[a-zA-Z0-9]*", "", text)
    text = re.sub(r"```$", "", text).strip()

    start, end = text.find("{"), text.rfind("}")
    json_str = text[start:end+1] if start != -1 and end != -1 else text

    try:
        parsed = json.loads(json_str)
        return {
            "summary": parsed.get("summary", ""),
            "requires_human_review": bool(parsed.get("requires_human_review", False)),
            "risk_level": parsed.get("risk_level", "none"),
            "reason": parsed.get("reason", "")
        }
    except Exception:
        return {
            "summary": "",
            "requires_human_review": False,
            "risk_level": "none",
            "reason": "LLM output not parseable.",
            "raw_output": text
        }

# -------------------------------------------------------------------------
# DOCUMENT PROCESSING
# -------------------------------------------------------------------------

def process_single_pdf(
    pdf_path: str,
    vclient: vision.ImageAnnotatorClient,
    oai_client: Optional[OpenAI],
    oai_model: str
) -> Dict[str, Any]:
    doc_name = os.path.basename(pdf_path)
    page_count = get_pdf_page_count(pdf_path)
    if page_count < 2:
        raise RuntimeError(f"PDF has {page_count} page(s); expected at least 2.")

    img1 = pdf_page_to_image_bgr(pdf_path, 0)
    img2 = pdf_page_to_image_bgr(pdf_path, 1)

    img1 = deskew(auto_rotate_opencv(img1))
    img2 = deskew(auto_rotate_opencv(img2))

    txt1 = ocr_full_page(vclient, img1)
    txt2 = ocr_full_page(vclient, img2)

    if SAVE_RAW_OCR:
        ensure_output_dir(RAW_OCR_DIR)
        base = os.path.splitext(doc_name)[0]
        p1 = os.path.join(RAW_OCR_DIR, f"{base}__page1.txt")
        p2 = os.path.join(RAW_OCR_DIR, f"{base}__page2.txt")
        with open(p1, "w", encoding="utf-8") as f:
            f.write(txt1)
        with open(p2, "w", encoding="utf-8") as f:
            f.write(txt2)

    header = parse_header_fields(txt1)
    answers = parse_answers_from_page1(txt1, oai_client=oai_client, oai_model=oai_model)
    comments = parse_comments_page2(txt2)

    if oai_client is not None:
        comments_analysis = analyze_comments_with_openai(oai_client, oai_model, comments)
    else:
        comments_analysis = {
            "summary": "",
            "requires_human_review": False,
            "risk_level": "none",
            "reason": "OpenAI client not configured."
        }

    return {
        "status": "ok",
        "source_file": doc_name,
        "header": header,
        "answers": answers,
        "comments": comments,
        "comments_analysis": comments_analysis,
        "debug": {
            "page_count": page_count,
            "page1_raw_length": len(txt1),
            "page2_raw_length": len(txt2)
        }
    }

# -------------------------------------------------------------------------
# COMPACT AGGREGATION / ANALYSIS
# -------------------------------------------------------------------------

def _course_key_from_header(header: Dict[str, Any]) -> str:
    c = (header.get("course") or "").strip()
    return c if c else "UNKNOWN"

def _to_scale_int(x: Any) -> Optional[int]:
    s = str(x).strip() if x is not None else ""
    return int(s) if s in {"1", "2", "3", "4", "5"} else None

def _to_yesno(x: Any) -> Optional[str]:
    s = str(x).strip().upper() if x is not None else ""
    return s if s in {"YES", "NO"} else None

def _new_stats_block() -> Dict[str, Any]:
    """
    Compact stats per question:
    - Q1: mean of 1..5
    - Q2-Q4: yes_rate (YES=1, NO=0)
    Always include missing/unidentified count.
    """
    return {
        "q1": {
            "A": {"sum": 0, "n": 0, "missing": 0},
            "B": {"sum": 0, "n": 0, "missing": 0},
        },
        "q2": {
            "A": {"yes": 0, "no": 0, "missing": 0},
            "B": {"yes": 0, "no": 0, "missing": 0},
            "C": {"yes": 0, "no": 0, "missing": 0},
            "D": {"yes": 0, "no": 0, "missing": 0},
        },
        "q3": {
            "A": {"yes": 0, "no": 0, "missing": 0},
            "B": {"yes": 0, "no": 0, "missing": 0},
            "C": {"yes": 0, "no": 0, "missing": 0},
        },
        "q4": {
            "A": {"yes": 0, "no": 0, "missing": 0},
            "B": {"yes": 0, "no": 0, "missing": 0},
            "C": {"yes": 0, "no": 0, "missing": 0},
        },
    }

def _finalize_stats(block: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert internal sums into compact output fields:
    - Q1 -> mean, n, missing
    - Q2-Q4 -> yes_rate, yes, no, identified, missing
    """
    out = {"q1": {}, "q2": {}, "q3": {}, "q4": {}}

    # Q1 means
    for sub in ["A", "B"]:
        s = block["q1"][sub]["sum"]
        n = block["q1"][sub]["n"]
        missing = block["q1"][sub]["missing"]
        out["q1"][sub] = {
            "mean": round(s / n, 3) if n else None,
            "n": n,
            "missing": missing
        }

    # YES/NO rates
    for q in ["q2", "q3", "q4"]:
        out[q] = {}
        for sub, rec in block[q].items():
            yes = rec["yes"]
            no = rec["no"]
            missing = rec["missing"]
            identified = yes + no
            out[q][sub] = {
                "yes_rate": round(yes / identified, 3) if identified else None,
                "yes": yes,
                "no": no,
                "identified": identified,
                "missing": missing
            }

    return out

def build_compact_analysis(results_by_doc: Dict[str, Any]) -> Dict[str, Any]:
    totals = {
        "documents_total": 0,
        "documents_ok": 0,
        "documents_error": 0,
    }

    overall_stats = _new_stats_block()
    by_course_stats: Dict[str, Dict[str, Any]] = {}

    # Human review listing + risk breakdown (only flagged docs)
    flagged_docs: List[str] = []
    risk_counts = {"low": 0, "medium": 0, "high": 0, "unknown": 0}
    docs_by_risk = {"low": [], "medium": [], "high": [], "unknown": []}

    def ensure_course(course: str):
        if course not in by_course_stats:
            by_course_stats[course] = {
                "documents_ok": 0,
                "stats": _new_stats_block()
            }

    # Ensure missing course title section always exists
    ensure_course("UNKNOWN")

    def add_q1(stats_block: Dict[str, Any], sub: str, val: Any):
        v = _to_scale_int(val)
        if v is None:
            stats_block["q1"][sub]["missing"] += 1
        else:
            stats_block["q1"][sub]["n"] += 1
            stats_block["q1"][sub]["sum"] += v

    def add_yesno(stats_block: Dict[str, Any], q: str, sub: str, val: Any):
        v = _to_yesno(val)
        if v is None:
            stats_block[q][sub]["missing"] += 1
        elif v == "YES":
            stats_block[q][sub]["yes"] += 1
        else:
            stats_block[q][sub]["no"] += 1

    for doc_name, doc in results_by_doc.items():
        totals["documents_total"] += 1

        if doc.get("status") != "ok":
            totals["documents_error"] += 1
            continue

        totals["documents_ok"] += 1

        header = doc.get("header", {}) or {}
        answers = doc.get("answers", {}) or {}
        ca = doc.get("comments_analysis", {}) or {}

        course = _course_key_from_header(header)
        ensure_course(course)
        by_course_stats[course]["documents_ok"] += 1

        # human review flags
        if bool(ca.get("requires_human_review", False)):
            flagged_docs.append(doc_name)
            rl = str(ca.get("risk_level", "")).strip().lower()
            if rl not in {"low", "medium", "high"}:
                rl = "unknown"
            risk_counts[rl] += 1
            docs_by_risk[rl].append(doc_name)

        # Q1
        q1 = answers.get("q1", {}) or {}
        add_q1(overall_stats, "A", q1.get("A"))
        add_q1(overall_stats, "B", q1.get("B"))
        add_q1(by_course_stats[course]["stats"], "A", q1.get("A"))
        add_q1(by_course_stats[course]["stats"], "B", q1.get("B"))

        # Q2
        q2 = answers.get("q2", {}) or {}
        for sub in ["A", "B", "C", "D"]:
            add_yesno(overall_stats, "q2", sub, q2.get(sub))
            add_yesno(by_course_stats[course]["stats"], "q2", sub, q2.get(sub))

        # Q3
        q3 = answers.get("q3", {}) or {}
        for sub in ["A", "B", "C"]:
            add_yesno(overall_stats, "q3", sub, q3.get(sub))
            add_yesno(by_course_stats[course]["stats"], "q3", sub, q3.get(sub))

        # Q4
        q4 = answers.get("q4", {}) or {}
        for sub in ["A", "B", "C"]:
            add_yesno(overall_stats, "q4", sub, q4.get(sub))
            add_yesno(by_course_stats[course]["stats"], "q4", sub, q4.get(sub))

    # Finalize compact stats
    overall = _finalize_stats(overall_stats)

    by_course_out: Dict[str, Any] = {}
    for course, rec in by_course_stats.items():
        by_course_out[course] = {
            "documents_ok": rec["documents_ok"],
            "stats": _finalize_stats(rec["stats"])
        }

    # Keep deterministic ordering: UNKNOWN first, then alphabetical
    ordered_by_course = {}
    if "UNKNOWN" in by_course_out:
        ordered_by_course["UNKNOWN"] = by_course_out.pop("UNKNOWN")
    for k in sorted(by_course_out.keys()):
        ordered_by_course[k] = by_course_out[k]

    return {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_dir": SCANNED_DOCS_DIR,
            "use_llm_for_answers": USE_LLM_FOR_ANSWERS
        },
        "overall": {
            **totals,
            "stats": overall
        },
        "by_course": ordered_by_course,
        "human_review": {
            "flagged_total": len(flagged_docs),
            "risk_level_counts": risk_counts,
            "flagged_documents": flagged_docs,
            "flagged_documents_by_risk": docs_by_risk
        }
    }

# -------------------------------------------------------------------------
# BATCH MAIN
# -------------------------------------------------------------------------

def list_pdfs(folder: str) -> List[str]:
    out = []
    for name in os.listdir(folder):
        if name.lower().endswith(".pdf"):
            out.append(os.path.join(folder, name))
    out.sort()
    return out

def run_batch_pipeline():
    ensure_output_dir(OUTPUT_DIR)

    # Load env once
    load_dotenv(DOTENV_PATH)
    api_key = os.getenv("OPENAI_API_KEY")
    oai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Init clients once
    print("[INFO] Initializing Vision...")
    vclient = make_vision_client()

    oai_client: Optional[OpenAI] = None
    if api_key:
        oai_client = OpenAI(api_key=api_key)
        print(f"[INFO] OpenAI enabled (model={oai_model})")
    else:
        print("[WARN] OPENAI_API_KEY not set -> skipping LLM steps (comments + answer cleaning).")

    pdfs = list_pdfs(SCANNED_DOCS_DIR)
    if not pdfs:
        raise RuntimeError(f"No PDFs found in: {SCANNED_DOCS_DIR}")

    all_results: Dict[str, Any] = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_dir": SCANNED_DOCS_DIR,
            "openai_model": oai_model if api_key else None,
            "use_llm_for_answers": USE_LLM_FOR_ANSWERS,
            "save_raw_ocr": SAVE_RAW_OCR,
            "raw_ocr_dir": RAW_OCR_DIR if SAVE_RAW_OCR else None,
        },
        "documents": {},
    }

    print(f"[INFO] Found {len(pdfs)} PDF(s). Processing...")

    for i, pdf_path in enumerate(pdfs, start=1):
        doc_name = os.path.basename(pdf_path)
        print(f"[INFO] ({i}/{len(pdfs)}) Processing: {doc_name}")

        try:
            doc_result = process_single_pdf(
                pdf_path=pdf_path,
                vclient=vclient,
                oai_client=oai_client,
                oai_model=oai_model
            )
            all_results["documents"][doc_name] = doc_result
        except Exception as e:
            all_results["documents"][doc_name] = {
                "status": "error",
                "source_file": doc_name,
                "error": str(e)
            }

    # Write combined results JSON
    with open(RESULTS_ALL_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Write compact analysis JSON
    analysis = build_compact_analysis(all_results["documents"])
    with open(ANALYSIS_JSON, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    print("[INFO] DONE.")
    print(f" - Combined results: {RESULTS_ALL_JSON}")
    print(f" - Compact analysis : {ANALYSIS_JSON}")

if __name__ == "__main__":
    run_batch_pipeline()
