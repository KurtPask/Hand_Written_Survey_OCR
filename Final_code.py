r"""
DOCEXTRACT - High Risk / Critique Form OCR via OpenAI Vision JSON extraction

What this version adds (per your request):
1) “Basic analytics” charts:
   - YES/NO counts + YES rates for Q2–Q4
   - 1–5 histograms for Q1A/Q1B + mean/median
   - Risk level counts + trend-ready data
   - Risk tag counts (safety/harassment/etc.)
   - Contradiction rate (comments vs ratings)

2) Better free-response risk analysis:
   - LLM returns: summary, risk_level, requires_human_review, risk_tags, contradiction flag, contradiction_reason, confidence
   - Deterministic keyword guardrail to force minimum risk + human review
   - Risk score computed (severity * confidence) + small heuristic boosts

3) Parallel processing:
   - Batch parallel (ThreadPoolExecutor)
   - Per-document parallel (page1 + page2 extraction concurrently)

Install:
  pip install pymupdf opencv-python numpy python-dotenv openai matplotlib
"""

import os, re, json, time, base64, difflib, threading
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI

from survey_ocr.config import get_config

# =============================================================================
# CONFIG
# =============================================================================

_CONFIG = get_config()
SCANNED_DOCS_DIR = str(_CONFIG.scanned_docs_dir)
DOTENV_PATH = str(_CONFIG.dotenv_path)

OUTPUT_DIR = str(_CONFIG.output_dir)
RAW_OAI_DIR = str(_CONFIG.raw_oai_dir)

RESULTS_ALL_JSON = str(_CONFIG.results_all_json)
ANALYSIS_JSON = str(_CONFIG.analysis_json)

ANALYSIS_CHARTS_DIR = str(_CONFIG.analysis_charts_dir)

RENDER_DPI = int(os.getenv("DOCEXTRACT_RENDER_DPI", "220"))
MAX_LONG_EDGE = int(os.getenv("DOCEXTRACT_MAX_LONG_EDGE", "1800"))

MIN_SECONDS_BETWEEN_REQUESTS = 0.05
OPENAI_RETRIES = 5
DEFAULT_MODEL = "gpt-4o-mini"

DEFAULT_WORKERS = min(6, (os.cpu_count() or 4))
MAX_WORKERS = max(1, int(os.getenv("DOCEXTRACT_WORKERS", str(DEFAULT_WORKERS))))
PER_DOC_PARALLEL = os.getenv("PER_DOC_PARALLEL", "1").strip() not in {"0", "false", "False"}

DOC_LIMIT = int(os.getenv("DOC_LIMIT", "0"))

# Canonical order of the prompted questions on "standard" page 2
CANON_PAGE2_PROMPTS = [
    "Was the instructor able to clearly communicate all safety rules and instructions to you and clarify any questions asked?",
    "Is there anything that could be done differently to increase safety awareness, or improve the level of training?",
    "What skills, lessons or methods were especially effective for you?",
    "What can we improve upon?",
    "Do you feel we addressed your needs? If not, then why?"
]

# Canonical order for the WSD Student Critique (page 2) prompts
CANON_PAGE2_PROMPTS_WSD = [
    "1A. Did the classroom instruction prepare you for the Third Class Swim Test? If not, how can it be improved?",
    "1B. Were the swim videos and instructor demonstrations helpful in passing the swim test? If not, how can they be improved?",
    "2A. Were there any health or safety issues in the classroom, locker rooms or on the pool deck?",
    "2B. What recommendations do you have on how to improve the swim facilities?",
    "3A. Were the instructors professional? If not, how were they unprofessional?",
    "3B. Were the instructions presented in a clear and effective manner? If not, describe below.",
    "4A. What can Water Survival Division do better to motivate recruits to succeed in passing their Third Class Swim Test?"
]

# Deterministic “guardrail” keywords (cheap and effective)
# If any hits appear, we force at least medium risk + human review.
RISK_KEYWORDS = {
    "high": [
        "suicide", "kill", "death threat", "rape", "sexual assault", "assaulted", "weapon", "gun",
        "i will hurt", "threaten", "threatened", "molest", "stab"
    ],
    "medium": [
        "unsafe", "danger", "injury", "injured", "hurt", "slipped", "fall", "bleed", "bleeding",
        "harass", "harassment", "bully", "bullying", "racist", "sexist", "retaliation", "fear",
        "unprofessional", "yelled", "screamed", "threat", "assault"
    ],
    "low": [
        "confusing", "unclear", "rushed", "disorganized", "bad attitude", "poor instruction",
        "dirty", "broken", "crowded"
    ],
}

SEVERITY_WEIGHT = {"none": 0.0, "low": 1.0, "medium": 3.0, "high": 5.0}


# =============================================================================
# THROTTLE (per-thread)
# =============================================================================

_last_req = threading.local()
def throttle():
    now = time.time()
    last = getattr(_last_req, "t", 0.0)
    dt = now - last
    if dt < MIN_SECONDS_BETWEEN_REQUESTS:
        time.sleep(MIN_SECONDS_BETWEEN_REQUESTS - dt)
    _last_req.t = time.time()


# =============================================================================
# FILE UTILS
# =============================================================================

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_pdfs(folder: str) -> List[str]:
    out = []
    for n in os.listdir(folder):
        if n.lower().endswith(".pdf"):
            out.append(os.path.join(folder, n))
    out.sort()
    return out

def get_pdf_page_count(pdf_path: str) -> int:
    doc = fitz.open(pdf_path)
    n = doc.page_count
    doc.close()
    return n


# =============================================================================
# IMAGE + PREPROCESS
# =============================================================================

def downscale_long_edge(img: np.ndarray, max_long_edge: int) -> np.ndarray:
    h, w = img.shape[:2]
    le = max(h, w)
    if le <= max_long_edge:
        return img
    s = max_long_edge / float(le)
    return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

def pdf_pages_to_bgr(pdf_path: str, page_indices: List[int], dpi: int) -> List[np.ndarray]:
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(dpi/72, dpi/72)
    pages = []
    for idx in page_indices:
        page = doc.load_page(idx)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        pages.append(downscale_long_edge(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), MAX_LONG_EDGE))
    doc.close()
    return pages

def img_to_data_url_png(img_bgr: np.ndarray) -> str:
    ok, enc = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("PNG encode failed.")
    b64 = base64.b64encode(enc.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def enhance_handwriting(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def enhance_handwriting_binary(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11,
    )
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

def crop_rel(img_bgr: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    X0, Y0 = int(x0*w), int(y0*h)
    X1, Y1 = int(x1*w), int(y1*h)
    X0, Y0 = max(0,X0), max(0,Y0)
    X1, Y1 = min(w,X1), min(h,Y1)
    if X1 <= X0+10 or Y1 <= Y0+10:
        return img_bgr
    return img_bgr[Y0:Y1, X0:X1].copy()


# =============================================================================
# NORMALIZERS
# =============================================================================

def normalize_yesno(v: str) -> str:
    v = (v or "").strip().upper()
    if v in {"YES", "Y"}:
        return "YES"
    if v in {"NO", "N"}:
        return "NO"
    return "UNKNOWN"

def normalize_scale(v: str) -> str:
    v = (v or "").strip()
    return v if v in {"1","2","3","4","5"} else "UNKNOWN"


# =============================================================================
# OPENAI CALL (vision -> JSON), with rate limit retry
# =============================================================================

def openai_json(
    client: OpenAI,
    model: str,
    prompt: str,
    images: List[Tuple[str, np.ndarray]],
    max_output_tokens: int,
    raw_save_path: Optional[str] = None
) -> Dict[str, Any]:
    content = [{"type":"input_text","text":prompt}]
    for label, img in images:
        content.append({"type":"input_text","text":f"IMAGE_LABEL: {label}"})
        content.append({"type":"input_image","image_url": img_to_data_url_png(img)})

    last_err = None
    for attempt in range(OPENAI_RETRIES+1):
        try:
            throttle()
            resp = client.responses.create(
                model=model,
                input=[{"role":"user","content":content}],
                max_output_tokens=max_output_tokens
            )
            text = (resp.output_text or "").strip()
            if raw_save_path:
                with open(raw_save_path, "w", encoding="utf-8") as f:
                    f.write(text)

            start, end = text.find("{"), text.rfind("}")
            js = text[start:end+1] if start!=-1 and end!=-1 else text
            return json.loads(js)

        except Exception as e:
            last_err = str(e)
            if "429" in last_err or "Rate limit" in last_err:
                time.sleep(0.7 + 0.3*attempt)
                continue
            time.sleep(0.3 + 0.2*attempt)

    return {"error": last_err or "unknown error"}


# =============================================================================
# PAGE 1: minimal extraction
# =============================================================================

def prompt_page1_minimal() -> str:
    return """
Extract ONLY the following from page 1:

1) header:
{ "course":"","unit":"","division":"","instructor":"","date":"","cin":"" }

2) answers for these fixed items:
1A, 1B are scale 1-5
2A-2D are YES/NO
3A-3C are YES/NO
4A-4C are YES/NO

Return ONLY JSON:
{
  "header": {"course":"","unit":"","division":"","instructor":"","date":"","cin":""},
  "answers": {
    "q1":{"A":"","B":""},
    "q2":{"A":"","B":"","C":"","D":""},
    "q3":{"A":"","B":"","C":""},
    "q4":{"A":"","B":"","C":""}
  }
}

Rules:
- Do not invent answers. If unclear, return "".
- Allow Y/N as alternatives for YES/NO.
""".strip()

def extract_page1(client: OpenAI, model: str, page1_bgr: np.ndarray, raw_path: str) -> Dict[str, Any]:
    return openai_json(
        client, model,
        prompt_page1_minimal(),
        images=[("page1_full", page1_bgr)],
        max_output_tokens=900,
        raw_save_path=raw_path
    )

def _count_nonempty_page1_fields(out: Dict[str, Any]) -> int:
    if not isinstance(out, dict):
        return -1
    c = 0
    for v in ((out.get("header", {}) or {}).values()):
        if str(v or "").strip():
            c += 1
    answers = out.get("answers", {}) or {}
    for q in ["q1", "q2", "q3", "q4"]:
        for v in ((answers.get(q, {}) or {}).values()):
            if str(v or "").strip():
                c += 1
    return c

def extract_page1_ensemble(client: OpenAI, model: str, page1_bgr: np.ndarray, raw_path: str) -> Dict[str, Any]:
    base = extract_page1(client, model, page1_bgr, raw_path)
    hi_contrast = openai_json(
        client,
        model,
        prompt_page1_minimal(),
        images=[("page1_full_binary", enhance_handwriting_binary(page1_bgr))],
        max_output_tokens=900,
        raw_save_path=raw_path.replace(".txt", "__binary.txt"),
    )

    if "error" in hi_contrast and "error" not in base:
        return base
    if "error" in base and "error" not in hi_contrast:
        return hi_contrast
    return hi_contrast if _count_nonempty_page1_fields(hi_contrast) > _count_nonempty_page1_fields(base) else base


# =============================================================================
# PAGE 2: three-layout flexible extraction
# =============================================================================

def prompt_page2_flexible_three_layouts(
    canon_standard: List[str],
    canon_wsd: List[str],
) -> str:
    standard_text = "\n".join([f"- {p}" for p in canon_standard])
    wsd_text = "\n".join([f"- {p}" for p in canon_wsd])

    return f"""
You will get crops covering PAGE 2 of a training critique form. There are THREE possible layouts:

(1) "prompted_standard":
- Printed free-response questions (listed in STANDARD_PROMPTS below)
- May also include OPS/LCPO/DIVO staff comment boxes.

(2) "simple":
- A mostly blank page with ONE big comment area (no printed free-response questions).

(3) "wsd":
- The Water Survival Division (WSD) Student Critique back page.
- It has section headers (TRAINING / FACILITIES / INSTRUCTORS / RECRUIT FEEDBACK) and sub-questions labeled like 1A, 1B, 2A, 2B, 3A, 3B, 4A.

Task:
Extract ONLY HANDWRITTEN content and map it to the correct fields.

Return ONLY JSON in exactly this shape:
{{
  "layout": "prompted_standard" | "simple" | "wsd" | "unknown",
  "trainee_comments": "",
  "staff_comments": {{"ops":"","lcpo":"","divo":""}},
  "prompted_qas": [{{"prompt":"","answer":""}}]
}}

Rules:
- If layout is "prompted_standard":
  - Always include prompted_qas entries for STANDARD_PROMPTS in that exact order (answer="" if blank).

- If layout is "wsd":
  - Always include prompted_qas entries for WSD_PROMPTS in that exact order (answer="" if blank).
  - Do NOT include STANDARD_PROMPTS.

- If layout is "simple":
  - Put all handwriting into trainee_comments.
  - Set prompted_qas to the full STANDARD_PROMPTS list with answer="" (so downstream stays stable).
  - staff_comments should be empty strings.

- Only handwriting goes into answers; do NOT copy printed prompts or labels as answers.
- Copy handwritten OPS/LCPO/DIVO notes into staff_comments IF those boxes exist; otherwise leave empty strings.
- Use empty strings for anything unwritten.

STANDARD_PROMPTS (prompted_standard layout):
{standard_text}

WSD_PROMPTS (wsd layout):
{wsd_text}
""".strip()

def extract_page2_handwriting(client: OpenAI, model: str, page2_bgr: np.ndarray, raw_path: str) -> Dict[str, Any]:
    p2_clahe = enhance_handwriting(page2_bgr)
    p2_binary = enhance_handwriting_binary(page2_bgr)

    def _crops(img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        return [
            ("p2_full", crop_rel(img, 0.01, 0.01, 0.99, 0.99)),
            ("p2_top", crop_rel(img, 0.02, 0.01, 0.98, 0.36)),
            ("p2_mid_top", crop_rel(img, 0.02, 0.30, 0.98, 0.58)),
            ("p2_mid_bottom", crop_rel(img, 0.02, 0.52, 0.98, 0.80)),
            ("p2_bottom", crop_rel(img, 0.02, 0.74, 0.98, 0.99)),
        ]

    base = openai_json(
        client, model,
        prompt_page2_flexible_three_layouts(CANON_PAGE2_PROMPTS, CANON_PAGE2_PROMPTS_WSD),
        images=_crops(p2_clahe),
        max_output_tokens=1100,
        raw_save_path=raw_path
    )

    second = openai_json(
        client,
        model,
        prompt_page2_flexible_three_layouts(CANON_PAGE2_PROMPTS, CANON_PAGE2_PROMPTS_WSD),
        images=_crops(p2_binary),
        max_output_tokens=1100,
        raw_save_path=raw_path.replace(".txt", "__binary.txt"),
    )

    def _response_score(out: Dict[str, Any]) -> int:
        if not isinstance(out, dict) or "error" in out:
            return -1
        score = len(str(out.get("trainee_comments", "") or "").strip())
        for v in ((out.get("staff_comments", {}) or {}).values()):
            score += len(str(v or "").strip())
        for qa in (out.get("prompted_qas", []) or []):
            score += len(str((qa or {}).get("answer", "") or "").strip())
        return score

    return second if _response_score(second) > _response_score(base) else base


# =============================================================================
# ORDERING + STRUCTURE
# =============================================================================

def _norm_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s\?\.]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def reorder_prompted_qas(prompted_qas: Any, canon_prompts: List[str]) -> List[Dict[str, str]]:
    if not isinstance(prompted_qas, list):
        return []
    canon_norm = [_norm_text(c) for c in canon_prompts]
    used = set()
    ordered: List[Dict[str, str]] = []

    for c in canon_norm:
        best_i = None
        best_score = 0.0
        for i, qa in enumerate(prompted_qas):
            if i in used:
                continue
            p = _norm_text(str((qa or {}).get("prompt", "") or ""))
            if not p:
                continue
            score = difflib.SequenceMatcher(None, p, c).ratio()
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is not None and best_score >= 0.55:
            used.add(best_i)
            qa = prompted_qas[best_i] or {}
            ordered.append({
                "prompt": str(qa.get("prompt","") or "").strip(),
                "answer": str(qa.get("answer","") or "").strip()
            })

    for i, qa in enumerate(prompted_qas):
        if i in used:
            continue
        qa = qa or {}
        ordered.append({
            "prompt": str(qa.get("prompt","") or "").strip(),
            "answer": str(qa.get("answer","") or "").strip()
        })

    return [qa for qa in ordered if qa.get("prompt") or qa.get("answer")]

def build_prompted_qas_map(prompted_qas: List[Dict[str, str]], canon_prompts: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {q: "" for q in canon_prompts}
    if not prompted_qas:
        return out

    ordered = reorder_prompted_qas(prompted_qas, canon_prompts)
    canon_norm = [_norm_text(c) for c in canon_prompts]

    for qa in ordered:
        p = _norm_text(qa.get("prompt", ""))
        a = str(qa.get("answer", "") or "").strip()
        if not (p or a):
            continue

        best_i = None
        best_score = 0.0
        for idx, cn in enumerate(canon_norm):
            if out[canon_prompts[idx]]:
                continue
            score = difflib.SequenceMatcher(None, p, cn).ratio()
            if score > best_score:
                best_score = score
                best_i = idx

        if best_i is not None and best_score >= 0.45:
            out[canon_prompts[best_i]] = a

    return out


# =============================================================================
# RISK ANALYSIS (LLM + guardrails + contradiction)
# =============================================================================

def _flatten_answers_for_triage(answers: Dict[str, Any]) -> str:
    """
    Create a compact text summary of structured ratings so the model can check contradictions.
    """
    q1 = answers.get("q1", {}) or {}
    q2 = answers.get("q2", {}) or {}
    q3 = answers.get("q3", {}) or {}
    q4 = answers.get("q4", {}) or {}

    def yn_block(qname: str, block: Dict[str, str]) -> str:
        parts = []
        for k in sorted(block.keys()):
            parts.append(f"{qname}{k}={block.get(k,'UNKNOWN')}")
        return " ".join(parts)

    return (
        f"Q1A={q1.get('A','UNKNOWN')} Q1B={q1.get('B','UNKNOWN')}\n"
        f"{yn_block('Q2', q2)}\n"
        f"{yn_block('Q3', q3)}\n"
        f"{yn_block('Q4', q4)}"
    )

def _keyword_guardrail(comments: str) -> Dict[str, Any]:
    """
    Deterministic keyword scan. Returns:
      - min_risk_level (none/low/medium/high)
      - force_human_review (bool)
      - hits (list of strings)
    """
    text = (comments or "").lower()
    hits = []
    level = "none"

    def any_hit(words: List[str]) -> List[str]:
        found = []
        for w in words:
            if w in text:
                found.append(w)
        return found

    high_hits = any_hit(RISK_KEYWORDS["high"])
    med_hits = any_hit(RISK_KEYWORDS["medium"])
    low_hits = any_hit(RISK_KEYWORDS["low"])

    if high_hits:
        level = "high"
        hits.extend(high_hits)
    elif med_hits:
        level = "medium"
        hits.extend(med_hits)
    elif low_hits:
        level = "low"
        hits.extend(low_hits)

    force = level in {"medium", "high"}
    return {"min_risk_level": level, "force_human_review": force, "hits": hits}

def prompt_comment_triage_v2(comments: str, ratings_summary: str) -> str:
    return f"""
You are triaging trainee feedback for potential human review.

You are given:
1) Structured ratings (scale + yes/no)
2) Free-response handwriting text extracted from the form

Return ONLY JSON in exactly this shape:
{{
  "summary": "",
  "risk_level": "none" | "low" | "medium" | "high",
  "requires_human_review": false,
  "confidence": 0.0,
  "risk_tags": {{
    "safety": false,
    "harassment_bullying": false,
    "threats_self_harm": false,
    "medical_injury": false,
    "facility_issue": false,
    "instruction_quality": false,
    "professionalism": false,
    "retaliation_fear": false
  }},
  "contradiction_with_ratings": false,
  "contradiction_reason": ""
}}

Rules:
- confidence is 0.0 to 1.0 based on how clear and specific the comments are.
- requires_human_review MUST be true if:
  - threats/self-harm/violence/sexual assault content
  - explicit harassment/bullying
  - serious safety failures or injuries
  - credible retaliation fear
- risk_level guidance:
  - high: threats/violence/self-harm/sexual assault OR serious injury/major safety failure
  - medium: unsafe conditions, harassment/bullying, injury concerns, retaliation fear, repeated severe unprofessional conduct
  - low: general dissatisfaction, mild unprofessionalism, unclear instruction, facility complaints without danger
  - none: neutral/positive or empty
- contradiction_with_ratings should be true if ratings look very positive (e.g. Q1A/Q1B are 4-5 and most YES)
  but comments describe serious problems (unsafe, harassment, injury, etc.), or the reverse.
- Do NOT include extra keys.
- If comments are empty, risk_level=none, requires_human_review=false, confidence=0.0.

Structured ratings:
{ratings_summary}

Comments:
{comments}
""".strip()

def analyze_comments_v2(client: OpenAI, model: str, comments: str, answers: Dict[str, Any]) -> Dict[str, Any]:
    comments = (comments or "").strip()
    if not comments:
        base = {
            "summary": "",
            "risk_level": "none",
            "requires_human_review": False,
            "confidence": 0.0,
            "risk_tags": {
                "safety": False,
                "harassment_bullying": False,
                "threats_self_harm": False,
                "medical_injury": False,
                "facility_issue": False,
                "instruction_quality": False,
                "professionalism": False,
                "retaliation_fear": False
            },
            "contradiction_with_ratings": False,
            "contradiction_reason": "",
            "keyword_guardrail": {"min_risk_level": "none", "force_human_review": False, "hits": []},
            "risk_score": 0.0,
            "risk_score_reason": "No comments extracted."
        }
        return base

    ratings_summary = _flatten_answers_for_triage(answers)

    out = openai_json(
        client=client,
        model=model,
        prompt=prompt_comment_triage_v2(comments, ratings_summary),
        images=[],
        max_output_tokens=450
    )

    if "error" in out:
        # Fail closed-ish: keep it reviewable but don’t panic
        out = {
            "summary": "",
            "risk_level": "none",
            "requires_human_review": False,
            "confidence": 0.0,
            "risk_tags": {
                "safety": False,
                "harassment_bullying": False,
                "threats_self_harm": False,
                "medical_injury": False,
                "facility_issue": False,
                "instruction_quality": False,
                "professionalism": False,
                "retaliation_fear": False
            },
            "contradiction_with_ratings": False,
            "contradiction_reason": "",
            "raw_error": out.get("error", "triage failed")
        }

    # Normalize risk_level
    rl = str(out.get("risk_level", "none") or "none").lower().strip()
    if rl not in {"none", "low", "medium", "high"}:
        rl = "none"

    # Normalize confidence
    conf = out.get("confidence", 0.0)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    # Normalize tags
    tags_in = out.get("risk_tags", {}) or {}
    tags = {
        "safety": bool(tags_in.get("safety", False)),
        "harassment_bullying": bool(tags_in.get("harassment_bullying", False)),
        "threats_self_harm": bool(tags_in.get("threats_self_harm", False)),
        "medical_injury": bool(tags_in.get("medical_injury", False)),
        "facility_issue": bool(tags_in.get("facility_issue", False)),
        "instruction_quality": bool(tags_in.get("instruction_quality", False)),
        "professionalism": bool(tags_in.get("professionalism", False)),
        "retaliation_fear": bool(tags_in.get("retaliation_fear", False)),
    }

    requires = bool(out.get("requires_human_review", False))
    contradiction = bool(out.get("contradiction_with_ratings", False))
    contradiction_reason = str(out.get("contradiction_reason", "") or "").strip()
    summary = str(out.get("summary", "") or "").strip()

    # Deterministic guardrail
    guard = _keyword_guardrail(comments)
    min_rl = guard["min_risk_level"]
    if SEVERITY_WEIGHT[min_rl] > SEVERITY_WEIGHT[rl]:
        rl = min_rl
    if guard["force_human_review"]:
        requires = True

    # Compute risk_score = severity_weight * confidence + small boosts
    base_score = SEVERITY_WEIGHT[rl] * conf

    boosts = 0.0
    reasons = []
    if contradiction:
        boosts += 0.5
        reasons.append("contradiction_with_ratings")
    if any(tags.values()):
        boosts += 0.3
        reasons.append("risk_tags_present")
    if len(comments) >= 200:
        boosts += 0.2
        reasons.append("long_comment")
    if guard["hits"]:
        boosts += 0.4
        reasons.append("keyword_hits")

    risk_score = round(base_score + boosts, 3)

    return {
        "summary": summary,
        "risk_level": rl,
        "requires_human_review": requires,
        "confidence": round(conf, 3),
        "risk_tags": tags,
        "contradiction_with_ratings": contradiction,
        "contradiction_reason": contradiction_reason,
        "keyword_guardrail": guard,
        "risk_score": risk_score,
        "risk_score_reason": ", ".join(reasons) if reasons else "severity*confidence"
    }


# =============================================================================
# COMPACT Q1–Q4 analysis (now also aggregates risk tags + contradiction)
# =============================================================================

def _course_key_from_header(header: Dict[str, Any]) -> str:
    c = (header.get("course") or "").strip()
    return c if c else "UNKNOWN"

def _to_scale_int(x: Any) -> Optional[int]:
    s = str(x).strip() if x is not None else ""
    return int(s) if s in {"1","2","3","4","5"} else None

def _to_yesno(x: Any) -> Optional[str]:
    s = str(x).strip().upper() if x is not None else ""
    if s in {"Y"}: s = "YES"
    if s in {"N"}: s = "NO"
    return s if s in {"YES","NO"} else None

def _new_stats_block() -> Dict[str, Any]:
    return {
        "q1": {"A":{"sum":0,"n":0,"missing":0}, "B":{"sum":0,"n":0,"missing":0}},
        "q2": {k:{"yes":0,"no":0,"missing":0} for k in ["A","B","C","D"]},
        "q3": {k:{"yes":0,"no":0,"missing":0} for k in ["A","B","C"]},
        "q4": {k:{"yes":0,"no":0,"missing":0} for k in ["A","B","C"]},
    }

def _finalize_stats(block: Dict[str, Any]) -> Dict[str, Any]:
    out = {"q1":{}, "q2":{}, "q3":{}, "q4":{}}
    for sub in ["A","B"]:
        s = block["q1"][sub]["sum"]
        n = block["q1"][sub]["n"]
        m = block["q1"][sub]["missing"]
        out["q1"][sub] = {"mean": round(s/n,3) if n else None, "n": n, "missing": m}
    for q in ["q2","q3","q4"]:
        for sub, rec in block[q].items():
            yes, no, miss = rec["yes"], rec["no"], rec["missing"]
            identified = yes + no
            out[q][sub] = {
                "yes_rate": round(yes/identified,3) if identified else None,
                "yes": yes, "no": no, "identified": identified, "missing": miss
            }
    return out

def build_compact_analysis(results_by_doc: Dict[str, Any]) -> Dict[str, Any]:
    totals = {"documents_total":0, "documents_ok":0, "documents_error":0}
    overall_stats = _new_stats_block()
    by_course_stats: Dict[str, Dict[str, Any]] = {}

    flagged_docs: List[str] = []
    risk_counts = {"none":0,"low":0,"medium":0,"high":0,"unknown":0}
    docs_by_risk = {"none":[], "low":[], "medium":[], "high":[], "unknown":[]}

    tag_counts = {
        "safety": 0,
        "harassment_bullying": 0,
        "threats_self_harm": 0,
        "medical_injury": 0,
        "facility_issue": 0,
        "instruction_quality": 0,
        "professionalism": 0,
        "retaliation_fear": 0,
    }
    contradiction_total = 0
    contradiction_docs: List[str] = []

    def ensure_course(course: str):
        if course not in by_course_stats:
            by_course_stats[course] = {"documents_ok":0, "stats": _new_stats_block()}
    ensure_course("UNKNOWN")

    def add_q1(stats_block: Dict[str, Any], sub: str, val: Any):
        v = _to_scale_int(val)
        if v is None: stats_block["q1"][sub]["missing"] += 1
        else:
            stats_block["q1"][sub]["n"] += 1
            stats_block["q1"][sub]["sum"] += v

    def add_yesno(stats_block: Dict[str, Any], q: str, sub: str, val: Any):
        v = _to_yesno(val)
        if v is None: stats_block[q][sub]["missing"] += 1
        elif v == "YES": stats_block[q][sub]["yes"] += 1
        else: stats_block[q][sub]["no"] += 1

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

        # Risk aggregation
        rl = str(ca.get("risk_level", "unknown") or "unknown").lower().strip()
        if rl not in {"none","low","medium","high"}:
            rl = "unknown"
        risk_counts[rl] += 1
        docs_by_risk[rl].append(doc_name)

        if bool(ca.get("requires_human_review", False)):
            flagged_docs.append(doc_name)

        tags = ca.get("risk_tags", {}) or {}
        for k in tag_counts.keys():
            if bool(tags.get(k, False)):
                tag_counts[k] += 1

        if bool(ca.get("contradiction_with_ratings", False)):
            contradiction_total += 1
            contradiction_docs.append(doc_name)

        # Q1/Q2/Q3/Q4 aggregation
        q1 = answers.get("q1", {}) or {}
        add_q1(overall_stats,"A", q1.get("A"))
        add_q1(overall_stats,"B", q1.get("B"))
        add_q1(by_course_stats[course]["stats"],"A", q1.get("A"))
        add_q1(by_course_stats[course]["stats"],"B", q1.get("B"))

        q2 = answers.get("q2", {}) or {}
        for sub in ["A","B","C","D"]:
            add_yesno(overall_stats,"q2", sub, q2.get(sub))
            add_yesno(by_course_stats[course]["stats"],"q2", sub, q2.get(sub))

        q3 = answers.get("q3", {}) or {}
        for sub in ["A","B","C"]:
            add_yesno(overall_stats,"q3", sub, q3.get(sub))
            add_yesno(by_course_stats[course]["stats"],"q3", sub, q3.get(sub))

        q4 = answers.get("q4", {}) or {}
        for sub in ["A","B","C"]:
            add_yesno(overall_stats,"q4", sub, q4.get(sub))
            add_yesno(by_course_stats[course]["stats"],"q4", sub, q4.get(sub))

    overall = _finalize_stats(overall_stats)

    by_course_out = {}
    for course, rec in by_course_stats.items():
        by_course_out[course] = {"documents_ok": rec["documents_ok"], "stats": _finalize_stats(rec["stats"])}

    ordered = {}
    if "UNKNOWN" in by_course_out:
        ordered["UNKNOWN"] = by_course_out.pop("UNKNOWN")
    for k in sorted(by_course_out.keys()):
        ordered[k] = by_course_out[k]

    return {
        "meta": {"generated_at": datetime.now().isoformat(timespec="seconds"), "source_dir": SCANNED_DOCS_DIR},
        "overall": {**totals, "stats": overall},
        "by_course": ordered,
        "human_review": {
            "flagged_total": len(flagged_docs),
            "flagged_documents": flagged_docs
        },
        "risk": {
            "risk_level_counts": risk_counts,
            "documents_by_risk_level": docs_by_risk,
            "risk_tag_counts": tag_counts,
            "contradiction_total": contradiction_total,
            "contradiction_documents": contradiction_docs
        }
    }


# =============================================================================
# CHARTS (basic + risk tag charts)
# =============================================================================

def _safe_close_fig(fig):
    try:
        plt.close(fig)
    except Exception:
        pass

def _collect_numeric_for_charts(results_docs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Collect per-doc numeric values for charting:
      - q1A/q1B rating values (ints)
      - yes/no counts by subquestion
      - risk level counts
      - risk tag counts
      - contradiction counts
    """
    q1A = []
    q1B = []
    yn_counts = {
        "Q2A": {"YES":0,"NO":0,"UNKNOWN":0},
        "Q2B": {"YES":0,"NO":0,"UNKNOWN":0},
        "Q2C": {"YES":0,"NO":0,"UNKNOWN":0},
        "Q2D": {"YES":0,"NO":0,"UNKNOWN":0},
        "Q3A": {"YES":0,"NO":0,"UNKNOWN":0},
        "Q3B": {"YES":0,"NO":0,"UNKNOWN":0},
        "Q3C": {"YES":0,"NO":0,"UNKNOWN":0},
        "Q4A": {"YES":0,"NO":0,"UNKNOWN":0},
        "Q4B": {"YES":0,"NO":0,"UNKNOWN":0},
        "Q4C": {"YES":0,"NO":0,"UNKNOWN":0},
    }
    risk_levels = {"none":0,"low":0,"medium":0,"high":0,"unknown":0}
    tag_counts = {
        "safety": 0,
        "harassment_bullying": 0,
        "threats_self_harm": 0,
        "medical_injury": 0,
        "facility_issue": 0,
        "instruction_quality": 0,
        "professionalism": 0,
        "retaliation_fear": 0,
    }
    contradiction = 0
    comment_present = {"yes":0,"no":0}

    for name, doc in results_docs.items():
        if (doc or {}).get("status") != "ok":
            continue

        answers = (doc or {}).get("answers", {}) or {}
        ca = (doc or {}).get("comments_analysis", {}) or {}

        # Q1
        a = (answers.get("q1", {}) or {}).get("A", "UNKNOWN")
        b = (answers.get("q1", {}) or {}).get("B", "UNKNOWN")
        if str(a) in {"1","2","3","4","5"}: q1A.append(int(a))
        if str(b) in {"1","2","3","4","5"}: q1B.append(int(b))

        # yes/no
        for q, subs in [("q2", ["A","B","C","D"]), ("q3", ["A","B","C"]), ("q4", ["A","B","C"])]:
            block = answers.get(q, {}) or {}
            for s in subs:
                key = f"{q.upper()}{s}"
                v = str(block.get(s, "UNKNOWN") or "UNKNOWN").upper().strip()
                if v not in {"YES","NO"}: v = "UNKNOWN"
                yn_counts[key][v] += 1

        # comments present
        comments = str((doc or {}).get("comments", "") or "").strip()
        if comments:
            comment_present["yes"] += 1
        else:
            comment_present["no"] += 1

        # risk levels
        rl = str(ca.get("risk_level", "unknown") or "unknown").lower().strip()
        if rl not in risk_levels:
            rl = "unknown"
        risk_levels[rl] += 1

        # tags
        tags = ca.get("risk_tags", {}) or {}
        for k in tag_counts:
            if bool(tags.get(k, False)):
                tag_counts[k] += 1

        # contradiction
        if bool(ca.get("contradiction_with_ratings", False)):
            contradiction += 1

    return {
        "q1A": q1A,
        "q1B": q1B,
        "yn_counts": yn_counts,
        "risk_levels": risk_levels,
        "tag_counts": tag_counts,
        "contradiction_total": contradiction,
        "comment_present": comment_present
    }

def save_analysis_charts(results: Dict[str, Any], analysis: Dict[str, Any], out_dir: str):
    ensure_dir(out_dir)

    docs = (results or {}).get("documents", {}) or {}
    c = _collect_numeric_for_charts(docs)

    # 1) YES/NO counts per subquestion (stacked)
    labels = list(c["yn_counts"].keys())
    yes_vals = [c["yn_counts"][k]["YES"] for k in labels]
    no_vals  = [c["yn_counts"][k]["NO"] for k in labels]
    unk_vals = [c["yn_counts"][k]["UNKNOWN"] for k in labels]

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    x = np.arange(len(labels))
    ax.bar(x, yes_vals, label="YES")
    ax.bar(x, no_vals, bottom=yes_vals, label="NO")
    ax.bar(x, unk_vals, bottom=(np.array(yes_vals)+np.array(no_vals)), label="UNKNOWN")
    ax.set_title("YES/NO/UNKNOWN Counts by Subquestion (Q2–Q4)")
    ax.set_xlabel("Subquestion")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "yesno_counts_stacked.png"), dpi=160)
    _safe_close_fig(fig)

    # 2) YES rates bar chart
    yes_rates = []
    for k in labels:
        y = c["yn_counts"][k]["YES"]
        n = c["yn_counts"][k]["NO"]
        denom = y + n
        yes_rates.append((y/denom) if denom else 0.0)

    fig = plt.figure(figsize=(12, 4.5))
    ax = fig.add_subplot(111)
    ax.bar(x, yes_rates)
    ax.set_title("YES Rates by Subquestion (Q2–Q4) (UNKNOWN excluded)")
    ax.set_xlabel("Subquestion")
    ax.set_ylabel("YES Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "yesno_yes_rates.png"), dpi=160)
    _safe_close_fig(fig)

    # 3) Q1 histograms (A and B)
    def hist_scale(vals: List[int], fname: str, title: str):
        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.add_subplot(111)
        bins = [0.5,1.5,2.5,3.5,4.5,5.5]
        ax.hist(vals, bins=bins)
        ax.set_xticks([1,2,3,4,5])
        mean = (sum(vals)/len(vals)) if vals else 0.0
        median = float(np.median(vals)) if vals else 0.0
        ax.axvline(mean, linestyle="--", label=f"mean={mean:.2f}")
        ax.axvline(median, linestyle=":", label=f"median={median:.2f}")
        ax.set_title(title)
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=160)
        _safe_close_fig(fig)

    hist_scale(c["q1A"], "q1a_hist.png", "Q1A Distribution (1–5)")
    hist_scale(c["q1B"], "q1b_hist.png", "Q1B Distribution (1–5)")

    # 4) Comment present vs absent
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(["comments_present", "no_comments"], [c["comment_present"]["yes"], c["comment_present"]["no"]])
    ax.set_title("Documents With vs Without Any Free-Response Text")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comments_present.png"), dpi=160)
    _safe_close_fig(fig)

    # 5) Risk level counts
    rl = c["risk_levels"]
    rl_labels = ["none","low","medium","high","unknown"]
    rl_vals = [rl.get(k,0) for k in rl_labels]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(rl_labels, rl_vals)
    ax.set_title("Risk Level Counts (from Free-Response Triage)")
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "risk_level_counts.png"), dpi=160)
    _safe_close_fig(fig)

    # 6) Risk tag counts
    tags = c["tag_counts"]
    tag_labels = list(tags.keys())
    tag_vals = [tags[k] for k in tag_labels]
    fig = plt.figure(figsize=(10, 4.5))
    ax = fig.add_subplot(111)
    ax.bar(range(len(tag_labels)), tag_vals)
    ax.set_title("Risk Tag Counts (multi-label)")
    ax.set_xlabel("Tag")
    ax.set_ylabel("Count")
    ax.set_xticks(range(len(tag_labels)))
    ax.set_xticklabels(tag_labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "risk_tag_counts.png"), dpi=160)
    _safe_close_fig(fig)

    # 7) Contradiction rate (single bar)
    total_ok = sum(rl_vals)
    contrad = c["contradiction_total"]
    rate = (contrad / total_ok) if total_ok else 0.0
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(["contradictions"], [rate])
    ax.set_ylim(0, 1)
    ax.set_title("Contradiction Rate (comments vs ratings)")
    ax.set_ylabel("Rate")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "contradiction_rate.png"), dpi=160)
    _safe_close_fig(fig)


# =============================================================================
# PER DOC
# =============================================================================

def init_answers() -> Dict[str, Dict[str,str]]:
    return {
        "q1": {"A":"UNKNOWN","B":"UNKNOWN"},
        "q2": {"A":"UNKNOWN","B":"UNKNOWN","C":"UNKNOWN","D":"UNKNOWN"},
        "q3": {"A":"UNKNOWN","B":"UNKNOWN","C":"UNKNOWN"},
        "q4": {"A":"UNKNOWN","B":"UNKNOWN","C":"UNKNOWN"},
    }

def process_pdf(pdf_path: str, api_key: str, model: str) -> Dict[str, Any]:
    doc_name = os.path.basename(pdf_path)
    base = os.path.splitext(doc_name)[0]

    page_count = get_pdf_page_count(pdf_path)
    if page_count < 2:
        return {"status":"error","source_file":doc_name,"error":f"PDF has {page_count} page(s), expected >=2."}

    p1, p2 = pdf_pages_to_bgr(pdf_path, [0,1], RENDER_DPI)

    ensure_dir(RAW_OAI_DIR)

    p1_raw = os.path.join(RAW_OAI_DIR, f"{base}__page1.json.txt")
    p2_raw = os.path.join(RAW_OAI_DIR, f"{base}__page2_hw.json.txt")

    # Parallelize page1 + page2 extraction inside the doc
    out1 = None
    out2 = None

    if PER_DOC_PARALLEL:
        def _do_p1():
            c = OpenAI(api_key=api_key)
            return extract_page1_ensemble(c, model, p1, p1_raw)

        def _do_p2():
            c = OpenAI(api_key=api_key)
            return extract_page2_handwriting(c, model, p2, p2_raw)

        with ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(_do_p1)
            f2 = ex.submit(_do_p2)
            out1 = f1.result()
            out2 = f2.result()
    else:
        c = OpenAI(api_key=api_key)
        out1 = extract_page1_ensemble(c, model, p1, p1_raw)
        out2 = extract_page2_handwriting(c, model, p2, p2_raw)

    if "error" in (out1 or {}):
        return {"status":"error","source_file":doc_name,"error":out1.get("error","page1 error")}

    header = (out1 or {}).get("header", {}) or {}
    answers = init_answers()
    answers_in = (out1 or {}).get("answers", {}) or {}

    for q in answers:
        for s in answers[q]:
            v = str(((answers_in.get(q) or {}).get(s) or "")).strip()
            if q == "q1":
                answers[q][s] = normalize_scale(v)
            else:
                answers[q][s] = normalize_yesno(v)

    if "error" in (out2 or {}):
        out2 = {"layout":"unknown", "trainee_comments":"", "staff_comments":{}, "prompted_qas":[]}

    layout = str((out2 or {}).get("layout","") or "").strip().lower()
    if layout not in {"prompted_standard", "simple", "wsd"}:
        layout = "unknown"

    trainee_comments = str((out2 or {}).get("trainee_comments","") or "").strip()
    staff_comments = (out2 or {}).get("staff_comments", {}) or {}

    if layout == "wsd":
        canon_prompts = CANON_PAGE2_PROMPTS_WSD
    else:
        canon_prompts = CANON_PAGE2_PROMPTS

    prompted_qas_raw = (out2 or {}).get("prompted_qas", [])
    prompted_qas = reorder_prompted_qas(prompted_qas_raw, canon_prompts)
    prompted_qas_map = build_prompted_qas_map(prompted_qas, canon_prompts)

    parts = []
    if trainee_comments:
        parts.append(trainee_comments)

    for qa in prompted_qas:
        p = str(qa.get("prompt","") or "").strip()
        a = str(qa.get("answer","") or "").strip()
        if p and a:
            parts.append(f"{p}: {a}")
        elif a:
            parts.append(a)

    for role_key, label in [("ops","OPS Comments"), ("lcpo","LCPO Comments"), ("divo","DIVO Comments")]:
        val = str(staff_comments.get(role_key,"") or "").strip()
        if val:
            parts.append(f"{label}: {val}")

    comments = "\n".join([x for x in parts if x]).strip()

    has_prompt_answers = any((qa.get("answer") or "").strip() for qa in prompted_qas)
    if layout == "wsd":
        version = "wsd_back"
    elif layout == "prompted_standard" or has_prompt_answers:
        version = "prompted_back"
    else:
        version = "simple_back"

    # Upgraded triage (with contradiction + tags + risk_score + keyword guardrails)
    ca = analyze_comments_v2(OpenAI(api_key=api_key), model, comments, answers)

    structured = dict(prompted_qas_map)
    structured["OPS Comments"] = str(staff_comments.get("ops","") or "").strip()
    structured["LCPO Comments"] = str(staff_comments.get("lcpo","") or "").strip()
    structured["DIVO Comments"] = str(staff_comments.get("divo","") or "").strip()

    return {
        "status":"ok",
        "source_file": doc_name,
        "version_detected": version,
        "header": {
            "course": str(header.get("course","") or "").strip(),
            "unit": str(header.get("unit","") or "").strip(),
            "division": str(header.get("division","") or "").strip(),
            "instructor": str(header.get("instructor","") or "").strip(),
            "date": str(header.get("date","") or "").strip(),
            "cin": str(header.get("cin","") or "").strip(),
        },
        "answers": answers,
        "free_text_sections": {
            "trainee_comments": trainee_comments,
            "staff_comments": {
                "ops": str(staff_comments.get("ops","") or "").strip(),
                "lcpo": str(staff_comments.get("lcpo","") or "").strip(),
                "divo": str(staff_comments.get("divo","") or "").strip(),
            },
            "other_sections": {}
        },
        "prompted_qas": prompted_qas,
        "prompted_qas_structured": structured,
        "comments": comments,
        "comments_analysis": ca,
        "debug": {
            "page_count": page_count,
            "openai_model": model,
            "page2_layout_detected": layout,
            "page2_canon_used": "wsd" if canon_prompts is CANON_PAGE2_PROMPTS_WSD else "standard",
            "per_doc_parallel": PER_DOC_PARALLEL
        }
    }


# =============================================================================
# BATCH MAIN
# =============================================================================

def run_batch():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(RAW_OAI_DIR)
    ensure_dir(ANALYSIS_CHARTS_DIR)

    load_dotenv(DOTENV_PATH)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing (.env or env vars).")

    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    pdfs = list_pdfs(SCANNED_DOCS_DIR)
    if not pdfs:
        raise RuntimeError(f"No PDFs found in {SCANNED_DOCS_DIR}")

    if DOC_LIMIT and DOC_LIMIT > 0:
        pdfs = pdfs[:DOC_LIMIT]

    results = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_dir": SCANNED_DOCS_DIR,
            "openai_model": model,
            "page2_prompt_orders": {
                "standard": CANON_PAGE2_PROMPTS,
                "wsd": CANON_PAGE2_PROMPTS_WSD
            },
            "doc_limit": DOC_LIMIT,
            "workers": MAX_WORKERS,
            "per_doc_parallel": PER_DOC_PARALLEL
        },
        "documents": {}
    }

    workers = max(1, min(MAX_WORKERS, len(pdfs)))
    print(f"[INFO] Processing {len(pdfs)} PDFs with {workers} worker(s)... (DOC_LIMIT={DOC_LIMIT}, PER_DOC_PARALLEL={PER_DOC_PARALLEL})")

    def _run_single(pdf_path: str) -> Dict[str, Any]:
        return process_pdf(pdf_path, api_key=api_key, model=model)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_run_single, p): p for p in pdfs}
        for i, fut in enumerate(as_completed(futures), 1):
            p = futures[fut]
            name = os.path.basename(p)
            print(f"[INFO] ({i}/{len(pdfs)}) {name}")
            try:
                results["documents"][name] = fut.result()
            except Exception as e:
                results["documents"][name] = {"status":"error", "source_file": name, "error": str(e)}

    with open(RESULTS_ALL_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    analysis = build_compact_analysis(results["documents"])
    with open(ANALYSIS_JSON, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    # Charts (basic + risk tags + contradiction)
    try:
        save_analysis_charts(results, analysis, ANALYSIS_CHARTS_DIR)
    except Exception as e:
        print("[WARN] Chart generation failed:", str(e))

    print("[INFO] DONE")
    print(" -", RESULTS_ALL_JSON)
    print(" -", ANALYSIS_JSON)
    print(" - charts:", ANALYSIS_CHARTS_DIR)
    print(" - raw OpenAI:", RAW_OAI_DIR)
    if DOC_LIMIT and DOC_LIMIT > 0:
        print(f"[INFO] DOC_LIMIT was set to {DOC_LIMIT} (processed first {DOC_LIMIT} PDFs in sorted order).")


if __name__ == "__main__":
    run_batch()
