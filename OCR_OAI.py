"""

- Page 1: OpenAI extraction -> header + Q1–Q4
- Page 2: OpenAI handwriting-only extraction (4 crops) -> trainee/staff/prompted QAs
- Reorders prompted_qas to match the document's question order (canonical list)
- Normalizes Y/N -> YES/NO
- Robust retry on 429

Install:
  pip install pymupdf opencv-python numpy python-dotenv openai
"""

import os, re, json, time, base64, difflib, threading
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz
import cv2
import numpy as np
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

RENDER_DPI = 150
MAX_LONG_EDGE = 1100

MIN_SECONDS_BETWEEN_REQUESTS = 0.05
OPENAI_RETRIES = 5
DEFAULT_MODEL = "gpt-4o-mini"
MAX_WORKERS = max(1, int(os.getenv("DOCEXTRACT_WORKERS", "3")))

# Canonical order of the prompted questions on page 2 (edit to match your exact form text)
CANON_PAGE2_PROMPTS = [
    "Was the instructor able to clearly communicate all safety rules and instructions to you and clarify any questions asked?",
    "Is there anything that could be done differently to increase safety awareness, or improve the level of training?",
    "What skills, lessons or methods were especially effective for you?",
    "What can we improve upon?",
    "Do you feel we addressed your needs? If not, then why?"
]


# =============================================================================
# THROTTLE
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

def pdf_page_to_bgr(pdf_path: str, page_index: int, dpi: int) -> np.ndarray:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    doc.close()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return downscale_long_edge(img_bgr, MAX_LONG_EDGE)

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


# =============================================================================
# PAGE 2: flexible extraction for both layouts (prompted vs simple comment page)
# =============================================================================

def prompt_page2_flexible(canon_prompts: List[str]) -> str:
    prompts_text = "\n".join([f"- {p}" for p in canon_prompts])
    return f"""
You will get crops covering page 2 of a training feedback form.
There are TWO possible layouts:
1) Prompted back: printed free-response questions (listed below) plus OPS/LCPO/DIVO comment boxes.
2) Simple back: a single blank comment box without those questions.

Extract ONLY HANDWRITTEN content and map it.

Return ONLY JSON:
{{
  "layout": "prompted" | "simple" | "unknown",
  "trainee_comments": "",
  "staff_comments": {{"ops":"","lcpo":"","divo":""}},
  "prompted_qas": [{{"prompt":"","answer":""}}]
}}

Rules:
- Always include prompted_qas entries for these prompts, in this order (leave answer="" if blank):
{prompts_text}
- For the simple layout, put all handwriting into trainee_comments and leave prompted answers "".
- For the prompted layout, fill answers for the matching questions; if blank, keep "".
- Only handwriting goes into answers; do NOT copy printed prompts or labels as answers.
- Copy handwritten OPS/LCPO/DIVO notes into staff_comments.
- Use empty strings for anything unwritten.
""".strip()

def extract_page2_handwriting(client: OpenAI, model: str, page2_bgr: np.ndarray, raw_path: str) -> Dict[str, Any]:
    p2 = enhance_handwriting(page2_bgr)
    # Use generous, overlapping strips to avoid trimming top/bottom handwriting.
    crops = [
        ("p2_top", crop_rel(p2, 0.02, 0.02, 0.98, 0.42)),
        ("p2_mid", crop_rel(p2, 0.02, 0.35, 0.98, 0.72)),
        ("p2_bottom", crop_rel(p2, 0.02, 0.65, 0.98, 0.98)),
    ]
    return openai_json(
        client, model,
        prompt_page2_flexible(CANON_PAGE2_PROMPTS),
        images=crops,
        max_output_tokens=900,
        raw_save_path=raw_path
    )


# =============================================================================
# ORDERING: reorder prompted_qas to match CANON_PAGE2_PROMPTS
# =============================================================================

def _norm_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s\?]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def reorder_prompted_qas(prompted_qas: Any) -> List[Dict[str, str]]:
    """
    Return prompted_qas ordered by the document's canonical prompt order.
    - Fuzzy match extracted prompts to canonical prompts
    - Any unmatched QAs go at the end in original order
    """
    if not isinstance(prompted_qas, list):
        return []

    canon = CANON_PAGE2_PROMPTS
    canon_norm = [_norm_text(c) for c in canon]

    used = set()
    ordered: List[Dict[str, str]] = []

    # For each canonical prompt, find best matching extracted QA
    for ci, c in enumerate(canon_norm):
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
        # Accept if reasonably similar
        if best_i is not None and best_score >= 0.55:
            used.add(best_i)
            qa = prompted_qas[best_i] or {}
            ordered.append({
                "prompt": str(qa.get("prompt","") or "").strip(),
                "answer": str(qa.get("answer","") or "").strip()
            })

    # Append remaining unmatched in original order
    for i, qa in enumerate(prompted_qas):
        if i in used:
            continue
        qa = qa or {}
        ordered.append({
            "prompt": str(qa.get("prompt","") or "").strip(),
            "answer": str(qa.get("answer","") or "").strip()
        })

    # Drop empty rows
    ordered = [qa for qa in ordered if qa.get("prompt") or qa.get("answer")]
    return ordered


# =============================================================================
# STRUCTURED PAGE 2 OUTPUT
# =============================================================================

def build_prompted_qas_map(prompted_qas: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Map prompted_qas into a fixed question->answer dict using canonical prompts.
    Falls back to empty string when missing.
    """
    out: Dict[str, str] = {q: "" for q in CANON_PAGE2_PROMPTS}
    if not prompted_qas:
        return out

    # Reorder first to align best-effort with canonical ordering
    ordered = reorder_prompted_qas(prompted_qas)

    canon_norm = [_norm_text(c) for c in CANON_PAGE2_PROMPTS]
    for qa in ordered:
        p = _norm_text(qa.get("prompt", ""))
        a = str(qa.get("answer", "") or "").strip()
        if not (p or a):
            continue
        best_i = None
        best_score = 0.0
        for idx, cn in enumerate(canon_norm):
            if out[CANON_PAGE2_PROMPTS[idx]]:
                continue
            score = difflib.SequenceMatcher(None, p, cn).ratio()
            if score > best_score:
                best_score = score
                best_i = idx
        if best_i is not None and best_score >= 0.45:
            # Only use the handwritten answer; leave empty string if blank.
            out[CANON_PAGE2_PROMPTS[best_i]] = a
    return out


# =============================================================================
# COMMENTS ANALYSIS 
# =============================================================================

def prompt_comment_triage(comments: str) -> str:
    return f"""
Analyze trainee feedback. Summarize in 1 sentence.
Flag human review if: unsafe, harassment/bullying, threats/harm, serious safety failures.

Return ONLY JSON:
{{"summary":"","requires_human_review":false,"risk_level":"none","reason":""}}

Comments:
{comments}
""".strip()

def analyze_comments(client: OpenAI, model: str, comments: str) -> Dict[str, Any]:
    if not comments.strip():
        return {"summary":"","requires_human_review":False,"risk_level":"none","reason":"No comments extracted."}
    out = openai_json(client, model, prompt_comment_triage(comments), images=[], max_output_tokens=350)
    if "error" in out:
        return {"summary":"","requires_human_review":False,"risk_level":"none","reason":"LLM triage failed.", "raw_error": out["error"]}
    rl = str(out.get("risk_level","none") or "none").lower().strip()
    if rl not in {"none","low","medium","high"}:
        rl = "none"
    return {
        "summary": str(out.get("summary","") or "").strip(),
        "requires_human_review": bool(out.get("requires_human_review", False)),
        "risk_level": rl,
        "reason": str(out.get("reason","") or "").strip()
    }


# =============================================================================
# COMPACT Q1–Q4 analysis 
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
    risk_counts = {"low":0,"medium":0,"high":0,"unknown":0}
    docs_by_risk = {"low":[], "medium":[], "high":[], "unknown":[]}

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

        if bool(ca.get("requires_human_review", False)):
            flagged_docs.append(doc_name)
            rl = str(ca.get("risk_level", "")).strip().lower()
            if rl not in {"low","medium","high"}: rl = "unknown"
            risk_counts[rl] += 1
            docs_by_risk[rl].append(doc_name)

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
            "risk_level_counts": risk_counts,
            "flagged_documents": flagged_docs,
            "flagged_documents_by_risk": docs_by_risk
        }
    }


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

def process_pdf(pdf_path: str, client: OpenAI, model: str) -> Dict[str, Any]:
    doc_name = os.path.basename(pdf_path)
    base = os.path.splitext(doc_name)[0]

    page_count = get_pdf_page_count(pdf_path)
    if page_count < 2:
        return {"status":"error","source_file":doc_name,"error":f"PDF has {page_count} page(s), expected >=2."}

    p1, p2 = pdf_pages_to_bgr(pdf_path, [0,1], RENDER_DPI)

    ensure_dir(RAW_OAI_DIR)

    # Page 1
    p1_raw = os.path.join(RAW_OAI_DIR, f"{base}__page1.json.txt")
    out1 = extract_page1(client, model, p1, p1_raw)
    if "error" in out1:
        return {"status":"error","source_file":doc_name,"error":out1["error"]}

    header = out1.get("header", {}) or {}
    answers = init_answers()
    answers_in = out1.get("answers", {}) or {}

    for q in answers:
        for s in answers[q]:
            v = str(((answers_in.get(q) or {}).get(s) or "")).strip()
            if q == "q1":
                answers[q][s] = normalize_scale(v)
            else:
                answers[q][s] = normalize_yesno(v)

    # Page 2 handwriting
    p2_raw = os.path.join(RAW_OAI_DIR, f"{base}__page2_hw.json.txt")
    out2 = extract_page2_handwriting(client, model, p2, p2_raw)
    if "error" in out2:
        out2 = {"trainee_comments":"", "staff_comments":{}, "prompted_qas":[]}

    layout = str(out2.get("layout","") or "").strip().lower()
    if layout not in {"prompted","simple"}:
        layout = "unknown"

    trainee_comments = str(out2.get("trainee_comments","") or "").strip()
    staff_comments = out2.get("staff_comments", {}) or {}
    prompted_qas = reorder_prompted_qas(out2.get("prompted_qas", []))
    prompted_qas_map = build_prompted_qas_map(prompted_qas)

    # Build comments in FORM ORDER:
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

    # Include staff comments in the combined comment blob for downstream analysis
    for role_key, label in [("ops","OPS Comments"), ("lcpo","LCPO Comments"), ("divo","DIVO Comments")]:
        val = str(staff_comments.get(role_key,"") or "").strip()
        if val:
            parts.append(f"{label}: {val}")

    comments = "\n".join([x for x in parts if x]).strip()

    has_prompt_answers = any((qa.get("answer") or "").strip() for qa in prompted_qas)
    version = "prompted_back" if layout == "prompted" or has_prompt_answers else "simple_back"
    ca = analyze_comments(client, model, comments)

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
        "prompted_qas_structured": {
            **prompted_qas_map,
            "OPS Comments": str(staff_comments.get("ops","") or "").strip(),
            "LCPO Comments": str(staff_comments.get("lcpo","") or "").strip(),
            "DIVO Comments": str(staff_comments.get("divo","") or "").strip(),
        },
        "comments": comments,
        "comments_analysis": ca,
        "debug": {"page_count": page_count, "openai_model": model, "page2_layout_detected": layout}
    }


# =============================================================================
# BATCH MAIN
# =============================================================================

def run_batch():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(RAW_OAI_DIR)

    load_dotenv(DOTENV_PATH)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing (.env or env vars).")

    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    pdfs = list_pdfs(SCANNED_DOCS_DIR)
    if not pdfs:
        raise RuntimeError(f"No PDFs found in {SCANNED_DOCS_DIR}")

    results = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_dir": SCANNED_DOCS_DIR,
            "openai_model": model,
            "page2_prompt_order": CANON_PAGE2_PROMPTS
        },
        "documents": {}
    }

    workers = max(1, min(MAX_WORKERS, len(pdfs)))
    print(f"[INFO] Processing {len(pdfs)} PDFs (ordered page2 prompts) with {workers} worker(s)...")

    def _run_single(pdf_path: str) -> Dict[str, Any]:
        # Create per-thread client to avoid shared state issues under concurrency.
        local_client = OpenAI(api_key=api_key)
        return process_pdf(pdf_path, local_client, model)

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

    print("[INFO] DONE")
    print(" -", RESULTS_ALL_JSON)
    print(" -", ANALYSIS_JSON)
    print(" - raw OpenAI:", RAW_OAI_DIR)


if __name__ == "__main__":
    run_batch()
