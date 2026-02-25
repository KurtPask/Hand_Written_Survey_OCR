# =========================
# utils.py (Databricks notebook block) — v3
# Fixes:
#  - JSON serialization of numpy types (np.int64, np.float32, arrays, tensors)
#  - Much stronger crop fidelity:
#      * Per-field LOCAL alignment to template (ECC translation; phase-corr init)
#      * Ink extraction after local alignment
#      * For ruled/free-text fields: detect horizontal lines ON SCAN crop to bound answer
#      * Y-projection tightening to focus on handwriting band and exclude prompt text
#      * Connected-component based x tightening (keeps handwriting, drops stray noise)
# =========================

import os, re, json, time, uuid, traceback
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import cv2
from PIL import Image

import fitz  # PyMuPDF
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

import pytesseract
from rapidfuzz import fuzz


# -------------------------
# JSON safety
# -------------------------

def make_json_safe(x: Any) -> Any:
    """
    Convert numpy / torch types into JSON-serializable Python types.
    """
    # None / primitives
    if x is None or isinstance(x, (str, int, float, bool)):
        return x

    # numpy scalars
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)

    # numpy arrays
    if isinstance(x, np.ndarray):
        return x.tolist()

    # torch tensors
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()

    # dict / list / tuple
    if isinstance(x, dict):
        return {str(k): make_json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [make_json_safe(v) for v in x]

    # fallback
    try:
        return str(x)
    except Exception:
        return None


# -------------------------
# Databricks / path helpers
# -------------------------

def _strip_trailing_slash(p: str) -> str:
    return p[:-1] if p.endswith("/") else p

def join_path(a: str, b: str) -> str:
    return _strip_trailing_slash(a) + "/" + b.lstrip("/")

def is_s3_path(p: str) -> bool:
    return p.startswith("s3a://") or p.startswith("s3://")

def is_dbfs_uri(p: str) -> bool:
    return p.startswith("dbfs:/")

def dbfs_uri_to_local(p: str) -> str:
    assert is_dbfs_uri(p)
    return "/dbfs/" + p[len("dbfs:/"):].lstrip("/")

def ensure_dir_local(local_dir: str) -> None:
    os.makedirs(local_dir, exist_ok=True)

def ensure_dir_dbutils(dbutils, path: str) -> None:
    try:
        dbutils.fs.mkdirs(path)
    except Exception:
        pass

def _notebook_dir_guess(dbutils) -> Optional[str]:
    try:
        nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        nb_dir = os.path.dirname(nb_path)
        return "/Workspace" + nb_dir
    except Exception:
        return None

def resolve_local_or_relative_path(dbutils, path: str) -> str:
    if os.path.isabs(path) and os.path.exists(path):
        return path

    cand = os.path.abspath(path)
    if os.path.exists(cand):
        return cand

    nb_dir = _notebook_dir_guess(dbutils)
    if nb_dir:
        cand2 = os.path.join(nb_dir, path)
        if os.path.exists(cand2):
            return cand2

    raise FileNotFoundError(
        f"Could not resolve local path '{path}'. Tried:\n"
        f"  - {os.path.abspath(path)}\n"
        f"  - notebook-relative: {os.path.join(nb_dir, path) if nb_dir else '(notebook path unavailable)'}"
    )

def read_text_from_path(dbutils, path: str) -> str:
    if is_s3_path(path):
        tmp_dbfs = "dbfs:/tmp/ocr_cfg/" + uuid.uuid4().hex + ".txt"
        ensure_dir_dbutils(dbutils, os.path.dirname(tmp_dbfs))
        dbutils.fs.cp(path, tmp_dbfs, recurse=False)
        local = dbfs_uri_to_local(tmp_dbfs)
        with open(local, "r", encoding="utf-8") as f:
            return f.read()

    if is_dbfs_uri(path):
        local = dbfs_uri_to_local(path)
        with open(local, "r", encoding="utf-8") as f:
            return f.read()

    local = resolve_local_or_relative_path(dbutils, path)
    with open(local, "r", encoding="utf-8") as f:
        return f.read()

def load_json_config(dbutils, config_path: str) -> Dict[str, Any]:
    return json.loads(read_text_from_path(dbutils, config_path))

def write_json_to_output(dbutils, obj: Dict[str, Any], out_path: str) -> None:
    safe = make_json_safe(obj)
    content = json.dumps(safe, ensure_ascii=False, indent=2)

    if is_s3_path(out_path):
        tmp_dbfs = "dbfs:/tmp/ocr_out/" + uuid.uuid4().hex + ".json"
        ensure_dir_dbutils(dbutils, os.path.dirname(tmp_dbfs))
        tmp_local = dbfs_uri_to_local(tmp_dbfs)
        ensure_dir_local(os.path.dirname(tmp_local))
        with open(tmp_local, "w", encoding="utf-8") as f:
            f.write(content)
        ensure_dir_dbutils(dbutils, os.path.dirname(out_path))
        dbutils.fs.cp(tmp_dbfs, out_path, recurse=False)
        return

    if is_dbfs_uri(out_path):
        local = dbfs_uri_to_local(out_path)
        ensure_dir_local(os.path.dirname(local))
        with open(local, "w", encoding="utf-8") as f:
            f.write(content)
        return

    ensure_dir_dbutils(dbutils, os.path.dirname(out_path))
    dbutils.fs.put(out_path, content, overwrite=True)

def write_rgb_image_to_output(dbutils, rgb: np.ndarray, out_path: str) -> None:
    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")

    if is_s3_path(out_path):
        tmp_dbfs = "dbfs:/tmp/ocr_imgs/" + uuid.uuid4().hex + ".png"
        ensure_dir_dbutils(dbutils, os.path.dirname(tmp_dbfs))
        tmp_local = dbfs_uri_to_local(tmp_dbfs)
        ensure_dir_local(os.path.dirname(tmp_local))
        img.save(tmp_local, format="PNG")
        ensure_dir_dbutils(dbutils, os.path.dirname(out_path))
        dbutils.fs.cp(tmp_dbfs, out_path, recurse=False)
        return

    if is_dbfs_uri(out_path):
        local = dbfs_uri_to_local(out_path)
        ensure_dir_local(os.path.dirname(local))
        img.save(local, format="PNG")
        return

    tmp_dbfs = "dbfs:/tmp/ocr_imgs/" + uuid.uuid4().hex + ".png"
    ensure_dir_dbutils(dbutils, os.path.dirname(tmp_dbfs))
    tmp_local = dbfs_uri_to_local(tmp_dbfs)
    ensure_dir_local(os.path.dirname(tmp_local))
    img.save(tmp_local, format="PNG")
    ensure_dir_dbutils(dbutils, os.path.dirname(out_path))
    dbutils.fs.cp(tmp_dbfs, out_path, recurse=False)

def list_pdfs(dbutils, input_path: str, recurse: bool = True) -> List[str]:
    input_path = _strip_trailing_slash(input_path)
    out = []
    stack = [input_path]
    while stack:
        p = stack.pop()
        try:
            items = dbutils.fs.ls(p)
            for it in items:
                if it.isDir():
                    if recurse:
                        stack.append(_strip_trailing_slash(it.path))
                else:
                    if it.path.lower().endswith(".pdf"):
                        out.append(it.path)
        except Exception:
            if p.lower().endswith(".pdf"):
                out.append(p)
    return sorted(set(out))

def copy_to_local_file(dbutils, src_path: str, local_dir: str) -> str:
    ensure_dir_local(local_dir)
    base = os.path.basename(src_path)
    local_path = os.path.join(local_dir, base)

    dst_dbfs = "dbfs:/tmp/ocr_work/" + uuid.uuid4().hex + "/" + base
    ensure_dir_dbutils(dbutils, os.path.dirname(dst_dbfs))
    dbutils.fs.cp(src_path, dst_dbfs, recurse=False)

    local_from_dbfs = dbfs_uri_to_local(dst_dbfs)
    ensure_dir_local(os.path.dirname(local_path))
    os.replace(local_from_dbfs, local_path)
    return local_path


# -------------------------
# PDF rendering
# -------------------------

def render_pdf_pages(pdf_local_path: str, dpi: int = 300, max_pages: Optional[int] = None) -> List[Image.Image]:
    doc = fitz.open(pdf_local_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    n = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
    pages = []
    for i in range(n):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(img)
    doc.close()
    return pages

def extract_embedded_text(pdf_local_path: str, min_chars: int = 80) -> Optional[str]:
    try:
        doc = fitz.open(pdf_local_path)
        chunks = []
        for i in range(doc.page_count):
            t = (doc.load_page(i).get_text("text") or "").strip()
            if t:
                chunks.append(t)
        doc.close()
        text = "\n\n".join(chunks).strip()
        return text if len(text) >= min_chars else None
    except Exception:
        return None


# -------------------------
# Image helpers / preprocessing
# -------------------------

def pil_to_rgb_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))

def rgb_np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")

def to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

def normalize_for_ocr(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    g = cv2.bilateralFilter(g, d=7, sigmaColor=50, sigmaSpace=50)
    return g

def suppress_border_noise(gray: np.ndarray, border_frac: float = 0.03) -> np.ndarray:
    """
    Suppress dark border artifacts/stains near page edges that can destabilize global alignment.
    """
    H, W = gray.shape[:2]
    b = max(8, int(min(H, W) * border_frac))
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(mask, (b, b), (W - b, H - b), 255, -1)
    med = int(np.median(gray))
    out = gray.copy()
    out[mask == 0] = med
    return out

def binarize_inv(gray: np.ndarray) -> np.ndarray:
    g = normalize_for_ocr(gray)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 9)
    return bw

def downscale_max(gray: np.ndarray, max_dim: int = 1600) -> Tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    s = 1.0
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        gray = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return gray, s

def bbox_norm_to_px(b: List[float], W: int, H: int) -> List[int]:
    x0 = int(round(b[0] * W)); y0 = int(round(b[1] * H))
    x1 = int(round(b[2] * W)); y1 = int(round(b[3] * H))
    x0 = max(0, min(W - 1, x0)); x1 = max(1, min(W, x1))
    y0 = max(0, min(H - 1, y0)); y1 = max(1, min(H, y1))
    if x1 <= x0: x1 = min(W, x0 + 1)
    if y1 <= y0: y1 = min(H, y0 + 1)
    return [x0, y0, x1, y1]

def clamp_box(box: List[int], W: int, H: int) -> List[int]:
    x0, y0, x1, y1 = box
    x0 = max(0, min(W-1, x0))
    y0 = max(0, min(H-1, y0))
    x1 = max(1, min(W,   x1))
    y1 = max(1, min(H,   y1))
    if x1 <= x0: x1 = min(W, x0 + 1)
    if y1 <= y0: y1 = min(H, y0 + 1)
    return [x0, y0, x1, y1]

def crop_px(rgb: np.ndarray, box: List[int], pad: int = 0) -> np.ndarray:
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = box
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(W, x1 + pad); y1 = min(H, y1 + pad)
    return rgb[y0:y1, x0:x1].copy()

def crop_gray(gray: np.ndarray, box: List[int], pad: int = 0) -> np.ndarray:
    H, W = gray.shape[:2]
    x0, y0, x1, y1 = box
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(W, x1 + pad); y1 = min(H, y1 + pad)
    return gray[y0:y1, x0:x1].copy()

def shift_box(box: List[int], dx: int, dy: int, W: int, H: int) -> List[int]:
    x0, y0, x1, y1 = box
    return clamp_box([x0 + dx, y0 + dy, x1 + dx, y1 + dy], W, H)

def upscale_for_small_ocr(rgb: np.ndarray, min_side: int = 80) -> np.ndarray:
    h, w = rgb.shape[:2]
    if min(h, w) >= min_side:
        return rgb
    scale = float(min_side) / max(1, min(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


# -------------------------
# Local alignment (per-field crop) to template crop
# -------------------------

def _phase_corr_shift(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """
    Estimate shift to align b->a. Returns (dx, dy, response).
    """
    a32 = np.float32(a)
    b32 = np.float32(b)
    (dx, dy), resp = cv2.phaseCorrelate(a32, b32)
    return float(dx), float(dy), float(resp)

def local_align_scan_crop_to_template(scan_gray: np.ndarray, templ_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Align scan_gray to templ_gray using translation ECC with phase-correlation init.
    Returns aligned_scan_gray (same size as templ_gray), the 2x3 warp matrix, and diagnostics.
    """
    # work on edges to reduce sensitivity to brightness
    sg = normalize_for_ocr(scan_gray)
    tg = normalize_for_ocr(templ_gray)

    se = cv2.Canny(sg, 80, 160)
    te = cv2.Canny(tg, 80, 160)

    dx, dy, resp = _phase_corr_shift(te, se)  # shift se towards te
    warp = np.array([[1, 0, dx],
                     [0, 1, dy]], dtype=np.float32)

    diag = {"phase_dx": dx, "phase_dy": dy, "phase_resp": resp}

    # ECC refinement (translation)
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
        cc, warp2 = cv2.findTransformECC(te, se, warp, cv2.MOTION_TRANSLATION, criteria, None, 3)
        diag["ecc_cc"] = float(cc)
        warp = warp2
    except Exception as e:
        diag["ecc_error"] = str(e)

    aligned = cv2.warpAffine(
        scan_gray, warp, (templ_gray.shape[1], templ_gray.shape[0]),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return aligned, warp, diag

def warp_rgb_by_affine(rgb: np.ndarray, warp2x3: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    return cv2.warpAffine(rgb, warp2x3, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# -------------------------
# Handwriting isolation utilities
# -------------------------

def detect_horizontal_lines(gray: np.ndarray, min_len_frac: float = 0.35) -> List[Tuple[int,int,int,int]]:
    """
    Detect long horizontal lines in a scan crop.
    Returns list of (x0,y0,x1,y1) sorted by y.
    """
    bw = binarize_inv(gray)  # ink/lines = 255

    H, W = bw.shape
    k = max(25, int(W * 0.35))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w >= int(min_len_frac * W) and h <= max(14, H // 60):
            out.append((x, y, x+w, y+h))
    out.sort(key=lambda b: b[1])
    return out

def remove_lines_inpaint(gray: np.ndarray) -> np.ndarray:
    """
    Remove strong horizontal/vertical lines by inpainting.
    """
    bw = binarize_inv(gray)
    H, W = bw.shape
    hk = max(25, W // 25)
    vk = max(25, H // 25)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))

    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel, iterations=1)
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel, iterations=1)

    mask = cv2.bitwise_or(h_lines, v_lines)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    g2 = cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)
    return g2

def y_band_from_mask(mask: np.ndarray, min_band_h: int = 45) -> Optional[Tuple[int,int,Dict[str,Any]]]:
    """
    Use y-projection to find the densest handwriting band.
    Returns (y0,y1,diag) in mask coordinates.
    """
    m = (mask > 0).astype(np.uint8)
    proj = m.sum(axis=1).astype(np.float32)
    if proj.max() <= 0:
        return None

    # smooth
    k = max(5, len(proj)//120)
    ker = np.ones(k, dtype=np.float32) / k
    sm = np.convolve(proj, ker, mode="same")

    peak = float(sm.max())
    yc = int(np.argmax(sm))
    thr = 0.35 * peak

    # expand around peak until below threshold
    y0 = yc
    while y0 > 0 and sm[y0] >= thr:
        y0 -= 1
    y1 = yc
    while y1 < len(sm)-1 and sm[y1] >= thr:
        y1 += 1

    # enforce min band
    if (y1 - y0) < min_band_h:
        mid = (y0 + y1) // 2
        y0 = max(0, mid - min_band_h//2)
        y1 = min(len(sm), y0 + min_band_h)

    diag = {"yproj_peak": peak, "y_center": yc, "y0": y0, "y1": y1}
    return y0, y1, diag

def components_bbox(mask: np.ndarray, keep_k: int = 3, min_area: int = 30) -> Optional[Tuple[int,int,int,int]]:
    """
    BBox around union of top-K components by area.
    """
    m = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return None
    comps = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            comps.append((area, x, y, x+w, y+h))
    if not comps:
        return None
    comps.sort(reverse=True, key=lambda t: t[0])
    take = comps[:max(1, keep_k)]
    x0 = min(t[1] for t in take); y0 = min(t[2] for t in take)
    x1 = max(t[3] for t in take); y1 = max(t[4] for t in take)
    return (x0, y0, x1, y1)


# -------------------------
# Ink extraction (template subtraction AFTER local alignment)
# -------------------------

def extract_ink_only(scan_rgb_crop: np.ndarray, templ_rgb_crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Returns (ink_rgb, ink_mask, diag) where ink_mask is 0/255.
    Key improvement: local align scan_crop to template_crop before subtraction.
    """
    sg0 = cv2.cvtColor(scan_rgb_crop, cv2.COLOR_RGB2GRAY)
    tg0 = cv2.cvtColor(templ_rgb_crop, cv2.COLOR_RGB2GRAY)

    # local align scan->template
    aligned_sg, warp, al_diag = local_align_scan_crop_to_template(sg0, tg0)

    # align RGB with the same local warp used for grayscale alignment
    aligned_rgb = cv2.warpAffine(scan_rgb_crop, warp, (templ_rgb_crop.shape[1], templ_rgb_crop.shape[0]),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    sg = normalize_for_ocr(aligned_sg)
    tg = normalize_for_ocr(tg0)

    diff = cv2.absdiff(sg, tg)
    diff = cv2.GaussianBlur(diff, (3, 3), 0)

    otsu_val, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = int(max(14, min(60, 0.65 * otsu_val)))
    _, mask = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)

    # denoise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)

    # create ink-only visualization
    out = np.full_like(aligned_sg, 255)
    out[mask > 0] = aligned_sg[mask > 0]
    ink_rgb = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)

    diag = {"otsu": float(otsu_val), "thr": int(thr), "align": al_diag}
    return ink_rgb, mask, diag


# -------------------------
# Printed OCR + confidence
# -------------------------

def has_working_tesseract() -> bool:
    try:
        _ = pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

def tesseract_ocr_with_conf(rgb_crop: np.ndarray, psm: int = 6, extra_cfg: str = "") -> Tuple[str, float]:
    gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)
    g = normalize_for_ocr(gray)
    b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 35, 15)
    cfg = f"--oem 1 --psm {psm} {extra_cfg}".strip()

    try:
        data = pytesseract.image_to_data(b, output_type=pytesseract.Output.DICT, config=cfg)
        words, confs = [], []
        for w, c in zip(data.get("text", []), data.get("conf", [])):
            w = (w or "").strip()
            try:
                c = float(c)
            except Exception:
                c = -1.0
            if w and c >= 0:
                words.append(w)
                confs.append(c)
        text = " ".join(words).strip()
        conf = float(np.mean(confs) / 100.0) if confs else 0.0
        return text, conf
    except Exception:
        try:
            text = pytesseract.image_to_string(b, config=cfg).strip()
            return text, 0.0
        except Exception:
            return "", 0.0


# -------------------------
# Specialized decoders
# -------------------------

def ocr_rating_1_5_tesseract(rgb_crop: np.ndarray) -> str:
    rgb2 = upscale_for_small_ocr(rgb_crop, min_side=90)
    g = cv2.cvtColor(rgb2, cv2.COLOR_RGB2GRAY)
    b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 31, 9)
    txt = pytesseract.image_to_string(
        b,
        config="--oem 1 --psm 10 -c tessedit_char_whitelist=12345"
    ).strip()
    m = re.search(r"[1-5]", txt)
    return m.group(0) if m else ""

def ocr_yes_no_tesseract(rgb_crop: np.ndarray) -> str:
    rgb2 = upscale_for_small_ocr(rgb_crop, min_side=110)
    g = cv2.cvtColor(rgb2, cv2.COLOR_RGB2GRAY)
    b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 31, 9)
    txt = pytesseract.image_to_string(
        b,
        config="--oem 1 --psm 8 -c tessedit_char_whitelist=YESNOyesno"
    ).strip().upper()
    txt = re.sub(r"[^A-Z]", "", txt)
    if "YES" in txt or txt.startswith("Y"):
        return "YES"
    if "NO" in txt or txt.startswith("N"):
        return "NO"
    return ""


# -------------------------
# TrOCR loading + inference
# -------------------------

def ensure_local_hf_model_dir(dbutils, fmz_root: str, model_name: str, local_root_dbfs: str = "dbfs:/tmp/fmz_models") -> str:
    fmz_root = _strip_trailing_slash(fmz_root)
    local_root_dbfs = _strip_trailing_slash(local_root_dbfs)

    candidates = [
        join_path(fmz_root, model_name),
        join_path(fmz_root, model_name.replace("-", "_")),
    ]

    local_dbfs_dir = join_path(local_root_dbfs, model_name)
    local_dir = dbfs_uri_to_local(local_dbfs_dir)

    def looks_ready(d: str) -> bool:
        return (
            os.path.exists(os.path.join(d, "config.json"))
            or os.path.exists(os.path.join(d, "preprocessor_config.json"))
        )

    if looks_ready(local_dir):
        return local_dir

    src = None
    for cand in candidates:
        try:
            dbutils.fs.ls(cand)
            src = cand
            break
        except Exception:
            continue
    if src is None:
        raise FileNotFoundError(f"Could not find model '{model_name}' under FMZ root: {fmz_root}")

    ensure_dir_dbutils(dbutils, local_dbfs_dir)
    dbutils.fs.cp(src, local_dbfs_dir, recurse=True)
    return local_dir

@dataclass
class TrocrBundle:
    name: str
    processor: Any
    model: Any

def load_trocr_bundle(dbutils, fmz_root: str, model_name: str, device: str) -> TrocrBundle:
    local_dir = ensure_local_hf_model_dir(dbutils, fmz_root, model_name)
    proc = TrOCRProcessor.from_pretrained(local_dir, local_files_only=True)
    mdl = VisionEncoderDecoderModel.from_pretrained(local_dir, local_files_only=True)
    mdl.to(device)
    mdl.eval()
    return TrocrBundle(name=model_name, processor=proc, model=mdl)

def trocr_predict_with_conf(bundle: TrocrBundle, img: Image.Image, device: str, max_new_tokens: int = 64) -> Tuple[str, float]:
    proc, mdl = bundle.processor, bundle.model
    with torch.inference_mode():
        pixel_values = proc(images=img.convert("RGB"), return_tensors="pt").pixel_values.to(device)
        out = mdl.generate(
            pixel_values,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
        text = proc.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()

        scores = out.scores
        seq = out.sequences[0]
        if scores and len(seq) >= 2:
            token_ids = seq[-len(scores):]
            probs = []
            for s, tid in zip(scores, token_ids):
                p = torch.softmax(s[0], dim=-1)[int(tid)].item()
                probs.append(p)
            conf = float(np.mean(probs)) if probs else 0.0
        else:
            conf = 0.0

    return text, conf

def looks_like_gibberish(text: str) -> bool:
    if not text:
        return True
    t = text.strip()
    if len(t) < 2:
        return True
    alnum = sum(ch.isalnum() for ch in t)
    return (alnum / max(1, len(t))) < 0.35

def pick_best_by_confidence(candidates: List[Tuple[str, str, float]]) -> Tuple[str, str, float]:
    best = None
    best_score = -1e9
    for eng, txt, conf in candidates:
        score = float(conf) - (0.15 if looks_like_gibberish(txt) else 0.0)
        if score > best_score:
            best_score = score
            best = (eng, txt, float(conf))
    return best if best is not None else ("", "", 0.0)

def pick_best_handwriting_ocr(
    trocr_printed: TrocrBundle,
    trocr_hand: TrocrBundle,
    crop_rgb: np.ndarray,
    device: str,
    max_new_tokens: int = 64
) -> Dict[str, Any]:
    img = rgb_np_to_pil(crop_rgb)
    t_h, c_h = trocr_predict_with_conf(trocr_hand, img, device=device, max_new_tokens=max_new_tokens)
    t_p, c_p = trocr_predict_with_conf(trocr_printed, img, device=device, max_new_tokens=max_new_tokens)

    eng, txt, conf = pick_best_by_confidence([
        (trocr_hand.name, t_h, c_h),
        (trocr_printed.name, t_p, c_p),
    ])
    return {"text": txt, "confidence": conf, "engine": eng}


# -------------------------
# Template alignment (scan -> template) (global)
# -------------------------

def _scale_H(H: np.ndarray, s_scan: float, s_templ: float) -> np.ndarray:
    T_s = np.array([[s_scan, 0, 0],
                    [0, s_scan, 0],
                    [0, 0, 1]], dtype=np.float64)
    T_t = np.array([[s_templ, 0, 0],
                    [0, s_templ, 0],
                    [0, 0, 1]], dtype=np.float64)
    return np.linalg.inv(T_t) @ H @ T_s

def estimate_homography_orb(scan_gray: np.ndarray, templ_gray: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    scan_g = suppress_border_noise(normalize_for_ocr(scan_gray))
    templ_g = suppress_border_noise(normalize_for_ocr(templ_gray))

    scan_d, s_scan = downscale_max(scan_g, max_dim=1600)
    templ_d, s_templ = downscale_max(templ_g, max_dim=1600)

    orb = cv2.ORB_create(nfeatures=5000, fastThreshold=7, scaleFactor=1.2, nlevels=8)
    kp1, des1 = orb.detectAndCompute(scan_d, None)
    kp2, des2 = orb.detectAndCompute(templ_d, None)

    if des1 is None or des2 is None or len(kp1) < 12 or len(kp2) < 12:
        return None, {"method": "orb", "reason": "insufficient_features"}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)[:800]

    if len(matches) < 12:
        return None, {"method": "orb", "reason": "insufficient_matches", "matches": len(matches)}

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H_down, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4.0)
    if H_down is None or mask is None:
        return None, {"method": "orb", "reason": "homography_failed", "matches": len(matches)}

    inliers = int(mask.ravel().sum())
    if inliers < 12:
        return None, {"method": "orb", "reason": "too_few_inliers", "matches": len(matches), "inliers": inliers}

    H_full = _scale_H(H_down.astype(np.float64), s_scan=s_scan, s_templ=s_templ)
    return H_full, {"method": "orb", "matches": int(len(matches)), "inliers": int(inliers)}

def estimate_affine_ecc(scan_gray: np.ndarray, templ_gray: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    scan_g = suppress_border_noise(normalize_for_ocr(scan_gray))
    templ_g = suppress_border_noise(normalize_for_ocr(templ_gray))

    scan_d, s_scan = downscale_max(scan_g, max_dim=1400)
    templ_d, s_templ = downscale_max(templ_g, max_dim=1400)

    th, tw = templ_d.shape[:2]
    scan_resized = cv2.resize(scan_d, (tw, th), interpolation=cv2.INTER_AREA)

    scan_e = cv2.Canny(scan_resized, 80, 160)
    templ_e = cv2.Canny(templ_d, 80, 160)

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-5)

    try:
        cc, warp = cv2.findTransformECC(templ_e, scan_e, warp, cv2.MOTION_AFFINE, criteria, None, 5)
        H_aff = np.eye(3, dtype=np.float64)
        H_aff[:2, :] = warp.astype(np.float64)

        r_x = tw / scan_d.shape[1]
        r_y = th / scan_d.shape[0]
        T_s = np.array([[s_scan, 0, 0],
                        [0, s_scan, 0],
                        [0, 0, 1]], dtype=np.float64)
        R = np.array([[r_x, 0, 0],
                      [0, r_y, 0],
                      [0, 0, 1]], dtype=np.float64)
        T_t = np.array([[s_templ, 0, 0],
                        [0, s_templ, 0],
                        [0, 0, 1]], dtype=np.float64)

        H_full = np.linalg.inv(T_t) @ H_aff @ R @ T_s
        return H_full, {"method": "ecc_affine", "cc": float(cc)}
    except Exception as e:
        return None, {"method": "ecc_affine", "reason": str(e)}

def warp_scan_to_template(scan_rgb: np.ndarray, templ_shape_hw: Tuple[int, int], H: np.ndarray) -> np.ndarray:
    Ht, Wt = templ_shape_hw
    return cv2.warpPerspective(scan_rgb, H, (Wt, Ht), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# -------------------------
# Config-driven geometry
# -------------------------

def resolve_field_bbox_px(field: Dict[str, Any], page_W: int, page_H: int, anchors_px: Dict[str, List[int]]) -> List[int]:
    if field.get("bbox_norm") is not None:
        return bbox_norm_to_px(field["bbox_norm"], page_W, page_H)

    below = field.get("below_anchor")
    if below:
        aid = below["anchor_id"]
        if aid not in anchors_px:
            return [0, 0, page_W, page_H]
        ax0, ay0, ax1, ay1 = anchors_px[aid]

        y0 = int(round(ay1 + below.get("y_offset_norm", 0.0) * page_H))
        h  = int(round(below.get("height_norm", 0.05) * page_H))
        y1 = min(page_H, y0 + h)

        x0n = below.get("x0_norm", None)
        x1n = below.get("x1_norm", None)
        x0 = ax0 if x0n is None else int(round(x0n * page_W))
        x1 = ax1 if x1n is None else int(round(x1n * page_W))

        return clamp_box([x0, y0, x1, y1], page_W, page_H)

    ruled = field.get("ruled_lines_below_anchor")
    if ruled:
        aid = ruled["anchor_id"]
        if aid not in anchors_px:
            return [0, 0, page_W, page_H]

        ax0, ay0, ax1, ay1 = anchors_px[aid]
        y0 = int(round(ay1 + ruled.get("min_y_offset_norm", 0.02) * page_H))
        y1 = int(round(ay1 + ruled.get("max_y_search_norm", 0.18) * page_H))
        x0 = int(round(ruled.get("x0_norm", ax0 / max(1, page_W)) * page_W))
        x1 = int(round(ruled.get("x1_norm", ax1 / max(1, page_W)) * page_W))

        return clamp_box([x0, y0, x1, y1], page_W, page_H)

    left = field.get("left_of_anchor")
    if left:
        aid = left["anchor_id"]
        if aid not in anchors_px:
            return [0, 0, page_W, page_H]
        ax0, ay0, ax1, ay1 = anchors_px[aid]
        ah = max(1, ay1 - ay0)

        width = int(round(max(0.05 * page_W, left.get("width_norm", 0.09) * page_W)))
        gap   = int(round(left.get("x_gap_norm", 0.01) * page_W))
        ypad  = int(round(max(left.get("y_pad_norm", 0.01) * page_H, 0.10 * ah)))

        x1 = max(0, ax0 - gap)
        x0 = max(0, x1 - width)
        y0 = max(0, ay0 - ypad)
        y1 = min(page_H, ay1 + ypad)
        return clamp_box([x0, y0, x1, y1], page_W, page_H)

    right = field.get("right_of_anchor")
    if right:
        aid = right["anchor_id"]
        if aid not in anchors_px:
            return [0, 0, page_W, page_H]
        ax0, ay0, ax1, ay1 = anchors_px[aid]
        ah = max(1, ay1 - ay0)

        width = int(round(max(0.05 * page_W, right.get("width_norm", 0.09) * page_W)))
        gap   = int(round(right.get("x_gap_norm", 0.01) * page_W))
        ypad  = int(round(max(right.get("y_pad_norm", 0.01) * page_H, 0.10 * ah)))

        x0 = min(page_W - 1, ax1 + gap)
        x1 = min(page_W, x0 + width)
        y0 = max(0, ay0 - ypad)
        y1 = min(page_H, ay1 + ypad)
        return clamp_box([x0, y0, x1, y1], page_W, page_H)

    return [0, 0, page_W, page_H]


# -------------------------
# Tightening ruled/free-text fields using SCAN lines + ink projection
# -------------------------

def refine_ruled_box_using_scan_lines(
    warped_gray: np.ndarray,
    init_box_xyxy: List[int],
    pad_px: int,
    max_lines: int = 1,
    min_line_length_frac: float = 0.30,
    above_pad: int = 70,
    below_pad: int = 45,
    x_expand: int = 15,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Within the init_box, detect horizontal ruled lines on the SCAN crop,
    and shrink y-range around the first max_lines lines found.
    """
    H, W = warped_gray.shape[:2]
    init_box_xyxy = clamp_box(init_box_xyxy, W, H)

    crop_g = crop_gray(warped_gray, init_box_xyxy, pad=pad_px)
    lines = detect_horizontal_lines(crop_g, min_len_frac=min_line_length_frac)

    diag = {"found_lines": len(lines)}
    if not lines:
        return init_box_xyxy, diag

    use = lines[:max(1, max_lines)]
    x0 = max(0, min(b[0] for b in use) - x_expand)
    x1 = min(crop_g.shape[1], max(b[2] for b in use) + x_expand)
    y0 = max(0, min(b[1] for b in use) - above_pad)
    y1 = min(crop_g.shape[0], max(b[3] for b in use) + below_pad)

    # map back to page coords
    x0p, y0p, x1p, y1p = init_box_xyxy
    ox = max(0, x0p - pad_px)
    oy = max(0, y0p - pad_px)

    refined = [ox + x0, oy + y0, ox + x1, oy + y1]
    refined = clamp_box(refined, W, H)

    diag["line_boxes"] = use
    diag["refined_box_xyxy"] = refined
    return refined, diag

def _window_sum_from_integral(ii: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    """
    Sum on [y0:y1, x0:x1] from cv2.integral output.
    """
    return float(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])

def refine_box_by_ink_delta_peak(
    scan_gray: np.ndarray,
    templ_gray: np.ndarray,
    init_box_xyxy: List[int],
    search_px: int = 36,
    step_px: int = 2,
    min_ink_delta_ratio: float = 0.010,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Slide a fixed-size window around init_box and choose the location that maximizes
    scan-vs-template delta ink. Useful for tiny handwritten fields where edge/template
    matching is unstable.
    """
    H, W = scan_gray.shape[:2]
    box = clamp_box(init_box_xyxy, W, H)
    x0, y0, x1, y1 = box
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)

    sx0 = max(0, x0 - search_px)
    sy0 = max(0, y0 - search_px)
    sx1 = min(W, x1 + search_px)
    sy1 = min(H, y1 + search_px)

    if (sx1 - sx0) < bw or (sy1 - sy0) < bh:
        return box, {"ok": False, "reason": "search_window_too_small"}

    sg = normalize_for_ocr(scan_gray)
    tg = normalize_for_ocr(templ_gray)
    diff = cv2.absdiff(sg, tg)
    diff = cv2.GaussianBlur(diff, (3, 3), 0)

    # dynamic threshold for faint ink while rejecting low-level scanner noise
    q = float(np.percentile(diff, 85))
    thr = int(max(10, min(55, q)))
    ink = (diff >= thr).astype(np.uint8)

    ii = cv2.integral(ink)

    best_score = -1.0
    best_xy = (x0, y0)
    for yy in range(sy0, sy1 - bh + 1, max(1, step_px)):
        for xx in range(sx0, sx1 - bw + 1, max(1, step_px)):
            s = _window_sum_from_integral(ii, xx, yy, xx + bw, yy + bh)
            if s > best_score:
                best_score = s
                best_xy = (xx, yy)

    ink_ratio = float(best_score / max(1.0, bw * bh))
    diag = {
        "ok": ink_ratio >= min_ink_delta_ratio,
        "ink_ratio": ink_ratio,
        "min_ink_delta_ratio": float(min_ink_delta_ratio),
        "search_px": int(search_px),
        "step_px": int(step_px),
        "threshold": int(thr),
    }

    if ink_ratio < min_ink_delta_ratio:
        return box, diag

    nx0, ny0 = best_xy
    refined = clamp_box([nx0, ny0, nx0 + bw, ny0 + bh], W, H)
    diag["dx"] = int(nx0 - x0)
    diag["dy"] = int(ny0 - y0)
    diag["refined_box_xyxy"] = refined
    return refined, diag

def refine_box_by_template_match(
    scan_gray: np.ndarray,
    templ_gray: np.ndarray,
    init_box_xyxy: List[int],
    search_px: int = 80,
    min_score: float = 0.18,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Refine a box by matching the template patch inside a local search window on the warped scan.
    Helps compensate for local residual offsets (e.g., left/down drift) after global alignment.
    """
    H, W = scan_gray.shape[:2]
    box = clamp_box(init_box_xyxy, W, H)
    x0, y0, x1, y1 = box

    templ_patch = crop_gray(templ_gray, box, pad=0)
    if templ_patch.size == 0:
        return box, {"ok": False, "reason": "empty_template_patch"}

    sx0 = max(0, x0 - search_px)
    sy0 = max(0, y0 - search_px)
    sx1 = min(W, x1 + search_px)
    sy1 = min(H, y1 + search_px)
    scan_search = scan_gray[sy0:sy1, sx0:sx1]

    th, tw = templ_patch.shape[:2]
    sh, sw = scan_search.shape[:2]
    if sh < th or sw < tw:
        return box, {"ok": False, "reason": "search_smaller_than_template"}

    tp = cv2.Canny(normalize_for_ocr(templ_patch), 60, 140)
    sp = cv2.Canny(normalize_for_ocr(scan_search), 60, 140)

    try:
        res = cv2.matchTemplate(sp, tp, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
    except Exception as e:
        return box, {"ok": False, "reason": f"match_error:{e}"}

    new_x0 = sx0 + int(max_loc[0])
    new_y0 = sy0 + int(max_loc[1])
    dx = int(new_x0 - x0)
    dy = int(new_y0 - y0)

    diag = {
        "ok": bool(max_val >= min_score),
        "score": float(max_val),
        "dx": dx,
        "dy": dy,
        "search_px": int(search_px),
        "min_score": float(min_score),
    }
    if max_val < min_score:
        return box, diag

    refined = shift_box(box, dx, dy, W, H)
    diag["refined_box_xyxy"] = refined
    return refined, diag

def _collect_tesseract_word_boxes(
    gray: np.ndarray,
    psm: int,
    min_conf: float = 20.0,
) -> List[Tuple[int, int, int, int]]:
    """
    Collect confident word-level boxes from tesseract image_to_data.
    Returns boxes in (x0, y0, x1, y1) image coordinates.
    """
    cfg = f"--oem 1 --psm {int(psm)}"
    boxes: List[Tuple[int, int, int, int]] = []
    try:
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=cfg)
    except Exception:
        return boxes

    n = len(data.get("text", []))
    for i in range(n):
        txt = (data.get("text", [""] * n)[i] or "").strip()
        if not txt:
            continue
        try:
            conf = float(data.get("conf", ["-1"] * n)[i])
        except Exception:
            conf = -1.0
        if conf < float(min_conf):
            continue

        x = int(data.get("left", [0] * n)[i])
        y = int(data.get("top", [0] * n)[i])
        w = int(data.get("width", [0] * n)[i])
        h = int(data.get("height", [0] * n)[i])
        if w <= 1 or h <= 1:
            continue
        boxes.append((x, y, x + w, y + h))

    return boxes

def refine_box_by_tesseract_detector(
    scan_rgb: np.ndarray,
    templ_rgb: np.ndarray,
    init_box_xyxy: List[int],
    page_W: int,
    page_H: int,
    pad_px: int = 6,
    min_conf: float = 20.0,
    min_union_area_px: int = 180,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Model-driven text region refinement using tesseract detections on the
    subtraction-derived ink crop. This helps align handwriting crops when
    deterministic morphology drifts.
    """
    box = clamp_box(init_box_xyxy, page_W, page_H)
    crop_scan = crop_px(scan_rgb, box, pad=pad_px)
    crop_templ = crop_px(templ_rgb, box, pad=pad_px)

    if crop_scan.size == 0 or crop_templ.size == 0:
        return box, {"ok": False, "reason": "empty_crop"}

    ink_rgb, _, _ = extract_ink_only(crop_scan, crop_templ)
    ink_gray = cv2.cvtColor(ink_rgb, cv2.COLOR_RGB2GRAY)
    ink_gray = remove_lines_inpaint(ink_gray)
    ink_gray = normalize_for_ocr(ink_gray)

    boxes = []
    for psm in (6, 11):
        boxes.extend(_collect_tesseract_word_boxes(ink_gray, psm=psm, min_conf=min_conf))

    diag = {
        "ok": False,
        "detector": "tesseract_word_boxes",
        "word_boxes_found": len(boxes),
        "min_conf": float(min_conf),
    }
    if not boxes:
        return box, diag

    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)

    union_area = int(max(0, x1 - x0) * max(0, y1 - y0))
    diag["union_area_px"] = union_area
    if union_area < int(min_union_area_px):
        diag["reason"] = "union_too_small"
        return box, diag

    x0 = max(0, x0 - 8)
    y0 = max(0, y0 - 6)
    x1 = min(crop_scan.shape[1], x1 + 8)
    y1 = min(crop_scan.shape[0], y1 + 6)

    bx0, by0, _, _ = box
    ox = max(0, bx0 - pad_px)
    oy = max(0, by0 - pad_px)

    refined = clamp_box([ox + x0, oy + y0, ox + x1, oy + y1], page_W, page_H)

    diag["ok"] = True
    diag["refined_box_xyxy"] = refined
    return refined, diag


class CraftTextRegionRefiner:
    """
    Optional model-assisted detector for text localization inside a field crop.

    Uses craft_text_detector if available; gracefully degrades when dependency or
    model weights are unavailable.
    """

    def __init__(self):
        self._craft = None
        self._init_error: Optional[str] = None
        self._load_once()

    def _load_once(self) -> None:
        if self._craft is not None or self._init_error is not None:
            return
        try:
            from craft_text_detector import Craft  # type: ignore
            self._craft = Craft(
                output_dir=None,
                crop_type="box",
                cuda=torch.cuda.is_available(),
            )
        except Exception as e:
            self._init_error = str(e)
            self._craft = None

    def available(self) -> bool:
        return self._craft is not None

    def diagnose(self) -> Dict[str, Any]:
        return {
            "available": self.available(),
            "error": self._init_error,
        }

    def detect_boxes(self, rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self._craft is None:
            return []

        try:
            pred = self._craft.detect_text(image=rgb)
        except Exception:
            return []

        polys = pred.get("boxes") or pred.get("polys") or []
        out: List[Tuple[int, int, int, int]] = []
        h, w = rgb.shape[:2]
        for p in polys:
            try:
                a = np.asarray(p, dtype=np.float32).reshape(-1, 2)
                if a.shape[0] < 4:
                    continue
                x0 = int(max(0, np.floor(np.min(a[:, 0]))))
                y0 = int(max(0, np.floor(np.min(a[:, 1]))))
                x1 = int(min(w, np.ceil(np.max(a[:, 0]))))
                y1 = int(min(h, np.ceil(np.max(a[:, 1]))))
                if (x1 - x0) <= 1 or (y1 - y0) <= 1:
                    continue
                out.append((x0, y0, x1, y1))
            except Exception:
                continue
        return out


def refine_box_by_craft_detector(
    scan_rgb: np.ndarray,
    templ_rgb: np.ndarray,
    init_box_xyxy: List[int],
    page_W: int,
    page_H: int,
    craft_refiner: Optional[CraftTextRegionRefiner],
    pad_px: int = 6,
    min_union_area_px: int = 120,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Model-assisted field refinement using CRAFT text detector over subtraction-derived
    ink crop. Especially useful for irregular handwriting where deterministic
    projection/morphology can drift.
    """
    box = clamp_box(init_box_xyxy, page_W, page_H)
    diag: Dict[str, Any] = {
        "ok": False,
        "detector": "craft_mlt_25k",
    }

    if craft_refiner is None:
        diag["reason"] = "craft_refiner_not_set"
        return box, diag
    if not craft_refiner.available():
        diag["reason"] = "craft_unavailable"
        diag["craft_diag"] = craft_refiner.diagnose()
        return box, diag

    crop_scan = crop_px(scan_rgb, box, pad=pad_px)
    crop_templ = crop_px(templ_rgb, box, pad=pad_px)
    if crop_scan.size == 0 or crop_templ.size == 0:
        diag["reason"] = "empty_crop"
        return box, diag

    ink_rgb, _, _ = extract_ink_only(crop_scan, crop_templ)
    ink_gray = cv2.cvtColor(ink_rgb, cv2.COLOR_RGB2GRAY)
    ink_gray = remove_lines_inpaint(ink_gray)
    ink_rgb = cv2.cvtColor(ink_gray, cv2.COLOR_GRAY2RGB)

    boxes = craft_refiner.detect_boxes(ink_rgb)
    diag["word_boxes_found"] = len(boxes)
    if not boxes:
        diag["reason"] = "no_boxes"
        return box, diag

    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)

    union_area = int(max(0, x1 - x0) * max(0, y1 - y0))
    diag["union_area_px"] = union_area
    if union_area < int(min_union_area_px):
        diag["reason"] = "union_too_small"
        return box, diag

    x0 = max(0, x0 - 8)
    y0 = max(0, y0 - 6)
    x1 = min(crop_scan.shape[1], x1 + 8)
    y1 = min(crop_scan.shape[0], y1 + 6)

    bx0, by0, _, _ = box
    ox = max(0, bx0 - pad_px)
    oy = max(0, by0 - pad_px)
    refined = clamp_box([ox + x0, oy + y0, ox + x1, oy + y1], page_W, page_H)

    diag["ok"] = True
    diag["refined_box_xyxy"] = refined
    return refined, diag


# -------------------------
# Value normalization
# -------------------------

def normalize_value(raw: str, value_type: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    vt = (value_type or "text").lower()

    if vt == "rating_1_5":
        m = re.search(r"[1-5]", s)
        return m.group(0) if m else s

    if vt == "yes_no":
        u = s.upper().replace(" ", "")
        if "N/A" in u or u == "NA":
            return "N/A"
        if u.startswith("Y") or "YES" in u:
            return "YES"
        if u.startswith("N") or "NO" in u:
            return "NO"
        return s

    return s


# -------------------------
# Template cache
# -------------------------

@dataclass
class TemplateCache:
    form_id: str
    template_pdf_path: str
    dpi: int
    template_pages_rgb: List[np.ndarray]
    template_pages_gray: List[np.ndarray]

def load_template_cache(dbutils, form_cfg: Dict[str, Any], local_dir: str = "/dbfs/tmp/ocr_templates") -> TemplateCache:
    form_id = form_cfg.get("form_id", "unknown_form")
    template_pdf_path = form_cfg["template_pdf"]
    dpi = int(form_cfg.get("dpi", 300))

    if is_s3_path(template_pdf_path) or is_dbfs_uri(template_pdf_path):
        local_pdf = copy_to_local_file(dbutils, template_pdf_path, local_dir)
    else:
        local_pdf = resolve_local_or_relative_path(dbutils, template_pdf_path)

    template_pil_pages = render_pdf_pages(local_pdf, dpi=dpi, max_pages=None)
    rgb_pages = [pil_to_rgb_np(p) for p in template_pil_pages]
    gray_pages = [to_gray(p) for p in rgb_pages]

    return TemplateCache(
        form_id=form_id,
        template_pdf_path=template_pdf_path,
        dpi=dpi,
        template_pages_rgb=rgb_pages,
        template_pages_gray=gray_pages,
    )


# -------------------------
# Main per-PDF pipeline
# -------------------------

def has_working_tesseract_cached() -> bool:
    return has_working_tesseract()

def ocr_pdf_with_form_config(
    dbutils,
    pdf_path: str,
    device: str,
    trocr_printed: TrocrBundle,
    trocr_handwritten: TrocrBundle,
    form_cfg: Dict[str, Any],
    template_cache: TemplateCache,
    max_pages: Optional[int] = None,
    field_pad_px: int = 6,

    # Optional model-assisted detector
    craft_refiner: Optional[CraftTextRegionRefiner] = None,

    # NEW: Better crop tightening knobs
    enable_scan_line_refine: bool = True,
    enable_yproj_tighten: bool = True,
    enable_component_tighten: bool = True,
    enable_local_template_match_refine: bool = True,
    local_match_search_px: int = 90,
    local_match_min_score: float = 0.20,
    enable_ink_delta_refine: bool = True,
    ink_delta_search_px: int = 36,
    ink_delta_step_px: int = 2,
    ink_delta_min_ratio: float = 0.010,
    enable_tesseract_detector_refine: bool = True,
    tesseract_detector_min_conf: float = 20.0,
    tesseract_detector_min_union_area_px: int = 180,
    empty_ink_thresh: float = 0.0005,  # if ink coverage under this => treat as blank

    debug_dir: Optional[str] = None,
    debug_save_limit: int = 4000,
) -> Dict[str, Any]:
    t0 = time.time()
    tess_ok = has_working_tesseract_cached()

    local_pdf = copy_to_local_file(dbutils, pdf_path, "/dbfs/tmp/ocr_pdfs")
    embedded = extract_embedded_text(local_pdf)
    if embedded is not None:
        return {
            "pdf_path": pdf_path,
            "form_id": form_cfg.get("form_id"),
            "mode": "embedded_text",
            "runtime_sec": round(time.time() - t0, 3),
            "text": embedded,
            "fields": {},
            "pages": [],
        }

    dpi = int(form_cfg.get("dpi", template_cache.dpi))
    scanned_pil_pages = render_pdf_pages(local_pdf, dpi=dpi, max_pages=max_pages)

    out_pages = []
    all_fields: Dict[str, Any] = {}

    pdf_stem = re.sub(r"\.pdf$", "", os.path.basename(pdf_path), flags=re.IGNORECASE)
    debug_saved = 0

    for page_cfg in form_cfg.get("pages", []):
        pi = int(page_cfg.get("page_index", 0))
        if pi < 0 or pi >= len(scanned_pil_pages):
            continue
        if pi >= len(template_cache.template_pages_gray):
            continue

        scan_rgb = pil_to_rgb_np(scanned_pil_pages[pi])
        scan_gray = to_gray(scan_rgb)

        templ_gray = template_cache.template_pages_gray[pi]
        templ_rgb_page = template_cache.template_pages_rgb[pi]

        H, metrics = estimate_homography_orb(scan_gray, templ_gray)
        if H is None:
            H, metrics2 = estimate_affine_ecc(scan_gray, templ_gray)
            metrics = {"primary": metrics, "fallback": metrics2}

        page_result = {
            "page_index": pi,
            "alignment": {"ok": H is not None, **metrics},
            "anchors": [],
            "fields": []
        }

        if H is None:
            best = pick_best_handwriting_ocr(trocr_printed, trocr_handwritten, scan_rgb, device=device, max_new_tokens=256)
            page_result["fallback_text"] = best["text"]
            page_result["fallback_confidence"] = float(best["confidence"])
            page_result["fallback_engine"] = best["engine"]
            out_pages.append(page_result)
            continue

        Ht, Wt = templ_gray.shape[:2]
        warped_rgb = warp_scan_to_template(scan_rgb, templ_shape_hw=(Ht, Wt), H=H)
        warped_gray = to_gray(warped_rgb)

        # ---- anchors ----
        anchors_px: Dict[str, List[int]] = {}
        for a in page_cfg.get("anchors", []):
            aid = a["id"]
            box = bbox_norm_to_px(a["bbox_norm"], Wt, Ht)
            anchors_px[aid] = box

            a_crop = crop_px(warped_rgb, box, pad=field_pad_px)
            exp = (a.get("expected_text") or "").strip()

            if tess_ok:
                psm = int(a.get("tesseract_psm", 7))
                txt, conf = tesseract_ocr_with_conf(a_crop, psm=psm)
                engine = "tesseract"
            else:
                txt, conf = trocr_predict_with_conf(trocr_printed, rgb_np_to_pil(a_crop), device=device, max_new_tokens=64)
                engine = trocr_printed.name

            score = float(fuzz.ratio(txt.lower(), exp.lower())) if exp else None

            page_result["anchors"].append({
                "id": aid,
                "bbox_xyxy": box,
                "text": txt,
                "confidence": float(conf),
                "expected_text": exp if exp else None,
                "match_score_0_100": score,
                "engine": engine
            })

        # ---- fields ----
        for f in page_cfg.get("fields", []):
            fid = f["id"]
            ftype = (f.get("type", "handwriting") or "handwriting").lower()
            value_type = (f.get("value_type", "text") or "text").lower()
            max_new = int(f.get("max_new_tokens", 64))
            pad = int(f.get("pad_px", field_pad_px))

            init_box = resolve_field_bbox_px(f, page_W=Wt, page_H=Ht, anchors_px=anchors_px)
            init_box = clamp_box(init_box, Wt, Ht)

            refine_steps = []

            # Step A: if field is free-text or handwriting, tighten using scan-detected lines (works even when subtraction fails)
            refined_box = init_box
            # Step A0: local template match refinement (captures residual local drift on bad scans)
            if enable_local_template_match_refine:
                match_search_px = int(f.get("local_match_search_px", local_match_search_px))
                match_min_score = float(f.get("local_match_min_score", local_match_min_score))
                refined_box, diag0 = refine_box_by_template_match(
                    scan_gray=warped_gray,
                    templ_gray=templ_gray,
                    init_box_xyxy=refined_box,
                    search_px=match_search_px,
                    min_score=match_min_score,
                )
                refine_steps.append({"local_template_match_refine": diag0})

            if enable_ink_delta_refine and value_type in ("rating_1_5", "yes_no", "free_text", "text"):
                delta_search_px = int(f.get("ink_delta_search_px", ink_delta_search_px))
                delta_step_px = int(f.get("ink_delta_step_px", ink_delta_step_px))
                delta_min_ratio = float(f.get("ink_delta_min_ratio", ink_delta_min_ratio))
                refined_box, diag0b = refine_box_by_ink_delta_peak(
                    scan_gray=warped_gray,
                    templ_gray=templ_gray,
                    init_box_xyxy=refined_box,
                    search_px=delta_search_px,
                    step_px=max(1, delta_step_px),
                    min_ink_delta_ratio=delta_min_ratio,
                )
                refine_steps.append({"ink_delta_refine": diag0b})

            if enable_tesseract_detector_refine and tess_ok and ftype == "handwriting":
                refined_box, diag_tm = refine_box_by_tesseract_detector(
                    scan_rgb=warped_rgb,
                    templ_rgb=templ_rgb_page,
                    init_box_xyxy=refined_box,
                    page_W=Wt,
                    page_H=Ht,
                    pad_px=pad,
                    min_conf=float(f.get("tesseract_detector_min_conf", tesseract_detector_min_conf)),
                    min_union_area_px=int(f.get("tesseract_detector_min_union_area_px", tesseract_detector_min_union_area_px)),
                )
                refine_steps.append({"tesseract_detector_refine": diag_tm})

            if ftype == "handwriting":
                refined_box, diag_craft = refine_box_by_craft_detector(
                    scan_rgb=warped_rgb,
                    templ_rgb=templ_rgb_page,
                    init_box_xyxy=refined_box,
                    page_W=Wt,
                    page_H=Ht,
                    craft_refiner=craft_refiner,
                    pad_px=pad,
                    min_union_area_px=int(f.get("craft_detector_min_union_area_px", 120)),
                )
                refine_steps.append({"craft_detector_refine": diag_craft})

            if enable_scan_line_refine and value_type == "free_text":
                max_lines = int(f.get("scan_line_max_lines", 1))
                above_pad = int(f.get("scan_line_above_pad_px", 70))
                below_pad = int(f.get("scan_line_below_pad_px", 45))
                refined_box, diagA = refine_ruled_box_using_scan_lines(
                    warped_gray=warped_gray,
                    init_box_xyxy=refined_box,
                    pad_px=pad,
                    max_lines=max_lines,
                    min_line_length_frac=float(f.get("scan_line_min_line_length_frac", 0.30)),
                    above_pad=above_pad,
                    below_pad=below_pad,
                    x_expand=int(f.get("scan_line_x_expand_px", 15)),
                )
                refine_steps.append({"scan_line_refine": diagA})

            # Step B: extract ink using LOCAL-aligned subtraction inside refined_box
            crop_scan  = crop_px(warped_rgb, refined_box, pad=pad)
            crop_templ = crop_px(templ_rgb_page, refined_box, pad=pad)
            ink_rgb, ink_mask, ink_diag = extract_ink_only(crop_scan, crop_templ)

            # compute coverage
            ink_pixels = int(np.count_nonzero(ink_mask))
            coverage = float(ink_pixels / max(1, ink_mask.size))

            # Optional: y-projection tighten to focus on handwriting band (drops prompt above)
            if enable_yproj_tighten and value_type == "free_text" and coverage > empty_ink_thresh:
                band = y_band_from_mask(ink_mask, min_band_h=int(f.get("yproj_min_band_h", 45)))
                if band is not None:
                    y0b, y1b, ydiag = band
                    # map y band back to page coords
                    x0p, y0p, x1p, y1p = refined_box
                    oy = max(0, y0p - pad)
                    y0_page = oy + y0b
                    y1_page = oy + y1b
                    refined_box2 = clamp_box([x0p, y0_page, x1p, y1_page], Wt, Ht)
                    refined_box = refined_box2
                    refine_steps.append({"yproj_tighten": ydiag})

                    # re-crop after y-tighten
                    crop_scan  = crop_px(warped_rgb, refined_box, pad=pad)
                    crop_templ = crop_px(templ_rgb_page, refined_box, pad=pad)
                    ink_rgb, ink_mask, ink_diag = extract_ink_only(crop_scan, crop_templ)
                    ink_pixels = int(np.count_nonzero(ink_mask))
                    coverage = float(ink_pixels / max(1, ink_mask.size))

            # Optional: component tighten x-range (reduces giant left/right spans)
            if enable_component_tighten and coverage > empty_ink_thresh:
                cb = components_bbox(ink_mask, keep_k=3, min_area=40)
                if cb is not None:
                    x0c, y0c, x1c, y1c = cb
                    # expand a bit
                    x0c = max(0, x0c - 10); x1c = min(ink_mask.shape[1], x1c + 10)
                    # map to page coords
                    x0p, y0p, x1p, y1p = refined_box
                    ox = max(0, x0p - pad)
                    new_x0 = ox + x0c
                    new_x1 = ox + x1c
                    refined_box = clamp_box([new_x0, y0p, new_x1, y1p], Wt, Ht)
                    refine_steps.append({"component_tighten": {"cb": cb}})

                    crop_scan  = crop_px(warped_rgb, refined_box, pad=pad)
                    crop_templ = crop_px(templ_rgb_page, refined_box, pad=pad)
                    ink_rgb, ink_mask, ink_diag = extract_ink_only(crop_scan, crop_templ)
                    ink_pixels = int(np.count_nonzero(ink_mask))
                    coverage = float(ink_pixels / max(1, ink_mask.size))

            # Remove lines on the final crop to help TrOCR
            final_gray = cv2.cvtColor(ink_rgb, cv2.COLOR_RGB2GRAY)
            final_gray = remove_lines_inpaint(final_gray)
            final_rgb = cv2.cvtColor(final_gray, cv2.COLOR_GRAY2RGB)

            # Debug saves (init vs refined vs ink)
            if debug_dir and debug_saved < debug_save_limit:
                try:
                    d0 = join_path(debug_dir, f"{pdf_stem}/p{pi:02d}")
                    ensure_dir_dbutils(dbutils, d0)

                    init_scan = crop_px(warped_rgb, init_box, pad=pad)
                    init_templ = crop_px(templ_rgb_page, init_box, pad=pad)
                    init_ink, init_mask, init_diag = extract_ink_only(init_scan, init_templ)

                    write_rgb_image_to_output(dbutils, init_ink,  join_path(d0, f"{fid}__INIT_ink.png"))
                    write_rgb_image_to_output(dbutils, ink_rgb,   join_path(d0, f"{fid}__REF_ink.png"))
                    write_rgb_image_to_output(dbutils, final_rgb, join_path(d0, f"{fid}__REF_ink_nolines.png"))

                    diag_out = {
                        "field_id": fid,
                        "page_index": pi,
                        "init_box_xyxy": init_box,
                        "refined_box_xyxy": refined_box,
                        "ink_cov": coverage,
                        "ink_diag": ink_diag,
                        "refine_steps": refine_steps
                    }
                    write_json_to_output(dbutils, diag_out, join_path(d0, f"{fid}__DIAG.json"))

                    debug_saved += 4
                except Exception:
                    pass

            # Empty handling
            if coverage < empty_ink_thresh:
                raw_txt = ""
                raw_conf = 0.0
                engine = "empty_like"
                value = ""
            else:
                raw_txt = ""
                raw_conf = 0.0
                engine = ""

                if value_type == "rating_1_5" and tess_ok:
                    r = ocr_rating_1_5_tesseract(final_rgb)
                    if r:
                        raw_txt, raw_conf, engine = r, 0.99, "tesseract_rating_whitelist"
                    else:
                        best = pick_best_handwriting_ocr(trocr_printed, trocr_handwritten, final_rgb, device=device, max_new_tokens=8)
                        raw_txt, raw_conf, engine = best["text"], best["confidence"], best["engine"]

                elif value_type == "yes_no" and tess_ok:
                    yn = ocr_yes_no_tesseract(final_rgb)
                    if yn:
                        raw_txt, raw_conf, engine = yn, 0.98, "tesseract_yesno_whitelist"
                    else:
                        best = pick_best_handwriting_ocr(trocr_printed, trocr_handwritten, final_rgb, device=device, max_new_tokens=8)
                        raw_txt, raw_conf, engine = best["text"], best["confidence"], best["engine"]

                elif ftype == "printed":
                    candidates = []
                    if tess_ok:
                        psm = int(f.get("tesseract_psm", 7))
                        t_txt, t_conf = tesseract_ocr_with_conf(crop_px(warped_rgb, refined_box, pad=pad), psm=psm)
                        candidates.append(("tesseract", t_txt, float(t_conf)))
                    p_txt, p_conf = trocr_predict_with_conf(
                        trocr_printed, rgb_np_to_pil(crop_px(warped_rgb, refined_box, pad=pad)), device=device, max_new_tokens=max_new
                    )
                    candidates.append((trocr_printed.name, p_txt, float(p_conf)))
                    engine, raw_txt, raw_conf = pick_best_by_confidence(candidates)

                else:
                    best = pick_best_handwriting_ocr(
                        trocr_printed, trocr_handwritten, final_rgb, device=device, max_new_tokens=max_new
                    )
                    raw_txt, raw_conf, engine = best["text"], best["confidence"], best["engine"]

                value = normalize_value(raw_txt, value_type)

            field_obj = {
                "id": fid,
                "type": ftype,
                "value_type": value_type,
                "bbox_xyxy_init": init_box,
                "bbox_xyxy": refined_box,
                "ink_coverage": coverage,
                "ink_diag": ink_diag,
                "refine_steps": refine_steps,
                "raw_text": raw_txt,
                "text": value,
                "confidence": float(raw_conf),
                "engine": engine,
            }

            page_result["fields"].append(field_obj)

            all_fields[fid] = {
                "page_index": pi,
                "bbox_xyxy_init": init_box,
                "bbox_xyxy": refined_box,
                "ink_coverage": coverage,
                "text": value,
                "raw_text": raw_txt,
                "confidence": float(raw_conf),
                "engine": engine,
                "value_type": value_type,
                "refine_steps": refine_steps,
                "ink_diag": ink_diag,
            }

        out_pages.append(page_result)

    return {
        "pdf_path": pdf_path,
        "form_id": form_cfg.get("form_id"),
        "mode": "template_config_ocr_v3_localalign_scanlines_yproj",
        "dpi": int(form_cfg.get("dpi", template_cache.dpi)),
        "runtime_sec": round(time.time() - t0, 3),
        "fields": all_fields,
        "pages": out_pages,
    }
