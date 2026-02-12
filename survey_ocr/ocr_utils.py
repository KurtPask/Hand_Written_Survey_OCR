

from __future__ import annotations

import os
import re
import json
import math
import time
import glob
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# --- Optional dependencies ---
try:
    import fitz  # PyMuPDF
except Exception as e:
    raise RuntimeError("PyMuPDF (fitz) is required. Install: pip install pymupdf") from e

try:
    import cv2
except Exception as e:
    raise RuntimeError("OpenCV is required. Install: pip install opencv-python-headless") from e

try:
    from PIL import Image
except Exception as e:
    raise RuntimeError("Pillow is required. Install: pip install pillow") from e

# Transformers for TrOCR
try:
    import torch
    from transformers import VisionEncoderDecoderModel, TrOCRProcessor
except Exception as e:
    raise RuntimeError("transformers + torch + sentencepiece required. Install: pip install transformers torch sentencepiece") from e


# =========================
#        CONFIG
# =========================

@dataclass
class PipelineConfig:
    # ---- Paths you MUST adjust ----
    model_root: str
    trocr_printed_path: str
    trocr_handwritten_path: str

    # EasyOCR weights directory: set if you need offline/local weights.
    # If None, EasyOCR will use its default cache (may try to download).
    easyocr_model_dir: Optional[str] = None

    # ---- I/O ----
    input_path: str = ""
    output_dir: str = ""
    recursive: bool = True

    # ---- Rendering ----
    render_dpi: int = 250        # 200–300 is a good range for OCR
    max_pages: Optional[int] = None  # e.g., 5 for testing
    embedded_text_min_chars: int = 30  # If embedded text >= this, skip OCR

    # ---- Preprocess ----
    do_preprocess: bool = True
    deskew: bool = True
    clahe: bool = True
    denoise: bool = True
    binarize: bool = False

    # ---- Detection / recognition policy ----
    # If EasyOCR is installed, we use it. If not, we fall back to "whole-page TrOCR" (worse).
    enable_easyocr: bool = True
    require_easyocr: bool = False
    easyocr_langs: Tuple[str, ...] = ("en",)

    # Keep EasyOCR output if confidence high and text is "not gibberish"
    easyocr_keep_conf: float = 0.65

    # TrOCR fallback thresholds
    trocr_printed_min_conf: float = 0.35  # if printed conf below this, try handwritten
    trocr_choose_best: bool = True         # choose higher-confidence of printed vs handwritten

    # Performance
    max_regions_per_page: int = 250   # safety valve
    min_region_area_px: int = 300
    min_region_side_px: int = 12
    crop_pad_px: int = 6
    trocr_batch_size: int = 8
    torch_num_threads: int = 0        # 0 = leave default; else set e.g. 8

    # Output behavior
    write_jsonl: bool = True
    write_page_text: bool = True
    write_delta_if_spark: bool = False
    output_delta_path: Optional[str] = None  # e.g., "dbfs:/FileStore/ocr/delta/output"

# =========================
#       UTILITIES
# =========================

def dbfs_to_local(path: str) -> str:
    """
    Accepts dbfs:/... or /dbfs/... or local absolute/relative paths.
    Returns a local filesystem path usable by Python libs.
    """
    if path.startswith("dbfs:/"):
        return "/dbfs/" + path[len("dbfs:/"):].lstrip("/")
    return path

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def now_ms() -> int:
    return int(time.time() * 1000)

def looks_like_gibberish(text: str) -> bool:
    """
    Lightweight heuristic to reject garbage OCR:
    - too short
    - low alnum ratio
    - mostly punctuation
    """
    t = (text or "").strip()
    if len(t) < 2:
        return True
    # If it's mostly non-alphanumeric, likely junk
    alnum = sum(ch.isalnum() for ch in t)
    ratio = alnum / max(1, len(t))
    if ratio < 0.30 and len(t) > 4:
        return True
    # Lots of repeating junk
    if re.fullmatch(r"[_\-\.\,\;\:\!\?\(\)\[\]\{\}\s]+", t):
        return True
    return False

def safe_filename(s: str) -> str:
    s = os.path.basename(s)
    s = re.sub(r"[^a-zA-Z0-9\-\._]+", "_", s)
    return s[:200]

def sort_boxes_reading_order(items: List[Dict[str, Any]], y_tol: int = 12) -> List[Dict[str, Any]]:
    """
    Approximate reading order: group by line (y center) then sort by x.
    Works well for single-column docs; for complex layouts consider Docling/LayoutParser.
    """
    if not items:
        return items
    # Compute center points
    for it in items:
        x0, y0, x1, y1 = it["bbox_xyxy"]
        it["_cx"] = (x0 + x1) / 2.0
        it["_cy"] = (y0 + y1) / 2.0

    items = sorted(items, key=lambda d: (d["_cy"], d["_cx"]))

    # Merge into lines
    lines: List[List[Dict[str, Any]]] = []
    for it in items:
        if not lines:
            lines.append([it])
            continue
        if abs(it["_cy"] - lines[-1][-1]["_cy"]) <= y_tol:
            lines[-1].append(it)
        else:
            lines.append([it])

    # Sort each line left-to-right
    out: List[Dict[str, Any]] = []
    for line in lines:
        line_sorted = sorted(line, key=lambda d: d["_cx"])
        out.extend(line_sorted)

    # cleanup
    for it in out:
        it.pop("_cx", None)
        it.pop("_cy", None)
    return out

# =========================
#    IMAGE PREPROCESSING
# =========================

def preprocess_page_image_bgr(img_bgr: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """
    Normalize a rendered page image to improve OCR:
    - grayscale
    - denoise
    - CLAHE
    - binarize
    - deskew (rotate)
    Output: 3-channel BGR image (for downstream OCR libs)
    """
    if not cfg.do_preprocess:
        return img_bgr

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if cfg.denoise:
        # fast denoise
        gray = cv2.fastNlMeansDenoising(gray, h=10)

    if cfg.clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    if cfg.binarize:
        # adaptive threshold helps uneven illumination
        thr = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            35, 11
        )
    else:
        thr = gray

    if cfg.deskew:
        thr = deskew_binary(thr)

    # Return a 3-channel image
    if thr.ndim == 2:
        out = cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)
    else:
        out = thr
    return out

def deskew_binary(bin_img: np.ndarray) -> np.ndarray:
    """
    Deskew based on minAreaRect over foreground pixels.
    Assumes bin_img is 0/255-ish. We treat "ink" as dark pixels.
    """
    img = bin_img.copy()
    if img.ndim != 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Build text mask with Otsu thresholding so we use only likely ink pixels.
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # np.where returns (y, x); minAreaRect expects points in (x, y).
    ys, xs = np.where(mask > 0)
    coords = np.column_stack((xs, ys))
    if coords.size < 500:
        return bin_img  # not enough signal

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    # angle is in [-90, 0)
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.3:
        return bin_img

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(bin_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# =========================
#     PDF RENDERING
# =========================

def render_pdf_page_to_bgr(page: fitz.Page, dpi: int) -> np.ndarray:
    """
    Render PDF page to BGR image using PyMuPDF.
    """
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 3:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif pix.n == 4:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return bgr

# =========================
#     OCR ENGINES
# =========================

class EasyOCREngine:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.reader = None

        if not cfg.enable_easyocr:
            print("[INFO] EasyOCR explicitly disabled by config.")
            return

        try:
            import easyocr  # type: ignore
        except Exception:
            # EasyOCR not installed
            self.reader = None
            msg = "[WARN] EasyOCR import failed. Falling back to TrOCR-only path."
            if cfg.require_easyocr:
                raise RuntimeError(msg + " Set require_easyocr=False to allow fallback.")
            print(msg)
            return

        kwargs = {
            "lang_list": list(cfg.easyocr_langs),
            "gpu": False,
        }
        # Force local weights if you have them (recommended for enterprise offline)
        if cfg.easyocr_model_dir:
            kwargs["model_storage_directory"] = dbfs_to_local(cfg.easyocr_model_dir)
            kwargs["user_network_directory"] = dbfs_to_local(cfg.easyocr_model_dir)

        self.reader = easyocr.Reader(**kwargs)

    def available(self) -> bool:
        return self.reader is not None

    def readtext(self, img_bgr: np.ndarray) -> List[Tuple[List[List[float]], str, float]]:
        """
        Returns list of (bbox_points, text, confidence).
        bbox_points: 4 points [[x,y], [x,y], [x,y], [x,y]]
        """
        assert self.reader is not None
        # detail=1 returns bounding box + text + conf
        # paragraph=False keeps line-level boxes
        return self.reader.readtext(img_bgr, detail=1, paragraph=False)


def detect_text_regions_cv(img_bgr: np.ndarray, cfg: PipelineConfig) -> List[Tuple[int, int, int, int]]:
    """
    Lightweight text region proposal for environments where EasyOCR is unavailable.
    Returns XYXY boxes in approximate reading-order later via sort_boxes_reading_order.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Dark text as foreground.
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        11,
    )
    # Connect characters into word/line blobs.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    merged = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    boxes: List[Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        if area < cfg.min_region_area_px:
            continue
        if ww < cfg.min_region_side_px or hh < cfg.min_region_side_px:
            continue
        if ww > int(0.98 * w) and hh > int(0.98 * h):
            # Ignore contour that is effectively the whole page.
            continue
        boxes.append((x, y, x + ww, y + hh))

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes[: cfg.max_regions_per_page]

class TrOCREngine:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = dbfs_to_local(model_path)
        self.device = device

        self._validate_local_model_dir(self.model_path)

        self.processor = TrOCRProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_path, local_files_only=True)
        if self.model.config.decoder_start_token_id is None and self.processor.tokenizer.cls_token_id is not None:
            self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        if self.model.config.eos_token_id is None:
            self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id
        self.model.to(self.device)
        self.model.eval()

        # CPU-friendly
        torch.set_grad_enabled(False)

    @staticmethod
    def _validate_local_model_dir(model_path: str) -> None:
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"TrOCR model directory does not exist: {model_path}")

        has_weights = any(
            os.path.exists(os.path.join(model_path, fname))
            for fname in ("model.safetensors", "pytorch_model.bin")
        )
        if not has_weights:
            raise FileNotFoundError(
                f"No TrOCR model weights found in {model_path}. "
                "Expected model.safetensors or pytorch_model.bin."
            )

    @staticmethod
    def _compute_token_confidence(generate_outputs) -> float:
        """
        Compute an approximate confidence from per-token probabilities
        using output_scores from generate().
        """
        # outputs.scores: list[tensor(batch, vocab)] for each generated step (excluding prompt)
        # outputs.sequences: tensor(batch, seq_len)
        scores = getattr(generate_outputs, "scores", None)
        seqs = getattr(generate_outputs, "sequences", None)
        if scores is None or seqs is None or len(scores) == 0:
            return float("nan")

        # sequences include special tokens; align chosen tokens with score steps
        # generate outputs correspond to generated tokens (excluding start token)
        # We'll take the chosen token at each step and compute softmax prob.
        # This is approximate but useful for gating.
        probs = []
        # seqs shape: [B, L]
        # scores length: T
        # token indices for steps: take last T tokens from seqs
        chosen = seqs[:, -len(scores):]
        for t, logits in enumerate(scores):
            # logits shape [B, V]
            token_ids = chosen[:, t]
            p = torch.softmax(logits, dim=-1).gather(1, token_ids.unsqueeze(1)).squeeze(1)
            probs.append(p)

        p_all = torch.stack(probs, dim=1)  # [B, T]
        # geometric mean across tokens
        conf = torch.exp(torch.mean(torch.log(torch.clamp(p_all, min=1e-6)), dim=1))
        return conf.detach().cpu().numpy().tolist()

    def recognize_batch(
        self,
        pil_images: List[Image.Image],
        max_new_tokens: int = 64
    ) -> Tuple[List[str], List[float]]:
        if not pil_images:
            return [], []

        pixel_values = self.processor(images=pil_images, return_tensors="pt").pixel_values.to(self.device)

        outputs = self.model.generate(
            pixel_values,
            max_new_tokens=max_new_tokens,
            num_beams=2,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

        texts = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)
        conf = self._compute_token_confidence(outputs)

        # conf may be a list if batch; normalize
        if isinstance(conf, float):
            confs = [conf] * len(texts)
        else:
            confs = conf
        # Replace NaN confs with -1
        confs = [(-1.0 if (c is None or (isinstance(c, float) and math.isnan(c))) else float(c)) for c in confs]
        return texts, confs


@dataclass
class OCRModelBundle:
    """Container for OCR engines so they can be loaded once and reused."""

    easy: EasyOCREngine
    trocr_printed: Optional[TrOCREngine]
    trocr_handwritten: Optional[TrOCREngine]


_MODEL_BUNDLE_CACHE: Dict[str, OCRModelBundle] = {}


def _bundle_cache_key(cfg: PipelineConfig, device: str) -> str:
    return "|".join([
        dbfs_to_local(cfg.trocr_printed_path),
        dbfs_to_local(cfg.trocr_handwritten_path),
        dbfs_to_local(cfg.easyocr_model_dir) if cfg.easyocr_model_dir else "",
        str(cfg.enable_easyocr),
        ",".join(cfg.easyocr_langs),
        device,
    ])


def load_ocr_model_bundle(
    cfg: PipelineConfig,
    device: str = "cpu",
    use_cache: bool = True,
    load_handwritten: bool = True,
) -> OCRModelBundle:
    """
    Load OCR engines once and optionally reuse them via an in-process cache.

    This is useful for Databricks iterative development where model loading
    dominates runtime. Re-running OCR with the same paths can reuse cached
    engines and skip model reload cost.
    """
    cache_key = _bundle_cache_key(cfg, device)
    if use_cache and cache_key in _MODEL_BUNDLE_CACHE:
        return _MODEL_BUNDLE_CACHE[cache_key]

    easy = EasyOCREngine(cfg)
    trocr_printed = TrOCREngine(cfg.trocr_printed_path, device=device)
    trocr_handwritten = TrOCREngine(cfg.trocr_handwritten_path, device=device) if load_handwritten else None

    bundle = OCRModelBundle(
        easy=easy,
        trocr_printed=trocr_printed,
        trocr_handwritten=trocr_handwritten,
    )
    if use_cache:
        _MODEL_BUNDLE_CACHE[cache_key] = bundle
    return bundle


def clear_ocr_model_bundle_cache() -> None:
    """Clear cached OCR model bundles (useful to force model reload)."""
    _MODEL_BUNDLE_CACHE.clear()

# =========================
#     BOX/CROPPING
# =========================

def bbox_points_to_xyxy(pts: List[List[float]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = int(max(0, min(xs))), int(max(xs))
    y0, y1 = int(max(0, min(ys))), int(max(ys))
    return x0, y0, x1, y1

def crop_xyxy(img_bgr: np.ndarray, xyxy: Tuple[int, int, int, int], pad: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = xyxy
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad); y1 = min(h, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return img_bgr[0:1, 0:1].copy()
    return img_bgr[y0:y1, x0:x1].copy()

def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
