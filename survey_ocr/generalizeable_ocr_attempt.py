
from __future__ import annotations

from ocr_utils import *

import os
import re
import json
import math
import time
import glob
import argparse
from typing import Any, Dict, List, Optional

# =========================
#      MAIN PIPELINE
# =========================

def extract_embedded_text(page: fitz.Page) -> str:
    try:
        txt = page.get_text("text") or ""
        return txt.strip()
    except Exception:
        return ""


def should_use_embedded_text(embedded: str, cfg: PipelineConfig) -> bool:
    if not cfg.use_embedded_text_fast_path:
        return False
    t = (embedded or "").strip()
    if not t:
        return False
    word_count = len(re.findall(r"\w+", t))
    return len(t) >= cfg.embedded_text_min_chars and word_count >= cfg.embedded_text_min_words

def process_pdf(pdf_path: str, cfg: PipelineConfig,
                easy: EasyOCREngine,
                trocr_printed: Optional[TrOCREngine],
                trocr_hw: Optional[TrOCREngine]) -> Dict[str, Any]:

    t0 = now_ms()
    pdf_path_local = dbfs_to_local(pdf_path)
    doc = fitz.open(pdf_path_local)

    doc_id = os.path.splitext(os.path.basename(pdf_path_local))[0]
    results_pages: List[Dict[str, Any]] = []

    n_pages = doc.page_count
    if cfg.max_pages is not None:
        n_pages = min(n_pages, cfg.max_pages)

    for page_idx in range(n_pages):
        page = doc.load_page(page_idx)

        embedded = extract_embedded_text(page)
        if should_use_embedded_text(embedded, cfg):
            # Fast path
            results_pages.append({
                "page": page_idx,
                "source": "embedded_text",
                "regions": [],
                "page_text": embedded,
            })
            continue

        # Render + preprocess
        page_bgr = render_pdf_page_to_bgr(page, dpi=cfg.render_dpi)
        page_bgr = preprocess_page_image_bgr(page_bgr, cfg)

        # If EasyOCR exists, use it for detection+printed recognition
        if easy.available():
            raw = easy.readtext(page_bgr)
            # raw items: (bbox_pts, text, conf)
            # Convert to region dicts
            regions = []
            for bbox_pts, text, conf in raw[: cfg.max_regions_per_page]:
                xyxy = bbox_points_to_xyxy(bbox_pts)
                x0, y0, x1, y1 = xyxy
                area = max(0, (x1 - x0) * (y1 - y0))
                if area < cfg.easyocr_min_box_area_px:
                    continue
                if (x1 - x0) < cfg.easyocr_min_box_side_px or (y1 - y0) < cfg.easyocr_min_box_side_px:
                    continue
                regions.append({
                    "bbox_points": bbox_pts,
                    "bbox_xyxy": list(xyxy),
                    "text": (text or "").strip(),
                    "confidence": float(conf) if conf is not None else -1.0,
                    "model": "easyocr",
                })

            # Decide which regions need TrOCR fallback
            fallback_indices = []
            for i, r in enumerate(regions):
                if r["confidence"] < cfg.easyocr_keep_conf or looks_like_gibberish(r["text"]):
                    fallback_indices.append(i)

            # Run TrOCR on fallbacks (batched)
            if fallback_indices and (trocr_printed is not None):
                crops_pil = []
                crop_meta = []
                for i in fallback_indices:
                    xyxy = tuple(regions[i]["bbox_xyxy"])
                    crop = crop_xyxy(page_bgr, xyxy, pad=cfg.crop_pad_px)
                    crops_pil.append(bgr_to_pil(crop))
                    crop_meta.append(i)

                # Batched printed recognition
                printed_texts = []
                printed_confs = []
                for j in range(0, len(crops_pil), cfg.trocr_batch_size):
                    batch_imgs = crops_pil[j:j+cfg.trocr_batch_size]
                    t, c = trocr_printed.recognize_batch(
                        batch_imgs,
                        max_new_tokens=cfg.trocr_max_new_tokens,
                        num_beams=cfg.trocr_num_beams,
                    )
                    printed_texts.extend([x.strip() for x in t])
                    printed_confs.extend(c)

                # Optionally run handwritten on low-confidence printed
                chosen_texts = printed_texts[:]
                chosen_confs = printed_confs[:]
                chosen_model = ["trocr-printed"] * len(printed_texts)

                if trocr_hw is not None:
                    need_hw = [k for k, c in enumerate(printed_confs) if c < cfg.trocr_printed_min_conf or looks_like_gibberish(printed_texts[k])]
                    if need_hw:
                        hw_imgs = [crops_pil[k] for k in need_hw]
                        hw_texts_all = []
                        hw_confs_all = []
                        for j in range(0, len(hw_imgs), cfg.trocr_batch_size):
                            t, c = trocr_hw.recognize_batch(
                                hw_imgs[j:j+cfg.trocr_batch_size],
                                max_new_tokens=cfg.trocr_max_new_tokens,
                                num_beams=cfg.trocr_num_beams,
                            )
                            hw_texts_all.extend([x.strip() for x in t])
                            hw_confs_all.extend(c)

                        # merge decisions back
                        for idx_local, k in enumerate(need_hw):
                            hw_text = hw_texts_all[idx_local]
                            hw_conf = hw_confs_all[idx_local]
                            if cfg.trocr_choose_best:
                                # pick best between printed and handwritten
                                if hw_conf > chosen_confs[k]:
                                    chosen_texts[k] = hw_text
                                    chosen_confs[k] = hw_conf
                                    chosen_model[k] = "trocr-handwritten"
                            else:
                                chosen_texts[k] = hw_text
                                chosen_confs[k] = hw_conf
                                chosen_model[k] = "trocr-handwritten"

                # Apply TrOCR results back into regions
                for k, region_index in enumerate(crop_meta):
                    txt = chosen_texts[k]
                    conf = chosen_confs[k]
                    mdl = chosen_model[k]
                    # Only overwrite if it improves things or EasyOCR was junk
                    if (conf > regions[region_index]["confidence"]) or looks_like_gibberish(regions[region_index]["text"]):
                        regions[region_index]["text"] = txt
                        regions[region_index]["confidence"] = float(conf)
                        regions[region_index]["model"] = mdl

            # Reading order + page text
            regions_sorted = sort_boxes_reading_order(regions)
            page_text = "\n".join([r["text"] for r in regions_sorted if (r["text"] or "").strip()])

            results_pages.append({
                "page": page_idx,
                "source": "ocr",
                "regions": regions_sorted,
                "page_text": page_text,
            })

        else:
            # Fallback: detect candidate text regions with OpenCV and run region-level TrOCR.
            # This is much better than whole-page TrOCR for dense forms.
            if trocr_printed is None:
                raise RuntimeError("No OCR engine available. Install easyocr or provide TrOCR models.")

            boxes = detect_text_regions_cv(page_bgr, cfg)
            boxes = merge_line_boxes(boxes)
            regions: List[Dict[str, Any]] = []

            # If detector finds nothing, keep a last-resort whole-page pass.
            if not boxes:
                pil = bgr_to_pil(page_bgr)
                txt_p, conf_p = trocr_printed.recognize_batch(
                    [pil],
                    max_new_tokens=cfg.trocr_max_new_tokens,
                    num_beams=cfg.trocr_num_beams,
                )
                txt_p, conf_p = txt_p[0].strip(), conf_p[0]

                txt = txt_p
                conf = conf_p
                mdl = "trocr-printed-page"

                if trocr_hw is not None and (conf_p < cfg.trocr_printed_min_conf or looks_like_gibberish(txt_p)):
                    txt_h, conf_h = trocr_hw.recognize_batch(
                        [pil],
                        max_new_tokens=cfg.trocr_max_new_tokens,
                        num_beams=cfg.trocr_num_beams,
                    )
                    txt_h, conf_h = txt_h[0].strip(), conf_h[0]
                    if cfg.trocr_choose_best and conf_h > conf_p:
                        txt, conf, mdl = txt_h, conf_h, "trocr-handwritten-page"

                regions = [{
                    "bbox_points": None,
                    "bbox_xyxy": None,
                    "text": txt,
                    "confidence": float(conf),
                    "model": mdl
                }]
                page_text = txt
                source = "ocr_page_fallback"
            else:
                crops_pil = []
                crop_meta = []
                for xyxy in boxes:
                    crop = crop_xyxy(page_bgr, xyxy, pad=cfg.crop_pad_px)
                    crops_pil.append(bgr_to_pil(crop))
                    crop_meta.append(xyxy)

                printed_texts: List[str] = []
                printed_confs: List[float] = []
                for j in range(0, len(crops_pil), cfg.trocr_batch_size):
                    t, c = trocr_printed.recognize_batch(
                        crops_pil[j:j+cfg.trocr_batch_size],
                        max_new_tokens=cfg.trocr_max_new_tokens,
                        num_beams=cfg.trocr_num_beams,
                    )
                    printed_texts.extend([x.strip() for x in t])
                    printed_confs.extend(c)

                chosen_texts = printed_texts[:]
                chosen_confs = printed_confs[:]
                chosen_model = ["trocr-printed"] * len(chosen_texts)

                if trocr_hw is not None:
                    need_hw = [k for k, c in enumerate(printed_confs) if c < cfg.trocr_printed_min_conf or looks_like_gibberish(printed_texts[k])]
                    if need_hw:
                        hw_imgs = [crops_pil[k] for k in need_hw]
                        hw_texts_all: List[str] = []
                        hw_confs_all: List[float] = []
                        for j in range(0, len(hw_imgs), cfg.trocr_batch_size):
                            t, c = trocr_hw.recognize_batch(
                                hw_imgs[j:j+cfg.trocr_batch_size],
                                max_new_tokens=cfg.trocr_max_new_tokens,
                                num_beams=cfg.trocr_num_beams,
                            )
                            hw_texts_all.extend([x.strip() for x in t])
                            hw_confs_all.extend(c)
                        for idx_local, k in enumerate(need_hw):
                            hw_text = hw_texts_all[idx_local]
                            hw_conf = hw_confs_all[idx_local]
                            if cfg.trocr_choose_best:
                                if hw_conf > chosen_confs[k]:
                                    chosen_texts[k] = hw_text
                                    chosen_confs[k] = hw_conf
                                    chosen_model[k] = "trocr-handwritten"
                            else:
                                chosen_texts[k] = hw_text
                                chosen_confs[k] = hw_conf
                                chosen_model[k] = "trocr-handwritten"

                for i_box, xyxy in enumerate(crop_meta):
                    txt = chosen_texts[i_box]
                    if not txt:
                        continue
                    regions.append({
                        "bbox_points": None,
                        "bbox_xyxy": list(xyxy),
                        "text": txt,
                        "confidence": float(chosen_confs[i_box]),
                        "model": chosen_model[i_box],
                    })

                regions = sort_boxes_reading_order(regions)
                page_text = "\n".join([r["text"] for r in regions if (r["text"] or "").strip()])
                source = "ocr_cv_trocr_fallback"

            results_pages.append({
                "page": page_idx,
                "source": source,
                "regions": regions,
                "page_text": page_text,
            })

    doc.close()

    return {
        "doc_id": doc_id,
        "pdf_path": pdf_path,
        "n_pages": n_pages,
        "pages": results_pages,
        "elapsed_ms": now_ms() - t0
    }




def run_ocr_with_loaded_models(local_pdf_paths: List[str], cfg: PipelineConfig, model_bundle: OCRModelBundle) -> None:
    """Run OCR with preloaded models so repeated iterations avoid reload overhead."""
    for i, pdf_local in enumerate(local_pdf_paths, 1):
        print(f"\n[INFO] ({i}/{len(local_pdf_paths)}) {pdf_local}")
        res = process_pdf(pdf_local, cfg, model_bundle.easy, model_bundle.trocr_printed, model_bundle.trocr_handwritten)
        write_outputs(res, cfg)
        print(f"[OK] {res['doc_id']} elapsed_ms={res['elapsed_ms']}")

def list_pdfs(input_path: str, recursive: bool) -> List[str]:
    p = dbfs_to_local(input_path)
    if os.path.isfile(p) and p.lower().endswith(".pdf"):
        return [input_path]

    # Directory
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdfs = glob.glob(os.path.join(p, pattern), recursive=recursive)
    # Return in same "style" as input (local paths are fine; dbfs:/ also fine)
    return [pdf for pdf in pdfs]


def write_outputs(run_result: Dict[str, Any], cfg: PipelineConfig) -> None:
    out_dir = dbfs_to_local(cfg.output_dir)
    ensure_dir(out_dir)

    doc_id = safe_filename(run_result["doc_id"])
    jsonl_path = os.path.join(out_dir, f"{doc_id}.regions.jsonl")
    page_txt_path = os.path.join(out_dir, f"{doc_id}.pages.txt")

    if cfg.write_jsonl:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for page_obj in run_result["pages"]:
                for r in page_obj["regions"]:
                    row = {
                        "doc_id": doc_id,
                        "pdf_path": run_result["pdf_path"],
                        "page": page_obj["page"],
                        "source": page_obj["source"],
                        "bbox_xyxy": r.get("bbox_xyxy"),
                        "bbox_points": r.get("bbox_points"),
                        "text": r.get("text"),
                        "confidence": r.get("confidence"),
                        "model": r.get("model"),
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if cfg.write_page_text:
        with open(page_txt_path, "w", encoding="utf-8") as f:
            for page_obj in run_result["pages"]:
                f.write(f"\n===== PAGE {page_obj['page']} =====\n")
                f.write((page_obj.get("page_text") or "").strip() + "\n")

    # Optional: write to Delta if Spark is available
    if cfg.write_delta_if_spark and cfg.output_delta_path:
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()

            rows = []
            for page_obj in run_result["pages"]:
                for r in page_obj["regions"]:
                    rows.append((
                        doc_id,
                        run_result["pdf_path"],
                        int(page_obj["page"]),
                        str(page_obj["source"]),
                        r.get("bbox_xyxy"),
                        r.get("text"),
                        float(r.get("confidence", -1.0)),
                        str(r.get("model")),
                    ))

            df = spark.createDataFrame(rows, schema="""
                doc_id STRING,
                pdf_path STRING,
                page INT,
                source STRING,
                bbox_xyxy ARRAY<INT>,
                text STRING,
                confidence DOUBLE,
                model STRING
            """)
            df.write.format("delta").mode("append").save(cfg.output_delta_path)
        except Exception as e:
            print(f"[WARN] Delta write failed: {e}")
# =========================
#           CLI
# =========================

def build_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    model_root = dbfs_to_local(args.model_root)

    cfg = PipelineConfig(
        model_root=args.model_root,
        trocr_printed_path=args.trocr_printed_path or os.path.join(model_root, "trocr-base-printed"),
        trocr_handwritten_path=args.trocr_handwritten_path or os.path.join(model_root, "trocr-base-handwritten"),
        easyocr_model_dir=args.easyocr_model_dir,

        input_path=args.input_path,
        output_dir=args.output_dir,
        recursive=not args.no_recursive,

        render_dpi=args.render_dpi,
        max_pages=args.max_pages,
        embedded_text_min_chars=args.embedded_text_min_chars,
        embedded_text_min_words=getattr(args, "embedded_text_min_words", 8),
        use_embedded_text_fast_path=getattr(args, "use_embedded_text_fast_path", False),

        do_preprocess=not args.no_preprocess,
        deskew=not args.no_deskew,
        clahe=not args.no_clahe,
        denoise=not args.no_denoise,
        binarize=not args.no_binarize,

        enable_easyocr=not args.no_easyocr,
        easyocr_langs=tuple(args.easyocr_langs.split(",")) if args.easyocr_langs else ("en",),

        easyocr_keep_conf=args.easyocr_keep_conf,
        easyocr_min_box_area_px=getattr(args, "easyocr_min_box_area_px", 450),
        easyocr_min_box_side_px=getattr(args, "easyocr_min_box_side_px", 12),
        trocr_printed_min_conf=args.trocr_printed_min_conf,
        trocr_choose_best=not args.no_choose_best,

        max_regions_per_page=args.max_regions_per_page,
        crop_pad_px=args.crop_pad_px,
        trocr_batch_size=args.trocr_batch_size,
        trocr_max_new_tokens=getattr(args, "trocr_max_new_tokens", 96),
        trocr_num_beams=getattr(args, "trocr_num_beams", 4),
        torch_num_threads=args.torch_num_threads,

        write_jsonl=not args.no_jsonl,
        write_page_text=not args.no_page_text,
        write_delta_if_spark=args.write_delta,
        output_delta_path=args.output_delta_path
    )
    return cfg


# =========================
# Notebook runner cell
# =========================

import os, time, uuid

# ---------- EDIT THESE ----------
S3_MODEL_ROOT   = "s3a://foundational-model-zone/"          # folder that contains trocr-base-printed/, trocr-base-handwritten/, etc.
S3_INPUT_PDFS   = "s3://data-zone//scans_ready"      # folder containing PDFs (can be deep)
S3_OUTPUT_ROOT  = "s3://data-zone//results"     # where you want results written

# If you already have models on DBFS (e.g., dbfs:/FileStore/models), set USE_S3_MODELS=False and set DBFS_MODEL_ROOT accordingly.
USE_S3_MODELS   = True

# Optional: If EasyOCR must be offline, point this to a directory you staged weights into.
# If None, EasyOCR uses its default cache and may attempt downloads.
EASYOCR_MODEL_DIR_S3 = None  # e.g. "s3a://foundational-model-zone/easyocr_cache"

# OCR parameters
RENDER_DPI = 350
MAX_PAGES = None          # set e.g. 3 for a fast test
TORCH_NUM_THREADS = 8     # CPU threads for torch (set 0 to leave default)

# Better defaults for scanned forms: adaptive binarization often helps text isolation.
ENABLE_BINARIZE = True

# ---------- Internal staging dirs (local) ----------
RUN_ID = uuid.uuid4().hex[:8]
LOCAL_STAGE_ROOT = f"/dbfs/tmp/pdf_ocr_run_{RUN_ID}"
LOCAL_MODELS_DIR = os.path.join(LOCAL_STAGE_ROOT, "models")
LOCAL_PDFS_DIR   = os.path.join(LOCAL_STAGE_ROOT, "pdfs")
LOCAL_OUT_DIR    = os.path.join(LOCAL_STAGE_ROOT, "out")

os.makedirs(LOCAL_STAGE_ROOT, exist_ok=True)
os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)
os.makedirs(LOCAL_PDFS_DIR, exist_ok=True)
os.makedirs(LOCAL_OUT_DIR, exist_ok=True)

def _cp(src: str, dst: str, recurse: bool = False):
    """
    Copy using dbutils.fs.cp with correct scheme handling.
    - src can be s3a:/, dbfs:/, file:/...
    - dst can be file:/dbfs/... or s3a:/ or dbfs:/
    """
    print(f"[COPY] {src} -> {dst} (recurse={recurse})")
    dbutils.fs.cp(src, dst, recurse=recurse)

def list_pdfs_recursive(uri: str):
    """
    Recursively list PDFs under a Databricks filesystem URI (s3a:/, dbfs:/, etc.)
    Returns list of file URIs.
    """
    out = []
    stack = [uri]
    while stack:
        cur = stack.pop()
        for fi in dbutils.fs.ls(cur):
            if fi.isDir():
                stack.append(fi.path)
            else:
                if fi.path.lower().endswith(".pdf"):
                    out.append(fi.path)
    return sorted(out)

# ---------- Stage models locally ----------
if USE_S3_MODELS:
    # Copy only what we need for this pipeline
    needed_model_dirs = [
        "trocr-base-printed",
        "trocr-base-handwritten",
    ]
    for d in needed_model_dirs:
        _cp(f"{S3_MODEL_ROOT.rstrip('/')}/{d}", f"file:{LOCAL_MODELS_DIR}/{d}", recurse=True)

    if EASYOCR_MODEL_DIR_S3:
        _cp(EASYOCR_MODEL_DIR_S3, f"file:{LOCAL_MODELS_DIR}/easyocr_cache", recurse=True)
        LOCAL_EASYOCR_DIR = f"{LOCAL_MODELS_DIR}/easyocr_cache"
    else:
        LOCAL_EASYOCR_DIR = None

    LOCAL_MODEL_ROOT = LOCAL_MODELS_DIR
else:
    # Example if your models are already in DBFS:
    # DBFS_MODEL_ROOT = "dbfs:/FileStore/models"
    # LOCAL_MODEL_ROOT = "/dbfs/FileStore/models"
    raise RuntimeError("Set USE_S3_MODELS=True or implement your DBFS model root mapping here.")

# ---------- Stage PDFs locally ----------
pdf_uris = list_pdfs_recursive(S3_INPUT_PDFS)
if not pdf_uris:
    raise FileNotFoundError(f"No PDFs found under: {S3_INPUT_PDFS}")

print(f"[INFO] Found {len(pdf_uris)} PDFs under {S3_INPUT_PDFS}")
for i, pdf_uri in enumerate(pdf_uris, 1):
    base = os.path.basename(pdf_uri)
    dst_local_file = f"file:{LOCAL_PDFS_DIR}/{base}"
    _cp(pdf_uri, dst_local_file, recurse=False)

local_pdf_paths = [os.path.join(LOCAL_PDFS_DIR, os.path.basename(u)) for u in pdf_uris]

# ---------- Build PipelineConfig (uses your classes already defined above) ----------
cfg = PipelineConfig(
    model_root=LOCAL_MODEL_ROOT,
    trocr_printed_path=os.path.join(LOCAL_MODEL_ROOT, "trocr-base-printed"),
    trocr_handwritten_path=os.path.join(LOCAL_MODEL_ROOT, "trocr-base-handwritten"),
    easyocr_model_dir=LOCAL_EASYOCR_DIR,

    input_path=LOCAL_PDFS_DIR,
    output_dir=LOCAL_OUT_DIR,
    recursive=True,

    render_dpi=RENDER_DPI,
    max_pages=MAX_PAGES,
    binarize=ENABLE_BINARIZE,
    use_embedded_text_fast_path=False,
    easyocr_min_box_area_px=500,
    easyocr_min_box_side_px=14,
    trocr_max_new_tokens=96,
    trocr_num_beams=4,

    # You can tweak thresholds if needed:
    # easyocr_keep_conf=0.65,
    # trocr_printed_min_conf=0.35,

    torch_num_threads=TORCH_NUM_THREADS,
)

# Torch CPU threads
import torch
if cfg.torch_num_threads and cfg.torch_num_threads > 0:
    torch.set_num_threads(cfg.torch_num_threads)

# ---------- Init engines ----------
print("[INFO] Initializing OCR engines...")
# Models are cached in-process by path/config, so repeated runs in the same
# cluster session can skip reloads and iterate faster.
model_bundle = load_ocr_model_bundle(cfg, device="cpu", use_cache=True, load_handwritten=True)
easy = model_bundle.easy
trocr_printed = model_bundle.trocr_printed
trocr_hw = model_bundle.trocr_handwritten

# ---------- Run ----------
t0 = time.time()
print(f"[INFO] Running pipeline on {len(local_pdf_paths)} local PDFs...")

run_ocr_with_loaded_models(local_pdf_paths[:4], cfg, model_bundle)

elapsed = time.time() - t0
print(f"\n[INFO] Done OCR. Total seconds: {elapsed:.1f}")
print(f"[INFO] Local outputs at: {LOCAL_OUT_DIR}")

# ---------- Copy outputs back to S3 ----------
# Put each run under a unique prefix to avoid collisions
S3_RUN_OUT = f"{S3_OUTPUT_ROOT.rstrip('/')}/run_{RUN_ID}"
_cp(f"file:{LOCAL_OUT_DIR}", S3_RUN_OUT, recurse=True)
latest_run_true = str(S3_RUN_OUT)
print(f"[INFO] Outputs copied to: {S3_RUN_OUT}")
