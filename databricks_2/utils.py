# =========================================
# RUN v3: WSD Student Critique extraction (robust)
# =========================================

import os
import pandas as pd
from datetime import datetime, timezone

# You said these are declared in your first cell:
# FMZ_ROOT   = "s3a://fmz/"
# PDF_INPUT  = "s3://front/scans_ready"
# OUTPUT_ROOT= "s3://front/results"

TEMPLATE_PDF_LOCAL = "WSD Student Critique.pdf"  # local relative path
DPI = 300

# FMZ model dirs
TROCR_HANDWRITTEN_NAME = "trocr-base-handwritten"
TROCR_PRINTED_NAME     = "trocr-base-printed"   # optional but recommended

# Output
OUT_JSON_DIR     = _to_s3a(f"{OUTPUT_ROOT}/wsd_critique/json")
OUT_TABLE_DIR    = _to_s3a(f"{OUTPUT_ROOT}/wsd_critique/delta")
OUT_CROPS_DIR    = _to_s3a(f"{OUTPUT_ROOT}/wsd_critique/debug_field_crops")

ensure_dir_dbfs(OUT_JSON_DIR)
ensure_dir_dbfs(OUT_TABLE_DIR)
ensure_dir_dbfs(OUT_CROPS_DIR)

LOCAL_INPUT_DIR = "/dbfs/tmp/ocr_inputs"
LOCAL_MODEL_DIR = "/dbfs/tmp/fmz_model_cache"
os.makedirs(LOCAL_INPUT_DIR, exist_ok=True)
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

if not os.path.exists(TEMPLATE_PDF_LOCAL):
    raise FileNotFoundError(f"Template PDF not found at: {TEMPLATE_PDF_LOCAL}")

# Render template pages once
tpl_pages = render_pdf_pages_to_images(TEMPLATE_PDF_LOCAL, dpi=DPI)
if len(tpl_pages) < 2:
    raise RuntimeError("Template PDF must have 2 pages for WSD Student Critique.")

template_p0_gray = np.array(tpl_pages[0].convert("L"))
template_p1_gray = np.array(tpl_pages[1].convert("L"))

# Load models locally
hand_dir = copy_dir_uri_to_local(_to_s3a(f"{FMZ_ROOT.rstrip('/')}/{TROCR_HANDWRITTEN_NAME}"), LOCAL_MODEL_DIR)
trocr_hand = TrocrEngine(hand_dir)

trocr_print = None
try:
    print_dir = copy_dir_uri_to_local(_to_s3a(f"{FMZ_ROOT.rstrip('/')}/{TROCR_PRINTED_NAME}"), LOCAL_MODEL_DIR)
    trocr_print = TrocrEngine(print_dir)
except Exception:
    trocr_print = None

# Discover PDFs
pdf_uris = list_pdfs_recursive(PDF_INPUT)
print(f"Found {len(pdf_uris)} PDFs under {PDF_INPUT}")

rows = []

for i, pdf_uri in enumerate(pdf_uris):
    processed_utc = datetime.now(timezone.utc).isoformat()
    pdf_local = copy_uri_to_local_file(pdf_uri, LOCAL_INPUT_DIR)
    base = os.path.basename(pdf_uri).rsplit(".", 1)[0]

    extracted = extract_wsd_v3(
        pdf_local_path=pdf_local,
        template_p0_gray=template_p0_gray,
        template_p1_gray=template_p1_gray,
        trocr_hand=trocr_hand,
        trocr_print=trocr_print,
        dpi=DPI,
        conf_margin=2.0,
        debug_crop_dump=True,              # <- set False after you’re confident
        debug_crop_out_uri=OUT_CROPS_DIR,  # dumps crops per PDF under this folder
        pdf_base=base
    )

    extracted["source_pdf_uri"] = _to_s3a(pdf_uri)
    extracted["processed_utc"] = processed_utc

    # Write JSON
    write_json_to_uri(extracted, f"{OUT_JSON_DIR}/{base}.json", overwrite=True)

    # Flatten for analytics
    flat = {k: v for k, v in extracted.items() if not k.startswith("_")}
    flat["review_count"] = len(extracted.get("_review", []))
    rows.append(flat)

    if (i + 1) % 10 == 0:
        print(f"Processed {i+1}/{len(pdf_uris)}")

# Write Delta table
df = spark.createDataFrame(pd.DataFrame(rows))
(df.write.format("delta").mode("overwrite").save(OUT_TABLE_DIR))

print(f"✅ Delta: {OUT_TABLE_DIR}")
print(f"✅ JSON:  {OUT_JSON_DIR}")
print(f"🧪 Crops: {OUT_CROPS_DIR} (per-field crops for debugging/review)")