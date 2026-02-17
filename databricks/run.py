# =========================
# run.py (Databricks notebook block) — v3
# Uses utils.py v3 with:
#  - JSON-safe output
#  - per-field local alignment + scan-line refine + y-projection tighten
# =========================

# ---- 2) FORM SELECTION RULES ----
FORM_RULES = [
    {"match_regex": r"WSD|STUDENT_CRITIQUE|WATER_SURVIVAL", "config_path": "wsd_student_critique_config.json"},
]
DEFAULT_CONFIG_PATH = "wsd_student_critique_config.json"

# ---- 3) OPTIONAL TUNING ----
RECURSE      = True
MAX_PDFS     = None
MAX_PAGES    = None
FIELD_PAD_PX = 6

# ---- 4) Crop tightening switches ----
ENABLE_SCAN_LINE_REFINE = True
ENABLE_YPROJ_TIGHTEN    = True
ENABLE_COMPONENT_TIGHTEN= True
EMPTY_INK_THRESH        = 0.00035   # slightly lower to keep faint handwriting strokes

# ---- 5) DEBUG ----
DEBUG_SAVE = True
DEBUG_MAX_PDFS = 8
DEBUG_LIMIT_PER_PDF = 4000

# ---- 6) Device ----
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    torch.set_num_threads(max(1, os.cpu_count() // 2))
except Exception:
    pass

# ---- 7) Load models ----
trocr_printed = load_trocr_bundle(dbutils, FMZ_ROOT, "trocr-base-printed", device=device)
trocr_hand    = load_trocr_bundle(dbutils, FMZ_ROOT, "trocr-base-handwritten", device=device)

# ---- 8) PDFs ----
pdfs = list_pdfs(dbutils, PDF_INPUT, recurse=RECURSE)
if MAX_PDFS is not None:
    pdfs = pdfs[:MAX_PDFS]

run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
run_out_dir = join_path(OUTPUT_ROOT, f"run_id={run_id}")
ensure_dir_dbutils(dbutils, run_out_dir)

debug_dir = join_path(run_out_dir, "_debug_crops") if DEBUG_SAVE else None
if debug_dir:
    ensure_dir_dbutils(dbutils, debug_dir)

print(f"Run: {run_id}")
print(f"Device: {device}")
print(f"PDFs found: {len(pdfs)}")
print(f"Output dir: {run_out_dir}")
print(
    "Crop refine toggles: "
    f"scan_line={ENABLE_SCAN_LINE_REFINE}, "
    f"yproj={ENABLE_YPROJ_TIGHTEN}, "
    f"components={ENABLE_COMPONENT_TIGHTEN}, "
    f"empty_ink_thresh={EMPTY_INK_THRESH}"
)
if debug_dir:
    print(f"Debug dir: {debug_dir}")

# ---- 9) Config/template caching ----
_loaded_form_cfg: Dict[str, Dict[str, Any]] = {}
_loaded_templates: Dict[str, TemplateCache] = {}

def pick_form_config_for_pdf(pdf_path: str) -> Optional[Dict[str, Any]]:
    name = os.path.basename(pdf_path)
    for rule in FORM_RULES:
        if re.search(rule["match_regex"], name, flags=re.IGNORECASE):
            cfg_path = rule["config_path"]
            if cfg_path not in _loaded_form_cfg:
                _loaded_form_cfg[cfg_path] = load_json_config(dbutils, cfg_path)
            return _loaded_form_cfg[cfg_path]
    if DEFAULT_CONFIG_PATH:
        if DEFAULT_CONFIG_PATH not in _loaded_form_cfg:
            _loaded_form_cfg[DEFAULT_CONFIG_PATH] = load_json_config(dbutils, DEFAULT_CONFIG_PATH)
        return _loaded_form_cfg[DEFAULT_CONFIG_PATH]
    return None

def get_template_cache(form_cfg: Dict[str, Any]) -> TemplateCache:
    key = f"{form_cfg.get('template_pdf')}::dpi={form_cfg.get('dpi', 300)}"
    if key not in _loaded_templates:
        _loaded_templates[key] = load_template_cache(dbutils, form_cfg)
    return _loaded_templates[key]

# ---- 10) Run ----
manifest = {
    "run_id": run_id,
    "fmz_root": FMZ_ROOT,
    "pdf_input": PDF_INPUT,
    "output_root": OUTPUT_ROOT,
    "output_run_dir": run_out_dir,
    "device": device,
    "pdf_count": len(pdfs),
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "results": []
}

t_run = time.time()

for i, pdf_path in enumerate(pdfs, start=1):
    base = os.path.basename(pdf_path)
    stem = re.sub(r"\.pdf$", "", base, flags=re.IGNORECASE)
    out_json = join_path(run_out_dir, f"{stem}.json")

    print(f"[{i}/{len(pdfs)}] {base}")

    item = {"pdf_path": pdf_path, "out_json": out_json}

    try:
        form_cfg = pick_form_config_for_pdf(pdf_path)
        if form_cfg is None:
            raise ValueError(
                f"No form config matched filename '{base}'. "
                f"Add a FORM_RULES regex or set DEFAULT_CONFIG_PATH."
            )

        template_cache = get_template_cache(form_cfg)
        this_debug_dir = debug_dir if (debug_dir and i <= DEBUG_MAX_PDFS) else None

        result = ocr_pdf_with_form_config(
            dbutils=dbutils,
            pdf_path=pdf_path,
            device=device,
            trocr_printed=trocr_printed,
            trocr_handwritten=trocr_hand,
            form_cfg=form_cfg,
            template_cache=template_cache,
            max_pages=MAX_PAGES,
            field_pad_px=FIELD_PAD_PX,

            enable_scan_line_refine=ENABLE_SCAN_LINE_REFINE,
            enable_yproj_tighten=ENABLE_YPROJ_TIGHTEN,
            enable_component_tighten=ENABLE_COMPONENT_TIGHTEN,
            empty_ink_thresh=EMPTY_INK_THRESH,

            debug_dir=this_debug_dir,
            debug_save_limit=DEBUG_LIMIT_PER_PDF,
        )

        write_json_to_output(dbutils, result, out_json)

        item.update({
            "status": "ok",
            "form_id": result.get("form_id"),
            "mode": result.get("mode"),
            "runtime_sec": result.get("runtime_sec"),
            "num_fields": len(result.get("fields", {})),
        })

    except Exception as e:
        item.update({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()[:8000],
        })
        write_json_to_output(dbutils, item, out_json)

    manifest["results"].append(item)

manifest["total_runtime_sec"] = round(time.time() - t_run, 3)
manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

manifest_path = join_path(run_out_dir, "_MANIFEST.json")
write_json_to_output(dbutils, manifest, manifest_path)

print(f"Done. Manifest: {manifest_path}")
print(f"Total runtime (sec): {manifest['total_runtime_sec']}")
