# Hand_Written_Survey_OCR

## Production-oriented quick start

This repo now supports centralized configuration and a CLI for running the
pipelines in new environments without editing source files.

### Configuration

Set configuration via environment variables (recommended for production):

- `DOCEXTRACT_SCANNED_DOCS`: input folder for PDFs (default: `./Inputs`)
- `DOCEXTRACT_OUTPUT_DIR`: output folder (default: `./Outputs`)
- `DOCEXTRACT_RAW_OAI_DIR`: override raw OpenAI response output directory
- `DOCEXTRACT_RESULTS_ALL_JSON`: override results JSON path
- `DOCEXTRACT_ANALYSIS_JSON`: override analysis JSON path
- `DOCEXTRACT_ANALYSIS_CHARTS_DIR`: override chart output directory
- `DOCEXTRACT_DOTENV`: path to `.env` with API keys (default: `./.env`)
- `GCP_VISION_KEY`: path to Google Vision credentials JSON (only for GCP pipeline)

### CLI usage

Run the OpenAI-only pipeline (Final_code.py):

```bash
python -m survey_ocr.cli final
```

Run the OpenAI OCR pipeline (OCR_OAI.py):

```bash
python -m survey_ocr.cli ocr-oai
```

Run the Google Vision + OpenAI pipeline (TEST_MultiDoc_Pipeline.py):

```bash
python -m survey_ocr.cli gcp-pipeline
```
