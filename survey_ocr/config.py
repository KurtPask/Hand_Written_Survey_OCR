from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    scanned_docs_dir: Path
    output_dir: Path
    raw_oai_dir: Path
    results_all_json: Path
    analysis_json: Path
    analysis_charts_dir: Path
    dotenv_path: Path
    gcp_vision_key: Optional[Path]


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_config(project_root: Optional[Path] = None) -> AppConfig:
    root = project_root or _default_project_root()
    scanned_docs_dir = Path(
        os.getenv("DOCEXTRACT_SCANNED_DOCS", root / "Inputs")
    )
    output_dir = Path(
        os.getenv("DOCEXTRACT_OUTPUT_DIR", root / "Outputs")
    )
    raw_oai_dir = Path(
        os.getenv(
            "DOCEXTRACT_RAW_OAI_DIR",
            output_dir / "raw_openai_outputs_low_token",
        )
    )
    results_all_json = Path(
        os.getenv(
            "DOCEXTRACT_RESULTS_ALL_JSON",
            output_dir / "high_risk_forms_all.json",
        )
    )
    analysis_json = Path(
        os.getenv(
            "DOCEXTRACT_ANALYSIS_JSON",
            output_dir / "high_risk_forms_analysis.json",
        )
    )
    analysis_charts_dir = Path(
        os.getenv(
            "DOCEXTRACT_ANALYSIS_CHARTS_DIR",
            output_dir / "analysis_charts",
        )
    )
    dotenv_path = Path(os.getenv("DOCEXTRACT_DOTENV", root / ".env"))
    gcp_key_value = os.getenv("GCP_VISION_KEY")
    gcp_vision_key = Path(gcp_key_value) if gcp_key_value else None

    return AppConfig(
        project_root=root,
        scanned_docs_dir=scanned_docs_dir,
        output_dir=output_dir,
        raw_oai_dir=raw_oai_dir,
        results_all_json=results_all_json,
        analysis_json=analysis_json,
        analysis_charts_dir=analysis_charts_dir,
        dotenv_path=dotenv_path,
        gcp_vision_key=gcp_vision_key,
    )
