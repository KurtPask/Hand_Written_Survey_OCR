from __future__ import annotations

import argparse
import sys


def _run_final() -> int:
    import Final_code

    Final_code.run_batch()
    return 0


def _run_ocr_oai() -> int:
    import OCR_OAI

    OCR_OAI.run_batch()
    return 0


def _run_gcp_pipeline() -> int:
    import TEST_MultiDoc_Pipeline

    TEST_MultiDoc_Pipeline.run_batch_pipeline()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Survey OCR pipelines",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("final", help="Run the OpenAI-only batch pipeline")
    subparsers.add_parser("ocr-oai", help="Run the OpenAI-only OCR pipeline")
    subparsers.add_parser(
        "gcp-pipeline",
        help="Run the Google Vision + OpenAI batch pipeline",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "final":
        return _run_final()
    if args.command == "ocr-oai":
        return _run_ocr_oai()
    if args.command == "gcp-pipeline":
        return _run_gcp_pipeline()

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
