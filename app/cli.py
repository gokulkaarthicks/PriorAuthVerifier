"""Command-line entrypoint for PriorAuthVerifier."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from app.core.config import get_settings
from app.core.dependencies import build_services
from app.core.exceptions import (
    ConfigurationError,
    DocumentProcessingError,
    PriorAuthVerifierError,
)
from app.core.logging import configure_logging, get_logger
from app.schemas.analysis import AnalysisResponse, ReadinessStatus
from app.services.analysis_service import AnalysisService

EXIT_SUCCESS = 0
EXIT_UNEXPECTED = 1
EXIT_USAGE = 2
EXIT_DOCUMENT = 3
EXIT_CONFIGURATION = 4

_DESC_PREVIEW = 100
_EXPLAIN_PREVIEW = 140


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for end-to-end analysis."""
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="PriorAuthVerifier end-to-end prior-authorization analysis",
    )
    parser.add_argument(
        "--patient",
        type=Path,
        required=True,
        help="Path to a patient chart PDF",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        required=True,
        help="Path to a policy PDF",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print AnalysisResponse JSON only",
    )
    return parser


def print_human_readable(result: AnalysisResponse) -> None:
    """Print a concise human-readable analysis report."""
    procedure = result.procedure
    print(f"Procedure: {procedure.procedure_name}")
    print(f"CPT: {procedure.cpt_code}")
    print(f"Procedure status: {procedure.status}")
    print(f"Matching policy criteria found: {result.matching_policy_criteria_found}")
    print(f"Overall readiness: {result.overall_status.value}")
    print()

    if result.overall_status == ReadinessStatus.NOT_APPLICABLE:
        print(f"Next step: {result.next_step}")
        print()
        return

    print("Requirements:")
    if not result.criteria:
        print("- none")
    else:
        for criterion in result.criteria:
            description = (criterion.description or "").strip()
            desc_preview = " ".join(description.split())[:_DESC_PREVIEW]
            explanation = " ".join(criterion.explanation.split())[:_EXPLAIN_PREVIEW]
            pages = ", ".join(str(page) for page in criterion.page_numbers) or "none"
            print(f"- {criterion.criterion_id}")
            if desc_preview:
                print(f"  Description: {desc_preview}")
            print(f"  Status: {criterion.status.value}")
            print(f"  Evidence pages: {pages}")
            print(f"  Explanation: {explanation}")
    print()

    print("Exclusions:")
    if not result.exclusions:
        print("- none")
    else:
        for exclusion in result.exclusions:
            triggered = "triggered" if exclusion.triggered else "not triggered"
            print(
                f"- {exclusion.exclusion_id}: {exclusion.status.value} "
                f"({triggered})"
            )
    print()

    print("Missing items:")
    if not result.missing_items:
        print("- none")
    else:
        for missing in result.missing_items:
            print(f"- {missing.criterion_id}: {missing.description}")
    print()

    print("Needs review:")
    if not result.review_items:
        print("- none")
    else:
        for review in result.review_items:
            print(f"- {review.criterion_id}: {review.description}")
    print()

    print(f"Next step: {result.next_step}")
    print()


def print_json(result: AnalysisResponse) -> None:
    payload = result.model_dump(mode="json")
    print(json.dumps(payload, indent=2, ensure_ascii=True))


def map_exception_to_exit_code(exc: BaseException) -> int:
    """Map application exceptions to deterministic process exit codes."""
    if isinstance(exc, ConfigurationError):
        return EXIT_CONFIGURATION
    if isinstance(exc, DocumentProcessingError):
        return EXIT_DOCUMENT
    if isinstance(exc, PriorAuthVerifierError):
        return EXIT_UNEXPECTED
    return EXIT_UNEXPECTED


async def run_analysis(
    *,
    patient_path: Path,
    policy_path: Path,
) -> AnalysisResponse:
    """Build shared services and execute the analysis pipeline."""
    settings = get_settings()
    services = build_services(settings)
    analysis_service = AnalysisService(services)
    return await analysis_service.analyze(patient_path, policy_path)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint returning a process exit code."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return EXIT_SUCCESS
        if isinstance(code, int):
            return code if code in {0, 2} else EXIT_USAGE
        return EXIT_USAGE

    settings = get_settings()
    configure_logging(settings)
    logger = get_logger(__name__)

    try:
        result = asyncio.run(
            run_analysis(
                patient_path=args.patient,
                policy_path=args.policy,
            )
        )
    except PriorAuthVerifierError as exc:
        logger.error("cli_analysis_failed", error=str(exc))
        print(f"Error: {exc}", file=sys.stderr)
        return map_exception_to_exit_code(exc)
    except Exception as exc:
        logger.error("cli_unexpected_failure", error=str(exc))
        print("Error: unexpected application failure", file=sys.stderr)
        return EXIT_UNEXPECTED

    if args.json:
        print_json(result)
    else:
        print_human_readable(result)
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
