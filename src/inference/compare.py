"""Compare benchmark results between system and user prompt modes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_json_response(raw_response: str | None) -> dict[str, Any] | None:
    """Parse JSON response from raw_response column."""
    if raw_response is None or pd.isna(raw_response):
        return None

    try:
        # Handle responses wrapped in markdown code blocks
        content = str(raw_response).strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        return dict(json.loads(content))
    except (json.JSONDecodeError, TypeError):
        return None


def normalize_response(parsed: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize parsed response for comparison."""
    if parsed is None:
        return None

    # Extract key fields for comparison
    normalized: dict[str, Any] = {
        "image_type": parsed.get("image_type", ""),
        "areas": [],
    }

    areas = parsed.get("areas", [])
    for area in areas:
        normalized_area: dict[str, Any] = {
            "area_name": area.get("area_name", ""),
            "contaminations": [],
        }
        contaminations = area.get("contaminations", [])
        for cont in contaminations:
            normalized_area["contaminations"].append(
                {
                    "contamination_type": cont.get("contamination_type", ""),
                    "severity": cont.get("severity", ""),
                }
            )
        # Sort contaminations for consistent comparison
        normalized_area["contaminations"].sort(key=lambda x: (x["contamination_type"], x["severity"]))
        normalized["areas"].append(normalized_area)

    # Sort areas for consistent comparison
    normalized["areas"].sort(key=lambda x: x["area_name"])

    return normalized


def compare_responses(system_raw: str | None, user_raw: str | None) -> dict[str, Any]:
    """Compare two responses and return comparison result."""
    system_parsed = parse_json_response(system_raw)
    user_parsed = parse_json_response(user_raw)

    result: dict[str, Any] = {
        "system_parsed": system_parsed is not None,
        "user_parsed": user_parsed is not None,
        "identical": False,
        "differences": [],
    }

    if system_parsed is None or user_parsed is None:
        return result

    system_norm = normalize_response(system_parsed)
    user_norm = normalize_response(user_parsed)

    if system_norm == user_norm:
        result["identical"] = True
        return result

    # Find specific differences
    differences = []

    # Compare image_type
    if system_norm and user_norm:
        if system_norm.get("image_type") != user_norm.get("image_type"):
            differences.append(
                {
                    "field": "image_type",
                    "system": system_norm.get("image_type"),
                    "user": user_norm.get("image_type"),
                }
            )

        # Compare areas
        system_areas = {a["area_name"]: a for a in system_norm.get("areas", [])}
        user_areas = {a["area_name"]: a for a in user_norm.get("areas", [])}

        all_area_names = set(system_areas.keys()) | set(user_areas.keys())
        for area_name in all_area_names:
            system_area = system_areas.get(area_name)
            user_area = user_areas.get(area_name)

            if system_area is None:
                differences.append({"field": f"area:{area_name}", "system": None, "user": "present"})
            elif user_area is None:
                differences.append({"field": f"area:{area_name}", "system": "present", "user": None})
            elif system_area != user_area:
                differences.append(
                    {
                        "field": f"area:{area_name}",
                        "system": system_area.get("contaminations"),
                        "user": user_area.get("contaminations"),
                    }
                )

    result["differences"] = differences
    return result


def compare_benchmark_results(system_csv: Path, user_csv: Path) -> dict[str, Any]:
    """
    Compare benchmark results from system and user mode CSVs.

    Args:
        system_csv: Path to system mode benchmark results CSV
        user_csv: Path to user mode benchmark results CSV

    Returns:
        Dictionary with comparison metrics and details
    """
    system_df = pd.read_csv(system_csv)
    user_df = pd.read_csv(user_csv)

    # Merge on file_name
    merged = system_df.merge(user_df, on="file_name", suffixes=("_system", "_user"))

    total_compared = len(merged)
    identical_count = 0
    different_count = 0
    system_parse_errors = 0
    user_parse_errors = 0
    comparison_details: list[dict[str, Any]] = []

    for _, row in merged.iterrows():
        system_raw = row.get("raw_response_system")
        user_raw = row.get("raw_response_user")

        comparison = compare_responses(system_raw, user_raw)

        if not comparison["system_parsed"]:
            system_parse_errors += 1
        if not comparison["user_parsed"]:
            user_parse_errors += 1

        if comparison["identical"]:
            identical_count += 1
        elif comparison["system_parsed"] and comparison["user_parsed"]:
            different_count += 1
            comparison_details.append(
                {
                    "file_name": row["file_name"],
                    "differences": comparison["differences"],
                }
            )

    return {
        "total_compared": total_compared,
        "identical": identical_count,
        "different": different_count,
        "identical_pct": (identical_count / total_compared * 100) if total_compared > 0 else 0,
        "different_pct": (different_count / total_compared * 100) if total_compared > 0 else 0,
        "system_parse_errors": system_parse_errors,
        "user_parse_errors": user_parse_errors,
        "details": comparison_details[:50],  # Limit details to first 50 differences
    }


def main() -> None:
    """CLI entry point for standalone comparison."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare benchmark results between system and user modes")
    parser.add_argument("system_csv", type=Path, help="Path to system mode results CSV")
    parser.add_argument("user_csv", type=Path, help="Path to user mode results CSV")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON file for detailed comparison")

    args = parser.parse_args()

    if not args.system_csv.exists():
        print(f"Error: System CSV not found: {args.system_csv}")
        return

    if not args.user_csv.exists():
        print(f"Error: User CSV not found: {args.user_csv}")
        return

    comparison = compare_benchmark_results(args.system_csv, args.user_csv)

    print("\n=== Benchmark Comparison Results ===")
    print(f"Total compared: {comparison['total_compared']}")
    print(f"Identical: {comparison['identical']} ({comparison['identical_pct']:.1f}%)")
    print(f"Different: {comparison['different']} ({comparison['different_pct']:.1f}%)")
    print(f"System parse errors: {comparison['system_parse_errors']}")
    print(f"User parse errors: {comparison['user_parse_errors']}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed comparison saved to: {args.output}")

    if comparison["details"]:
        print(f"\nFirst {min(5, len(comparison['details']))} differences:")
        for detail in comparison["details"][:5]:
            print(f"  - {detail['file_name']}: {len(detail['differences'])} difference(s)")


if __name__ == "__main__":
    main()
