"""Filter and extract unknown contamination types from inference results.

This script helps identify contamination types that are not in the guideline
so they can be added to future guideline updates.
"""

import json
from pathlib import Path

import pandas as pd


def filter_unknown_contaminations(results_csv: Path, output_csv: Path | None = None) -> pd.DataFrame:
    """
    Filter inference results to find contamination types not in guideline.

    Args:
        results_csv: Path to inference results CSV file
        output_csv: Optional path to save filtered results

    Returns:
        DataFrame with unknown contamination entries
    """
    # Read results
    df = pd.read_csv(results_csv)

    # Parse JSON responses and extract unknown contaminations
    unknown_entries = []

    for idx, row in df.iterrows():
        try:
            # Parse the response JSON
            response = json.loads(row.get("response", "{}"))

            # Check each area for unknown contaminations
            for area in response.get("areas", []):
                if not area.get("is_in_guideline", True):
                    unknown_entries.append(
                        {
                            "file_name": row.get("file_name", ""),
                            "image_type": response.get("image_type", ""),
                            "area_name": area.get("area_name", ""),
                            "contamination_type": area.get("contamination_type", ""),
                            "severity": area.get("severity", ""),
                            "timestamp": row.get("timestamp", ""),
                        }
                    )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to parse row {idx}: {e}")
            continue

    # Create DataFrame
    unknown_df = pd.DataFrame(unknown_entries)

    if len(unknown_df) > 0:
        print(f"\n✓ Found {len(unknown_df)} unknown contamination entries")

        # Show summary by contamination type
        print("\nSummary by contamination type:")
        summary = unknown_df.groupby(["contamination_type", "severity"]).size().reset_index(name="count")
        print(summary.to_string(index=False))

        # Save if output path provided
        if output_csv:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            unknown_df.to_csv(output_csv, index=False, encoding="utf-8")
            print(f"\n✓ Saved unknown contaminations to: {output_csv}")
    else:
        print("\n✓ No unknown contamination types found (all match guideline)")

    return unknown_df


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.utils.filter_unknown_contaminations <results_csv> [output_csv]")
        sys.exit(1)

    results_csv = Path(sys.argv[1])
    output_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    if not results_csv.exists():
        print(f"Error: File not found: {results_csv}")
        sys.exit(1)

    filter_unknown_contaminations(results_csv, output_csv)
