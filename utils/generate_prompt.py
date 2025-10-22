#!/usr/bin/env python3
"""
Generate VLM prompt from guideline CSV.

This script:
1. Reads a guideline CSV with contamination criteria
2. Transforms the CSV into a normalized format
3. Generates a prompt template based on the guidelines
4. Saves the prompt to a versioned file
"""

import argparse
import csv
import sys
from pathlib import Path


# Fixed vehicle areas
INTERIOR_AREAS = ["driver_seat", "passenger_seat", "cup_holder", "back_seat"]
EXTERIOR_AREAS = ["front", "passenger_side", "driver_side", "rear"]

# Area name mappings for display
INTERIOR_AREAS_DISPLAY = {
    "driver_seat": "운전석 (Driver Seat)",
    "passenger_seat": "조수석 (Passenger Seat)",
    "cup_holder": "컵홀더 (Cup Holder)",
    "back_seat": "뒷좌석 (Back Seat)",
}
EXTERIOR_AREAS_DISPLAY = {
    "front": "전면 (Front)",
    "passenger_side": "조수석 방향 (Passenger Side)",
    "driver_side": "운전석 방향 (Driver Side)",
    "rear": "후면 (Rear)",
}


def transform_guideline_csv(input_csv: Path) -> list[dict]:
    """
    Transform guideline CSV from wide format to long format.

    Input format:
        오염항목, 내외부 구분, 양호 (Good), 보통 (Normal), 심각 (Critical)

    Output format:
        오염항목, 내외부 구분, 오염 기준, 기준 내용

    Args:
        input_csv: Path to input CSV file

    Returns:
        List of dictionaries with transformed data
    """
    transformed_rows = []

    with open(input_csv, encoding="utf-8") as f:
        # Read all lines and skip empty ones
        lines = [line for line in f if line.strip() and not all(c in [",", " ", "\t"] for c in line.strip())]

    # Parse CSV from cleaned lines
    import io

    csv_data = io.StringIO("".join(lines))
    reader = csv.DictReader(csv_data)

    for row in reader:
        # Skip empty rows
        if not row.get("오염항목") or not row.get("오염항목").strip():
            continue

        contamination_type = row["오염항목"].strip()
        # Support both column name formats
        area_type = row.get("내외부 구분", row.get("내/외부 구분", "")).strip()

        # Transform each severity level into a separate row
        for severity, column_name in [
            ("양호", "양호 (Good)"),
            ("보통", "보통 (Normal)"),
            ("심각", "심각 (Critical)"),
        ]:
            description = row.get(column_name, "").strip()
            if description:
                transformed_rows.append(
                    {
                        "오염항목": contamination_type,
                        "내외부 구분": area_type,
                        "오염 기준": severity,
                        "기준 내용": description,
                    }
                )

    return transformed_rows


def save_transformed_guideline(rows: list[dict], output_path: Path):
    """
    Save transformed guideline to CSV.

    Args:
        rows: Transformed guideline rows
        output_path: Path to save the transformed CSV
    """
    fieldnames = ["오염항목", "내외부 구분", "오염 기준", "기준 내용"]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Transformed guideline saved to: {output_path}")


def generate_prompt_template(transformed_rows: list[dict]) -> str:
    """
    Generate prompt template from transformed guideline data.

    Args:
        transformed_rows: Transformed guideline data

    Returns:
        Generated prompt text
    """
    # Group data by contamination type and area
    interior_items = {}
    exterior_items = {}

    for row in transformed_rows:
        contamination_type = row["오염항목"]
        area_type = row["내외부 구분"]
        severity = row["오염 기준"]
        description = row["기준 내용"]

        target_dict = interior_items if area_type == "내부" else exterior_items

        if contamination_type not in target_dict:
            target_dict[contamination_type] = {}

        target_dict[contamination_type][severity] = description

    # Generate prompt
    prompt_parts = []

    # Header
    prompt_parts.append(
        """You are an expert vehicle cleanliness inspector. Analyze the given image and provide a comprehensive \
assessment of vehicle contamination according to the provided guidelines.

# Task Overview

You must analyze the input image and determine:
1. **Image Validity**: Is this a vehicle interior/exterior image, or out-of-distribution (OOD)?
2. **Area Classification**: If valid, is it interior or exterior?
3. **Location-Specific Assessment**: For each relevant vehicle area, identify contamination type and severity
4. **Overall Assessment**: Provide a summary of findings

# Vehicle Areas

**Interior Areas**: """
        + ", ".join([INTERIOR_AREAS_DISPLAY[area] for area in INTERIOR_AREAS])
        + """
**Exterior Areas**: """
        + ", ".join([EXTERIOR_AREAS_DISPLAY[area] for area in EXTERIOR_AREAS])
        + """

"""
    )

    # Interior contamination guidelines
    if interior_items:
        prompt_parts.append("# Interior Contamination Guidelines\n\n")
        for idx, (contamination_type, severities) in enumerate(interior_items.items(), 1):
            prompt_parts.append(f"## {idx}. {contamination_type}\n\n")
            for severity in ["양호", "보통", "심각"]:
                if severity in severities:
                    severity_label = {"양호": "Good", "보통": "Normal", "심각": "Critical"}[severity]
                    prompt_parts.append(f"**{severity_label}**:\n{severities[severity]}\n\n")

    # Exterior contamination guidelines
    if exterior_items:
        prompt_parts.append("# Exterior Contamination Guidelines\n\n")
        for idx, (contamination_type, severities) in enumerate(exterior_items.items(), 1):
            prompt_parts.append(f"## {idx}. {contamination_type}\n\n")
            for severity in ["양호", "보통", "심각"]:
                if severity in severities:
                    severity_label = {"양호": "Good", "보통": "Normal", "심각": "Critical"}[severity]
                    prompt_parts.append(f"**{severity_label}**:\n{severities[severity]}\n\n")

    # Generate contamination type enum values
    interior_enum_values = [f'"{_sanitize_enum_value(ct)}"' for ct in interior_items]
    exterior_enum_values = [f'"{_sanitize_enum_value(ct)}"' for ct in exterior_items]
    all_enum_values = interior_enum_values + exterior_enum_values

    # Output format instructions
    prompt_parts.append(
        f"""# Output Format

Provide your assessment in the following JSON format:

```json
{{
  "image_type": "interior" | "exterior" | "ood",
  "areas": [
    {{
      "area_name": "<one of: {', '.join(INTERIOR_AREAS + EXTERIOR_AREAS)}>",
      "contamination_detected": true | false,
      "contamination_type": {' | '.join(all_enum_values)} | "clean" | "unknown",
      "severity": "good" | "normal" | "critical" | "clean" | "not_applicable"
    }}
  ]
}}
```

# Assessment Guidelines

1. **Image Type Classification**:
   - If the image does not show a vehicle interior or exterior, classify as "ood"
   - Determine whether the image shows interior or exterior areas

2. **Area-Specific Assessment**:
   - For interior images, assess: {', '.join(INTERIOR_AREAS)}
   - For exterior images, assess: {', '.join(EXTERIOR_AREAS)}
   - For each area, determine if contamination is present
   - If contamination exists, identify the type and severity according to guidelines

3. **Severity Levels**:
   - **good** (양호): Minor contamination, acceptable condition
   - **normal** (보통): Moderate contamination, cleaning recommended
   - **critical** (심각): Severe contamination, immediate cleaning required
   - **clean**: No contamination detected
   - **not_applicable**: Area not visible or image is OOD

# Important Notes

- Be thorough and systematic in your analysis
- Use the provided guidelines strictly for severity classification
- If an area is not visible in the image, mark severity as "not_applicable"
- Only return the JSON output without additional text

Now, please analyze the provided image and return your assessment in the specified JSON format.
"""
    )

    return "".join(prompt_parts)


def _sanitize_enum_value(value: str) -> str:
    """Convert contamination type to enum-friendly format."""
    return value.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "").lower()


def main():
    """Main function to generate prompt from guideline CSV."""
    # Get the project root directory
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Generate VLM prompt from guideline CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python utils/generate_prompt.py \\
    --guideline guideline/version1.csv \\
    --output prompts/car_contamination_classification_prompt_v1.txt \\
    --save-transformed
        """,
    )
    parser.add_argument(
        "--guideline",
        type=Path,
        required=True,
        help="Path to guideline CSV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for generated prompt (if not specified, auto-generates version)",
    )
    parser.add_argument(
        "--save-transformed",
        action="store_true",
        help="Save transformed guideline CSV to guideline directory",
    )

    args = parser.parse_args()

    # Make paths absolute if they're relative
    guideline_path = args.guideline if args.guideline.is_absolute() else project_root / args.guideline

    if not guideline_path.exists():
        print(f"Error: Guideline file not found: {guideline_path}", file=sys.stderr)
        sys.exit(1)

    try:
        print("=" * 60)
        print("Step 1: Transforming guideline CSV")
        print("=" * 60)
        transformed_rows = transform_guideline_csv(guideline_path)
        print(f"Transformed {len(transformed_rows)} guideline entries")

        # Save transformed guideline if requested
        if args.save_transformed:
            transformed_output = guideline_path.parent / f"{guideline_path.stem}_transformed.csv"
            save_transformed_guideline(transformed_rows, transformed_output)

        print("\n" + "=" * 60)
        print("Step 2: Generating prompt template")
        print("=" * 60)
        prompt = generate_prompt_template(transformed_rows)
        print("Prompt generated successfully")

        # Determine output path
        if args.output:
            output_path = args.output if args.output.is_absolute() else project_root / args.output
        else:
            # Auto-generate version number
            prompts_dir = project_root / "prompts"
            prompts_dir.mkdir(exist_ok=True)
            existing_versions = list(prompts_dir.glob("car_contamination_classification_prompt_v*.txt"))
            if existing_versions:
                # Extract version numbers
                versions = []
                for p in existing_versions:
                    try:
                        version = int(p.stem.split("_v")[1])
                        versions.append(version)
                    except (IndexError, ValueError):
                        continue
                next_version = max(versions) + 1 if versions else 1
            else:
                next_version = 1
            output_path = prompts_dir / f"car_contamination_classification_prompt_v{next_version}.txt"

        # Save prompt
        print("\n" + "=" * 60)
        print("Step 3: Saving prompt")
        print("=" * 60)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"Prompt saved to: {output_path}")

        print("\n" + "=" * 60)
        print("✓ All steps completed successfully!")
        print("=" * 60)
        print("\nGenerated files:")
        if args.save_transformed:
            print(f"  - Transformed guideline: {transformed_output}")
        print(f"  - Prompt template: {output_path}")

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
