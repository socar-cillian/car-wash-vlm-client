"""Prompt generation module for VLM inference using Jinja2 templates."""

import csv
import io
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template


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

# Korean-only area name mappings (for v3 template)
INTERIOR_AREAS_KR = {
    "driver_seat": "운전석",
    "passenger_seat": "조수석",
    "cup_holder": "컵홀더",
    "back_seat": "뒷좌석",
}
EXTERIOR_AREAS_KR = {
    "front": "전면",
    "passenger_side": "조수석_방향",
    "driver_side": "운전석_방향",
    "rear": "후면",
}

# Severity level labels
SEVERITY_LABELS = {
    "양호": "Good",
    "보통": "Normal",
    "심각": "Critical",
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


def generate_prompt_template(
    transformed_rows: list[dict], template_path: Path | None = None, template_version: int = 1
) -> str:
    """
    Generate prompt template from transformed guideline data using Jinja2.

    Args:
        transformed_rows: Transformed guideline data
        template_path: Optional custom template path
        template_version: Template version to use (default: 1)

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

    # Prepare template data
    interior_area_display = [INTERIOR_AREAS_DISPLAY[area] for area in INTERIOR_AREAS]
    exterior_area_display = [EXTERIOR_AREAS_DISPLAY[area] for area in EXTERIOR_AREAS]

    # Korean-only area names (for v3 template)
    interior_areas_kr = [INTERIOR_AREAS_KR[area] for area in INTERIOR_AREAS]
    exterior_areas_kr = [EXTERIOR_AREAS_KR[area] for area in EXTERIOR_AREAS]
    all_areas_kr = interior_areas_kr + exterior_areas_kr

    # Generate contamination type enum values (for v1/v2 templates)
    interior_enum_values = [f'"{_sanitize_enum_value(ct)}"' for ct in interior_items]
    exterior_enum_values = [f'"{_sanitize_enum_value(ct)}"' for ct in exterior_items]
    all_enum_values = interior_enum_values + exterior_enum_values

    # Extract contamination type lists (for v3 template)
    interior_contamination_types = list(interior_items.keys())
    exterior_contamination_types = list(exterior_items.keys())

    # Extract severity levels from the data
    severity_levels = set()
    for row in transformed_rows:
        severity_levels.add(row["오염 기준"])
    severity_levels_list = sorted(
        severity_levels, key=lambda x: ["양호", "보통", "심각"].index(x) if x in ["양호", "보통", "심각"] else 999
    )

    template_data = {
        "interior_areas": interior_area_display,
        "exterior_areas": exterior_area_display,
        "interior_area_codes": INTERIOR_AREAS,
        "exterior_area_codes": EXTERIOR_AREAS,
        "all_areas": INTERIOR_AREAS + EXTERIOR_AREAS,
        "interior_items": interior_items,
        "exterior_items": exterior_items,
        "severity_labels": SEVERITY_LABELS,
        "contamination_types": all_enum_values,
        # For v3 template (Korean only)
        "interior_areas_kr": interior_areas_kr,
        "exterior_areas_kr": exterior_areas_kr,
        "all_areas_kr": all_areas_kr,
        "interior_contamination_types": interior_contamination_types,
        "exterior_contamination_types": exterior_contamination_types,
        "severity_levels": severity_levels_list,
    }

    # Load and render template
    if template_path and template_path.exists():
        # Use custom template
        env = Environment(loader=FileSystemLoader(template_path.parent))
        template = env.get_template(template_path.name)
    else:
        # Use default versioned template
        # Path from src/prompts/generator.py to prompts/templates/
        template_dir = Path(__file__).parent.parent.parent / "prompts" / "templates"
        template_filename = f"contamination_classification_v{template_version}.j2"

        if template_dir.exists() and (template_dir / template_filename).exists():
            env = Environment(loader=FileSystemLoader(template_dir))
            template = env.get_template(template_filename)
        else:
            # Fallback to inline template
            print(f"Warning: Template not found at {template_dir / template_filename}, using fallback")
            template = Template(_get_default_template())

    return template.render(**template_data)


def _sanitize_enum_value(value: str) -> str:
    """Convert contamination type to enum-friendly format."""
    return value.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "").lower()


def _get_default_template() -> str:
    """Get default template as string (fallback if template file not found)."""
    return """You are an expert vehicle cleanliness inspector. Analyze the given image and provide a comprehensive \
assessment of vehicle contamination according to the provided guidelines.

# Task Overview

You must analyze the input image and determine:
1. **Image Validity**: Is this a vehicle interior/exterior image, or out-of-distribution (OOD)?
2. **Area Classification**: If valid, is it interior or exterior?
3. **Location-Specific Assessment**: For each relevant vehicle area, identify contamination type and severity

# Vehicle Areas

**Interior Areas**: {{ interior_areas|join(', ') }}
**Exterior Areas**: {{ exterior_areas|join(', ') }}

{% if interior_items %}
# Interior Contamination Guidelines

{% for contamination_type, severities in interior_items.items() %}
## {{ loop.index }}. {{ contamination_type }}

{% for severity in ['양호', '보통', '심각'] %}
{% if severity in severities %}
**{{ severity_labels[severity] }}**:
{{ severities[severity] }}

{% endif %}
{% endfor %}
{% endfor %}
{% endif %}

{% if exterior_items %}
# Exterior Contamination Guidelines

{% for contamination_type, severities in exterior_items.items() %}
## {{ loop.index }}. {{ contamination_type }}

{% for severity in ['양호', '보통', '심각'] %}
{% if severity in severities %}
**{{ severity_labels[severity] }}**:
{{ severities[severity] }}

{% endif %}
{% endfor %}
{% endfor %}
{% endif %}

# Output Format

Provide your assessment in the following JSON format:

```json
{
  "image_type": "interior" | "exterior" | "ood",
  "areas": [
    {
      "area_name": "<one of: {{ all_areas|join(', ') }}>",
      "contamination_detected": true | false,
      "contamination_type": {{ contamination_types|join(' | ') }} | "clean" | "unknown",
      "severity": "good" | "normal" | "critical" | "clean" | "not_applicable"
    }
  ]
}
```

# Assessment Guidelines

1. **Image Type Classification**:
   - If the image does not show a vehicle interior or exterior, classify as "ood"
   - Determine whether the image shows interior or exterior areas

2. **Area-Specific Assessment**:
   - For interior images, assess: {{ interior_area_codes|join(', ') }}
   - For exterior images, assess: {{ exterior_area_codes|join(', ') }}
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
