"""Prompt generation module for VLM inference using LangChain templates."""

import csv
import io
from pathlib import Path

from langchain_core.prompts import PromptTemplate


# Fixed vehicle areas
INTERIOR_AREAS = ["driver_seat", "passenger_seat", "cup_holder", "back_seat"]
EXTERIOR_AREAS = ["front", "passenger_side", "driver_side", "rear"]

# Korean-only area name mappings
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

# Severity level labels (4-level system)
SEVERITY_LABELS = {
    "양호": "Good",
    "보통": "Normal",
    "심각": "Critical",
    "긴급": "Emergency",
}

# Severity labels for v2 guideline (use column names directly)
SEVERITY_LABELS_V2 = [
    "Level 1",
    "Level 2",
    "Level 3",
    "Level 4",
]


def parse_guideline_v2(input_csv: Path) -> list[dict]:
    """
    Parse guideline v2 CSV with new structure (no transformation needed).

    Input format (guideline_v2.csv):
        중분류 (부위),소분류 (오염 항목),현상 유지 (Level 1),관리 권장 (Level 2),
        즉시 조치 (Level 3),긴급/전문 관리 (Level 4)

    Output format (compatible with template system):
        오염항목, 내외부 구분, 오염 기준, 기준 내용, 부위

    Args:
        input_csv: Path to guideline v2 CSV file

    Returns:
        List of dictionaries with parsed data (area-specific rows preserved)
    """
    # Define area normalization + interior/exterior mapping
    area_map = {
        "1. 운전석": ("운전석", "내부"),
        "2. 조수석": ("조수석", "내부"),
        "3. 컵홀더": ("컵홀더", "내부"),
        "4. 뒷좌석": ("뒷좌석", "내부"),
        "5. 전면": ("전면", "외부"),
        "6. 조수석 방향": ("조수석_방향", "외부"),
        "7. 운전석 방향": ("운전석_방향", "외부"),
        "8. 후면": ("후면", "외부"),
    }

    with open(input_csv, encoding="utf-8") as f:
        # Read all lines and skip empty ones
        lines = [line for line in f if line.strip() and not all(c in [",", " ", "\t"] for c in line.strip())]

    # Parse CSV from cleaned lines
    csv_data = io.StringIO("".join(lines))
    reader = csv.DictReader(csv_data)

    # Use dict to track unique combinations
    unique_entries = {}

    for row in reader:
        # Skip empty rows
        if not row.get("중분류 (부위)") or not row.get("중분류 (부위)").strip():
            continue

        area_raw = row["중분류 (부위)"].strip()
        area_name, area_type = area_map.get(area_raw, (area_raw, "내부"))
        contamination_type = row["소분류 (오염 항목)"].strip()

        # Parse all 4 severity levels directly
        for severity_col in SEVERITY_LABELS_V2:
            description = row.get(severity_col, "").strip()
            if description:
                key = (area_name, contamination_type, area_type, severity_col)
                # Keep first occurrence for duplicates per area/type/severity
                if key in unique_entries:
                    continue

                unique_entries[key] = {
                    "부위": area_name,
                    "오염항목": contamination_type,
                    "내외부 구분": area_type,
                    "오염 기준": severity_col,
                    "기준 내용": description,
                }

    return list(unique_entries.values())


def _create_prompt_template() -> PromptTemplate:
    """
    Create LangChain prompt template for car contamination classification.

    Returns:
        PromptTemplate instance
    """
    template = """당신은 차량 청결 상태를 평가하는 전문가입니다. 주어진 이미지를 분석하고 제공된 가이드라인에 따라 차량의 오염 상태를 평가하세요.  # noqa: E501

# 작업 개요

입력 이미지를 분석하여 다음을 판단해야 합니다:
1. **이미지 유효성**: 차량 내부/외부 이미지인가, 아니면 관련 없는 이미지(OOD)인가?
2. **영역 분류**: 유효한 경우, 내부인가 외부인가?
3. **위치별 평가**: 각 차량 영역에 대해 오염 유형과 심각도를 식별

# 차량 영역
{vehicle_areas_section}

# 오염 가이드라인
{contamination_guidelines}

# 유효한 오염 유형
{contamination_types_section}

# 출력 형식

**중요**: 반드시 유효한 JSON 형식으로만 응답해야 합니다. JSON 앞뒤에 설명 텍스트를 포함하지 마세요.

```json
{{
  "image_type": "내부" | "외부" | "관련없음",
  "areas": [
    {{
      "area_name": "{area_names}",
      "contaminations": [
        {{
          "contamination_type": "<오염 유형 (위 목록 참조, 없으면 구체적으로 설명)>",
          "severity": "{severity_levels}",
          "is_in_guideline": true | false
        }}
      ]
    }}
  ]
}}
```

**주의**:
- areas 배열에는 오염이 감지된 영역만 포함하세요. 깨끗한 영역은 생략하세요.
- 한 부위에 여러 오염이 있으면 contaminations 배열에 모두 포함하세요.

# 평가 가이드라인

1. **이미지 유형 분류**:
   - 이미지가 차량 내부나 외부를 보여주지 않으면 "관련없음"으로 분류
   - 이미지가 내부 영역인지 외부 영역인지 판단

2. **영역별 평가**:
{area_evaluation_section}
   - 각 영역에 대해 오염이 있는지 판단
   - **중요**: 이미지에서 명확히 보이는 부위만 평가하세요. 보이지 않는 부분은 추정하지 말고 완전히 생략하세요.
   - 한 부위에 여러 오염이 동시에 존재할 수 있으므로 모든 오염을 contaminations 배열에 포함
   - 오염이 있으면 위에 나열된 오염 유형을 먼저 확인
   - 가이드라인에 있는 오염 유형이면 해당 한글 이름을 정확히 사용하고 is_in_guideline을 true로 설정
   - 가이드라인에 없는 새로운 오염 유형이면 구체적으로 설명하고 is_in_guideline을 false로 설정
   - 가이드라인에 따라 심각도 판단

3. **심각도 수준** (정확한 컬럼명 사용):
{severity_levels_section}

4. **가이드라인에 없는 부위 또는 오염 처리**:
   - **가이드라인에 없는 오염 타입이 발견되면:**
     - contaminations 배열에 추가
     - contamination_type: 오염을 구체적으로 설명 (예: "음료수 얼룩", "껌 자국", "페인트 묻음")
     - severity: 가이드라인의 심각도 기준을 참고하여 가장 적합한 수준 선택
     - is_in_guideline: false로 설정
   - **가이드라인에 없는 부위에서 오염이 발견되면:**
     - area_name: 부위를 구체적으로 설명 (예: "대시보드", "트랭크", "도어 포켓")
     - contaminations 배열에 오염 정보 포함
     - is_in_guideline: false로 설정
   - 이 정보는 향후 가이드라인 업데이트에 사용됩니다

5. **중요 규칙**:
   - **오염이 감지된 영역만 areas 배열에 포함** - 깨끗한 영역은 생략
   - **영역이 이미지에서 명확히 보이지 않으면 areas에 포함하지 않음** - 보이지 않는 부분은 추정하지 말 것
   - 한 부위에 여러 오염이 있으면 contaminations 배열에 모두 나열
   - 가이드라인에 있는 오염은 정확한 한글 이름 사용 필수
   - 유효한 JSON만 출력 - 추가 텍스트나 설명 없음
   - **area_name 값은 반드시 한글로 사용** (예: "{area_name_examples}")  # noqa: E501

# 출력 예시

**예시 1: 한 부위에 여러 오염 (가장 일반적인 케이스)**
```json
{{
  "image_type": "내부",
  "areas": [
    {{
      "area_name": "{example_interior_area_1}",
      "contaminations": [
        {{
          "contamination_type": "{example_interior_type_primary}",
          "severity": "{example_severity_secondary}",
          "is_in_guideline": true
        }},
        {{
          "contamination_type": "{example_interior_type_secondary}",
          "severity": "{example_severity_primary}",
          "is_in_guideline": true
        }}
      ]
    }},
    {{
      "area_name": "{example_interior_area_2}",
      "contaminations": [
        {{
          "contamination_type": "{example_interior_type_secondary}",
          "severity": "{example_severity_tertiary}",
          "is_in_guideline": true
        }}
      ]
    }}
  ]
}}
```

**예시 2: 가이드라인에 없는 오염 타입**
```json
{{
  "image_type": "내부",
  "areas": [
    {{
      "area_name": "{example_interior_area_1}",
      "contaminations": [
        {{
          "contamination_type": "{example_interior_type_primary}",
          "severity": "{example_severity_secondary}",
          "is_in_guideline": true
        }}
      ]
    }},
    {{
      "area_name": "뒷좌석",
      "contaminations": [
        {{
          "contamination_type": "새로운 오염 유형",
          "severity": "{example_severity_tertiary}",
          "is_in_guideline": false
        }}
      ]
    }}
  ]
}}
```

**예시 3: 가이드라인에 없는 부위**
```json
{{
  "image_type": "내부",
  "areas": [
    {{
      "area_name": "대시보드",
      "contaminations": [
        {{
          "contamination_type": "먼지 적재",
          "severity": "Level 2",
          "is_in_guideline": false
        }}
      ]
    }},
    {{
      "area_name": "트랭크",
      "contaminations": [
        {{
          "contamination_type": "흙/모래",
          "severity": "Level 3",
          "is_in_guideline": false
        }}
      ]
    }}
  ]
}}
```

**예시 4: 완전히 깨끗함**
```json
{{
  "image_type": "내부",
  "areas": []
}}
```

이제 제공된 이미지를 분석하고 지정된 JSON 형식으로 평가를 반환하세요. 기억하세요: JSON만 출력하고 다른 텍스트는 출력하지 마세요."""  # noqa: E501

    return PromptTemplate.from_template(template)


def generate_prompt_template(
    parsed_rows: list[dict], template_path: Path | None = None, template_version: int = 1
) -> str:
    """
    Generate prompt template from parsed guideline data using LangChain.

    Args:
        parsed_rows: Parsed guideline data from parse_guideline_v2()
        template_path: Optional custom template path (deprecated, kept for compatibility)
        template_version: Template version to use (deprecated, kept for compatibility)

    Returns:
        Generated prompt text
    """
    # Group data by area -> contamination -> severity
    interior_guidelines: dict[str, dict[str, dict[str, str]]] = {}
    exterior_guidelines: dict[str, dict[str, dict[str, str]]] = {}

    for row in parsed_rows:
        area_name = row.get("부위") or ""
        contamination_type = row["오염항목"]
        area_type = row["내외부 구분"]
        severity = row["오염 기준"]
        description = row["기준 내용"]

        target_dict = interior_guidelines if area_type == "내부" else exterior_guidelines
        area_bucket = target_dict.setdefault(area_name, {})
        area_bucket.setdefault(contamination_type, {})[severity] = description

    # Area order (Korean display)
    interior_area_order = [INTERIOR_AREAS_KR[area] for area in INTERIOR_AREAS]
    exterior_area_order = [EXTERIOR_AREAS_KR[area] for area in EXTERIOR_AREAS]

    interior_areas_present = [area for area in interior_area_order if area in interior_guidelines]
    exterior_areas_present = [area for area in exterior_area_order if area in exterior_guidelines]
    all_areas_kr = interior_areas_present + exterior_areas_present

    # Extract severity levels from the data (use exact column names)
    severity_levels = set()
    for row in parsed_rows:
        severity_levels.add(row["오염 기준"])

    # Sort by the order in SEVERITY_LABELS_V2
    severity_levels_list = []
    for severity in SEVERITY_LABELS_V2:
        if severity in severity_levels:
            severity_levels_list.append(severity)

    # Build vehicle areas section
    vehicle_areas_parts = []
    if interior_areas_present:
        vehicle_areas_parts.append(f"**내부 영역**: {', '.join(interior_areas_present)}")
    if exterior_areas_present:
        vehicle_areas_parts.append(f"**외부 영역**: {', '.join(exterior_areas_present)}")
    vehicle_areas_section = "\n".join(vehicle_areas_parts)

    # Build contamination guidelines section
    guidelines_parts = []

    if interior_guidelines:
        guidelines_parts.append("## 내부 오염 가이드라인")
        for area in interior_areas_present:
            contaminants = interior_guidelines.get(area, {})
            guidelines_parts.append(f"\n### {area}")
            for contamination_type, severities in contaminants.items():
                guidelines_parts.append(f"- **{contamination_type}**")
                for severity in severity_levels_list:
                    if severity in severities:
                        guidelines_parts.append(f"  - **{severity}**: {severities[severity]}")

    if exterior_guidelines:
        if guidelines_parts:
            guidelines_parts.append("")
        guidelines_parts.append("## 외부 오염 가이드라인")
        for area in exterior_areas_present:
            contaminants = exterior_guidelines.get(area, {})
            guidelines_parts.append(f"\n### {area}")
            for contamination_type, severities in contaminants.items():
                guidelines_parts.append(f"- **{contamination_type}**")
                for severity in severity_levels_list:
                    if severity in severities:
                        guidelines_parts.append(f"  - **{severity}**: {severities[severity]}")

    contamination_guidelines = "\n".join(guidelines_parts)

    # Build contamination types section
    contamination_types_parts = []
    if interior_guidelines:
        contamination_types_parts.append("**내부:**")
        for area in interior_areas_present:
            types_list = ", ".join(interior_guidelines[area].keys())
            contamination_types_parts.append(f"- {area}: {types_list}")
        contamination_types_parts.append("- 깨끗함 (오염 없음)")

    if exterior_guidelines:
        if contamination_types_parts:
            contamination_types_parts.append("")
        contamination_types_parts.append("**외부:**")
        for area in exterior_areas_present:
            types_list = ", ".join(exterior_guidelines[area].keys())
            contamination_types_parts.append(f"- {area}: {types_list}")
        contamination_types_parts.append("- 깨끗함 (오염 없음)")

    contamination_types_section = "\n".join(contamination_types_parts)

    # Build area evaluation section
    area_eval_parts = []
    if interior_areas_present:
        area_eval_parts.append(f"   - 내부 이미지의 경우, 다음을 모두 평가: {', '.join(interior_areas_present)}")
    if exterior_areas_present:
        area_eval_parts.append(f"   - 외부 이미지의 경우, 다음을 모두 평가: {', '.join(exterior_areas_present)}")
    area_evaluation_section = "\n".join(area_eval_parts)

    # Build severity levels section (use exact column names)
    severity_samples = severity_levels_list or ["Level 1"]
    severity_parts = []
    for severity in severity_samples:
        severity_parts.append(f"   - **{severity}**")
    severity_levels_section = "\n".join(severity_parts)

    # Prepare area name strings for template examples
    area_names_choice = '" | "'.join(all_areas_kr) if all_areas_kr else "부위명"
    area_name_examples = '", "'.join(all_areas_kr) if all_areas_kr else '운전석", "조수석'

    # Example contamination types bound to actual areas (fallback to generic label)
    def _first_area_with_data(area_order: list[str], guidelines: dict[str, dict[str, dict[str, str]]]) -> str:
        for area in area_order:
            if guidelines.get(area):
                return area
        return "내부 부위"

    def _first_two_types(area: str, guidelines: dict[str, dict[str, dict[str, str]]]) -> tuple[str, str]:
        types_list = list(guidelines.get(area, {}).keys())
        if not types_list:
            return ("오염항목", "오염항목")
        if len(types_list) == 1:
            return (types_list[0], types_list[0])
        return (types_list[0], types_list[1])

    example_interior_area_1 = _first_area_with_data(interior_areas_present, interior_guidelines)
    example_interior_area_2 = (
        _first_area_with_data(interior_areas_present[1:], interior_guidelines)
        if len(interior_areas_present) > 1
        else example_interior_area_1
    )
    example_interior_type_primary, example_interior_type_secondary = _first_two_types(
        example_interior_area_1, interior_guidelines
    )

    # Pick example severities to keep JSON examples in sync with the CSV
    example_severity_primary = severity_samples[0]
    example_severity_secondary = severity_samples[1] if len(severity_samples) > 1 else example_severity_primary
    example_severity_tertiary = severity_samples[2] if len(severity_samples) > 2 else example_severity_secondary
    severity_levels_value = '" | "'.join(severity_levels_list or severity_samples)

    # Create template and format with data
    prompt_template = _create_prompt_template()

    # Format the prompt
    formatted_prompt = prompt_template.format(
        vehicle_areas_section=vehicle_areas_section,
        contamination_guidelines=contamination_guidelines,
        contamination_types_section=contamination_types_section,
        area_names=area_names_choice,
        area_name_examples=area_name_examples,
        area_evaluation_section=area_evaluation_section,
        severity_levels_section=severity_levels_section,
        severity_levels=severity_levels_value,
        example_interior_area_1=example_interior_area_1,
        example_interior_area_2=example_interior_area_2,
        example_interior_type_primary=example_interior_type_primary,
        example_interior_type_secondary=example_interior_type_secondary,
        example_severity_primary=example_severity_primary,
        example_severity_secondary=example_severity_secondary,
        example_severity_tertiary=example_severity_tertiary,
    )

    return formatted_prompt
