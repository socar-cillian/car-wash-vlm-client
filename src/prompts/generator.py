"""Prompt generation module for VLM inference using LangChain templates."""

import csv
import io
from pathlib import Path

from langchain_core.prompts import PromptTemplate


# Fixed vehicle areas
INTERIOR_AREAS = ["driver_seat", "passenger_seat", "cup_holder", "back_seat"]
EXTERIOR_AREAS = ["front", "passenger_side", "driver_side", "rear"]

# Severity labels (Level 0 ~ Level 4)
SEVERITY_LABELS = [
    "Level 0",
    "Level 1",
    "Level 2",
    "Level 3",
    "Level 4",
]

# Area normalization + interior/exterior mapping
AREA_MAP = {
    "1. 운전석": ("운전석", "내부"),
    "2. 조수석": ("조수석", "내부"),
    "3. 컵홀더": ("컵홀더", "내부"),
    "4. 뒷좌석": ("뒷좌석", "내부"),
    "5. 전면": ("전면", "외부"),
    "6. 조수석 방향": ("조수석 방향", "외부"),
    "7. 운전석 방향": ("운전석 방향", "외부"),
    "8. 후면": ("후면", "외부"),
}

# Korean area name mappings
INTERIOR_AREAS_KR = {
    "driver_seat": "운전석",
    "passenger_seat": "조수석",
    "cup_holder": "컵홀더",
    "back_seat": "뒷좌석",
}
EXTERIOR_AREAS_KR = {
    "front": "전면",
    "passenger_side": "조수석 방향",
    "driver_side": "운전석 방향",
    "rear": "후면",
}


def parse_car_parts(car_parts_csv: Path) -> dict[str, list[str]]:
    """
    Parse car parts CSV to get sub-areas for each area.

    Input format:
        중분류 (부위),세부 부위

    Args:
        car_parts_csv: Path to car parts CSV file

    Returns:
        Dictionary mapping area name to list of sub-areas
        e.g., {"운전석": ["발판 (매트)", "시트", "도어 포켓 / 바닥"], ...}
    """
    with open(car_parts_csv, encoding="utf-8") as f:
        lines = [line for line in f if line.strip() and not all(c in [",", " ", "\t"] for c in line.strip())]

    csv_data = io.StringIO("".join(lines))
    reader = csv.DictReader(csv_data)

    area_to_sub_areas: dict[str, list[str]] = {}

    for row in reader:
        area_raw = row.get("중분류 (부위)", "").strip()
        sub_area = row.get("세부 부위", "").strip()

        if not area_raw or not sub_area:
            continue

        # Normalize area name
        area_name, _ = AREA_MAP.get(area_raw, (area_raw, "내부"))

        if area_name not in area_to_sub_areas:
            area_to_sub_areas[area_name] = []

        if sub_area not in area_to_sub_areas[area_name]:
            area_to_sub_areas[area_name].append(sub_area)

    return area_to_sub_areas


def parse_guideline(guideline_csv: Path, car_parts_csv: Path | None = None) -> tuple[list[dict], dict[str, list[str]]]:
    """
    Parse guideline CSV and car parts CSV.

    Input format (guideline CSV):
        중분류 (부위),소분류 (오염 항목),Level 0,Level 1,Level 2,Level 3,Level 4

    Input format (car_parts CSV):
        중분류 (부위),세부 부위

    Args:
        guideline_csv: Path to guideline CSV file
        car_parts_csv: Path to car parts CSV file (optional)

    Returns:
        Tuple of (parsed guideline rows, area to sub-areas mapping)
    """
    with open(guideline_csv, encoding="utf-8") as f:
        lines = [line for line in f if line.strip() and not all(c in [",", " ", "\t"] for c in line.strip())]

    csv_data = io.StringIO("".join(lines))
    reader = csv.DictReader(csv_data)

    unique_entries = {}

    for row in reader:
        if not row.get("중분류 (부위)") or not row.get("중분류 (부위)").strip():
            continue

        area_raw = row["중분류 (부위)"].strip()
        area_name, area_type = AREA_MAP.get(area_raw, (area_raw, "내부"))
        contamination_type = row["소분류 (오염 항목)"].strip()

        # Parse all 5 severity levels (Level 0 ~ Level 4)
        for severity_col in SEVERITY_LABELS:
            description = row.get(severity_col, "").strip()
            # Skip if description is empty or "-" (meaning no description for this level)
            if not description or description == "-":
                continue

            key = (area_name, contamination_type, area_type, severity_col)
            if key in unique_entries:
                continue

            unique_entries[key] = {
                "부위": area_name,
                "오염항목": contamination_type,
                "내외부 구분": area_type,
                "오염 기준": severity_col,
                "기준 내용": description,
            }

    # Parse car parts if provided
    area_to_sub_areas: dict[str, list[str]] = {}
    if car_parts_csv and car_parts_csv.exists():
        area_to_sub_areas = parse_car_parts(car_parts_csv)

    return list(unique_entries.values()), area_to_sub_areas


def _create_prompt_template() -> PromptTemplate:
    """
    Create LangChain prompt template for car contamination classification.

    Returns:
        PromptTemplate instance
    """
    template = """당신은 차량 청결 상태를 평가하는 전문가입니다. 주어진 이미지를 분석하고 제공된 가이드라인에 따라 차량의 오염 상태를 평가하세요.

# 작업 개요

입력 이미지를 분석하여 다음을 판단해야 합니다:
1. **이미지 유효성**: 차량 내부/외부 이미지인가, 아니면 관련 없는 이미지(OOD)인가?
2. **영역 분류**: 유효한 경우, 내부인가 외부인가?
3. **위치별 평가**: 각 차량 영역 및 세부 부위에 대해 오염 유형과 심각도를 식별

# 차량 영역 및 세부 부위

{vehicle_areas_table}

# 오염 가이드라인

{contamination_guidelines}

# 유효한 부위별 오염 항목

{contamination_types_table}

# 출력 형식

**중요**: 반드시 유효한 JSON 형식으로만 응답해야 합니다. JSON 앞뒤에 설명 텍스트를 포함하지 마세요.

```json
{{
  "image_type": "내부" | "외부" | "관련없음",
  "areas": [
    {{
      "area_name": "{area_names}",
      "sub_area": "<세부 부위 (위 테이블 참조)>",
      "contaminations": [
        {{
          "contamination_type": "<오염 유형 (위 테이블 참조)>",
          "severity": "{severity_levels}",
          "is_in_guideline": true | false
        }}
      ]
    }}
  ]
}}
```

**주의**:
- 이미지에서 보이는 모든 세부 부위를 평가하세요. 깨끗한 상태는 Level 0으로 기록합니다.
- 한 부위에 여러 오염이 있으면 contaminations 배열에 모두 포함하세요.
- **sub_area는 반드시 포함**해야 합니다. 세부 부위별로 오염을 구분하여 기록하세요.

# 평가 가이드라인

1. **이미지 유형 분류**:
   - 이미지가 차량 내부나 외부를 보여주지 않으면 "관련없음"으로 분류
   - 이미지가 내부 영역인지 외부 영역인지 판단

2. **영역별 평가**:
{area_evaluation_section}
   - **중요**: 이미지에서 명확히 보이는 부위만 평가하세요. 보이지 않는 부분은 추정하지 말고 완전히 생략하세요.
   - 각 세부 부위에 대해 해당하는 오염 항목을 확인
   - 오염이 있으면 가이드라인에 정의된 오염 유형을 먼저 확인
   - 가이드라인에 있는 오염 유형이면 해당 한글 이름을 정확히 사용하고 is_in_guideline을 true로 설정
   - 가이드라인에 없는 새로운 오염 유형이면 구체적으로 설명하고 is_in_guideline을 false로 설정

3. **심각도 수준** (Level 0 ~ Level 4):
   - **Level 0**: 오염 없음 (깨끗한 상태)
{severity_levels_section}

4. **가이드라인에 없는 부위 또는 오염 처리**:
   - **가이드라인에 없는 오염 타입이 발견되면:**
     - contaminations 배열에 추가
     - contamination_type: 오염을 구체적으로 설명
     - severity: 가이드라인의 심각도 기준을 참고하여 가장 적합한 수준 선택
     - is_in_guideline: false로 설정
   - **가이드라인에 없는 세부 부위에서 오염이 발견되면:**
     - sub_area: 세부 부위를 구체적으로 설명
     - contaminations 배열에 오염 정보 포함
     - is_in_guideline: false로 설정

5. **중요 규칙**:
   - **이미지에서 보이는 모든 세부 부위를 평가** - 깨끗하면 Level 0으로 기록
   - **영역이 이미지에서 명확히 보이지 않으면 areas에 포함하지 않음**
   - **sub_area 필드는 필수** - 세부 부위를 명확히 기록
   - 한 부위에 여러 오염이 있으면 contaminations 배열에 모두 나열
   - 가이드라인에 있는 오염은 정확한 한글 이름 사용 필수
   - 유효한 JSON만 출력 - 추가 텍스트나 설명 없음

# 출력 예시

**예시 1: 한 세부 부위에 여러 오염**
```json
{{
  "image_type": "내부",
  "areas": [
    {{
      "area_name": "운전석",
      "sub_area": "시트",
      "contaminations": [
        {{
          "contamination_type": "동물 털",
          "severity": "Level 2",
          "is_in_guideline": true
        }},
        {{
          "contamination_type": "시트 얼룩",
          "severity": "Level 1",
          "is_in_guideline": true
        }}
      ]
    }}
  ]
}}
```

**예시 2: 깨끗한 부위와 오염된 부위가 섞인 경우**
```json
{{
  "image_type": "내부",
  "areas": [
    {{
      "area_name": "운전석",
      "sub_area": "발판 (매트)",
      "contaminations": [
        {{
          "contamination_type": "모래/흙/부스러기",
          "severity": "Level 3",
          "is_in_guideline": true
        }}
      ]
    }},
    {{
      "area_name": "운전석",
      "sub_area": "시트",
      "contaminations": [
        {{
          "contamination_type": "동물 털",
          "severity": "Level 0",
          "is_in_guideline": true
        }},
        {{
          "contamination_type": "시트 얼룩",
          "severity": "Level 0",
          "is_in_guideline": true
        }}
      ]
    }},
    {{
      "area_name": "운전석",
      "sub_area": "도어 포켓 / 바닥",
      "contaminations": [
        {{
          "contamination_type": "쓰레기",
          "severity": "Level 2",
          "is_in_guideline": true
        }}
      ]
    }}
  ]
}}
```

**예시 3: 외부 - 여러 세부 부위 평가**
```json
{{
  "image_type": "외부",
  "areas": [
    {{
      "area_name": "전면",
      "sub_area": "본넷 / 범퍼 / 유리",
      "contaminations": [
        {{
          "contamination_type": "먼지/흙탕물",
          "severity": "Level 2",
          "is_in_guideline": true
        }},
        {{
          "contamination_type": "벌레/새 배설물",
          "severity": "Level 1",
          "is_in_guideline": true
        }}
      ]
    }},
    {{
      "area_name": "전면",
      "sub_area": "유리 / 차체",
      "contaminations": [
        {{
          "contamination_type": "스티커/부착물",
          "severity": "Level 0",
          "is_in_guideline": true
        }}
      ]
    }}
  ]
}}
```

**예시 4: 가이드라인에 없는 오염 또는 세부 부위**
```json
{{
  "image_type": "내부",
  "areas": [
    {{
      "area_name": "운전석",
      "sub_area": "대시보드",
      "contaminations": [
        {{
          "contamination_type": "먼지 적재",
          "severity": "Level 2",
          "is_in_guideline": false
        }}
      ]
    }}
  ]
}}
```

이제 제공된 이미지를 분석하고 지정된 JSON 형식으로 평가를 반환하세요. 기억하세요: JSON만 출력하고 다른 텍스트는 출력하지 마세요."""  # noqa: E501

    return PromptTemplate.from_template(template)


def generate_prompt(
    parsed_rows: list[dict],
    area_to_sub_areas: dict[str, list[str]],
) -> str:
    """
    Generate prompt template from parsed guideline data.

    Args:
        parsed_rows: Parsed guideline data from parse_guideline()
        area_to_sub_areas: Mapping from area to sub-areas from parse_car_parts()

    Returns:
        Generated prompt text
    """
    # Group data by area -> contamination -> severity
    # Structure: {area: {contamination: {severity: description}}}
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

    # Area order
    interior_area_order = [INTERIOR_AREAS_KR[area] for area in INTERIOR_AREAS]
    exterior_area_order = [EXTERIOR_AREAS_KR[area] for area in EXTERIOR_AREAS]

    interior_areas_present = [area for area in interior_area_order if area in interior_guidelines]
    exterior_areas_present = [area for area in exterior_area_order if area in exterior_guidelines]
    all_areas_kr = interior_areas_present + exterior_areas_present

    # Extract severity levels
    severity_levels = set()
    for row in parsed_rows:
        severity_levels.add(row["오염 기준"])

    severity_levels_list = [s for s in SEVERITY_LABELS if s in severity_levels]

    # Build vehicle areas table with sub-areas
    vehicle_areas_parts = []
    vehicle_areas_parts.append("## 내부 영역")
    vehicle_areas_parts.append("| 중분류 (부위) | 세부 부위 |")
    vehicle_areas_parts.append("|---------------|-----------|")

    for area in interior_areas_present:
        sub_areas = area_to_sub_areas.get(area, [])
        if sub_areas:
            sub_areas_str = ", ".join(sub_areas)
            vehicle_areas_parts.append(f"| {area} | {sub_areas_str} |")
        else:
            vehicle_areas_parts.append(f"| {area} | - |")

    vehicle_areas_parts.append("")
    vehicle_areas_parts.append("## 외부 영역")
    vehicle_areas_parts.append("| 중분류 (부위) | 세부 부위 |")
    vehicle_areas_parts.append("|---------------|-----------|")

    for area in exterior_areas_present:
        sub_areas = area_to_sub_areas.get(area, [])
        if sub_areas:
            sub_areas_str = ", ".join(sub_areas)
            vehicle_areas_parts.append(f"| {area} | {sub_areas_str} |")
        else:
            vehicle_areas_parts.append(f"| {area} | - |")

    vehicle_areas_table = "\n".join(vehicle_areas_parts)

    # Build contamination guidelines section (area -> contamination)
    guidelines_parts = []

    if interior_guidelines:
        guidelines_parts.append("## 내부 오염 가이드라인")
        for area in interior_areas_present:
            contaminants = interior_guidelines.get(area, {})
            guidelines_parts.append(f"\n### {area}")
            for contamination_type, severities in contaminants.items():
                guidelines_parts.append(f"#### {contamination_type}")
                for severity in severity_levels_list:
                    if severity in severities:
                        guidelines_parts.append(f"- **{severity}**: {severities[severity]}")

    if exterior_guidelines:
        if guidelines_parts:
            guidelines_parts.append("")
        guidelines_parts.append("## 외부 오염 가이드라인")
        for area in exterior_areas_present:
            contaminants = exterior_guidelines.get(area, {})
            guidelines_parts.append(f"\n### {area}")
            for contamination_type, severities in contaminants.items():
                guidelines_parts.append(f"#### {contamination_type}")
                for severity in severity_levels_list:
                    if severity in severities:
                        guidelines_parts.append(f"- **{severity}**: {severities[severity]}")

    contamination_guidelines = "\n".join(guidelines_parts)

    # Build contamination types table (area -> contamination types)
    contamination_types_parts = []
    contamination_types_parts.append("## 내부")
    contamination_types_parts.append("| 중분류 (부위) | 오염 항목 |")
    contamination_types_parts.append("|---------------|-----------|")

    for area in interior_areas_present:
        contaminants = interior_guidelines.get(area, {})
        types_list = ", ".join(contaminants.keys())
        contamination_types_parts.append(f"| {area} | {types_list} |")

    contamination_types_parts.append("")
    contamination_types_parts.append("## 외부")
    contamination_types_parts.append("| 중분류 (부위) | 오염 항목 |")
    contamination_types_parts.append("|---------------|-----------|")

    for area in exterior_areas_present:
        contaminants = exterior_guidelines.get(area, {})
        types_list = ", ".join(contaminants.keys())
        contamination_types_parts.append(f"| {area} | {types_list} |")

    contamination_types_table = "\n".join(contamination_types_parts)

    # Build area evaluation section
    area_eval_parts = []
    if interior_areas_present:
        area_eval_parts.append(f"   - 내부 이미지의 경우: {', '.join(interior_areas_present)}의 각 세부 부위를 평가")
    if exterior_areas_present:
        area_eval_parts.append(f"   - 외부 이미지의 경우: {', '.join(exterior_areas_present)}의 각 세부 부위를 평가")
    area_evaluation_section = "\n".join(area_eval_parts)

    # Build severity levels section
    severity_levels_for_section = [s for s in severity_levels_list if s != "Level 0"]
    severity_parts = [f"   - **{severity}**" for severity in severity_levels_for_section]
    severity_levels_section = "\n".join(severity_parts)

    # Prepare template values
    area_names_choice = '" | "'.join(all_areas_kr) if all_areas_kr else "부위명"
    severity_levels_value = '" | "'.join(severity_levels_list or ["Level 0"])

    # Create and format prompt
    prompt_template = _create_prompt_template()

    formatted_prompt = prompt_template.format(
        vehicle_areas_table=vehicle_areas_table,
        contamination_guidelines=contamination_guidelines,
        contamination_types_table=contamination_types_table,
        area_names=area_names_choice,
        area_evaluation_section=area_evaluation_section,
        severity_levels_section=severity_levels_section,
        severity_levels=severity_levels_value,
    )

    return formatted_prompt
