"""Prompt generation module for VLM inference using LangChain templates."""

import csv
import io
from pathlib import Path

from langchain_core.prompts import PromptTemplate


# Severity labels (Level 0 ~ Level 4)
SEVERITY_LABELS = [
    "Level 0",
    "Level 1",
    "Level 2",
    "Level 3",
    "Level 4",
]

# Area type mapping (구분 column)
AREA_TYPE_MAP = {
    "외부 부위": "외부",
    "내부 부위": "내부",
    "외부 오염": "외부",
    "내부 오염": "내부",
}


def parse_car_parts(car_parts_csv: Path) -> tuple[dict[str, list[tuple[str, str]]], list[str], list[str]]:
    """
    Parse car parts CSV (v2 format) to get sub-areas for each area type.

    Input format (car_parts_v2.csv):
        구분,검수 부위,세부 포함 영역

    Args:
        car_parts_csv: Path to car parts CSV file

    Returns:
        Tuple of:
        - Dictionary mapping part name to list of (detail, area_type) tuples
          e.g., {"본넷": [("엔진룸 덮개 전체", "외부"), ("전면 엠블럼 주변", "외부")], ...}
        - List of exterior parts in order
        - List of interior parts in order
    """
    with open(car_parts_csv, encoding="utf-8") as f:
        lines = [line for line in f if line.strip() and not all(c in [",", " ", "\t"] for c in line.strip())]

    csv_data = io.StringIO("".join(lines))
    reader = csv.DictReader(csv_data)

    part_to_details: dict[str, list[tuple[str, str]]] = {}
    exterior_parts: list[str] = []
    interior_parts: list[str] = []

    for row in reader:
        area_type_raw = row.get("구분", "").strip()
        part_name = row.get("검수 부위", "").strip()
        detail_area = row.get("세부 포함 영역", "").strip()

        if not area_type_raw or not part_name:
            continue

        area_type = AREA_TYPE_MAP.get(area_type_raw, "내부")

        if part_name not in part_to_details:
            part_to_details[part_name] = []
            if area_type == "외부":
                exterior_parts.append(part_name)
            else:
                interior_parts.append(part_name)

        if detail_area:
            part_to_details[part_name].append((detail_area, area_type))

    return part_to_details, exterior_parts, interior_parts


def parse_guideline(
    guideline_csv: Path, car_parts_csv: Path | None = None
) -> tuple[list[dict], dict[str, list[tuple[str, str]]], list[str], list[str], dict[str, list[str]]]:
    """
    Parse guideline CSV (v5 format) and car parts CSV (v2 format).

    Input format (guideline_v5.csv):
        구분,오염 항목,Level 0,Level 1,Level 2,Level 3,Level 4

    Input format (car_parts_v2.csv):
        구분,검수 부위,세부 포함 영역

    Args:
        guideline_csv: Path to guideline CSV file
        car_parts_csv: Path to car parts CSV file (optional)

    Returns:
        Tuple of:
        - parsed guideline rows
        - part_to_details mapping
        - exterior_parts list
        - interior_parts list
        - valid_levels mapping (contamination -> list of valid levels)
    """
    with open(guideline_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    unique_entries = {}
    valid_levels: dict[str, list[str]] = {}  # Track which levels are valid for each contamination

    for row in rows:
        area_type_raw = row.get("구분", "").strip()
        contamination_type = row.get("오염 항목", "").strip()

        if not area_type_raw or not contamination_type:
            continue

        area_type = AREA_TYPE_MAP.get(area_type_raw, "내부")

        # Track valid levels for this contamination type
        if contamination_type not in valid_levels:
            valid_levels[contamination_type] = []

        # Parse all 5 severity levels (Level 0 ~ Level 4)
        for severity_col in SEVERITY_LABELS:
            description = row.get(severity_col, "").strip()
            # Skip if description is empty or "-" (meaning no description for this level)
            if not description or description == "-":
                continue

            # Replace newlines with space for single-line output
            description = " ".join(description.split())

            # Track this level as valid for this contamination
            if severity_col not in valid_levels[contamination_type]:
                valid_levels[contamination_type].append(severity_col)

            key = (contamination_type, area_type, severity_col)
            if key in unique_entries:
                continue

            unique_entries[key] = {
                "오염항목": contamination_type,
                "내외부 구분": area_type,
                "오염 기준": severity_col,
                "기준 내용": description,
            }

    # Parse car parts if provided
    part_to_details: dict[str, list[tuple[str, str]]] = {}
    exterior_parts: list[str] = []
    interior_parts: list[str] = []

    if car_parts_csv and car_parts_csv.exists():
        part_to_details, exterior_parts, interior_parts = parse_car_parts(car_parts_csv)

    return list(unique_entries.values()), part_to_details, exterior_parts, interior_parts, valid_levels


def _create_prompt_template() -> PromptTemplate:
    """
    Create LangChain prompt template for car contamination classification.

    Returns:
        PromptTemplate instance
    """
    template = """당신은 차량 청결 상태를 평가하는 전문가입니다. 주어진 이미지를 분석하고 **반드시 제공된 가이드라인 내에서만** 차량의 오염 상태를 평가하세요.

# 작업 개요

입력 이미지를 분석하여 다음을 판단해야 합니다:
1. **이미지 유효성**: 차량 내부/외부 이미지인가, 아니면 평가 대상이 아닌 이미지(OOD)인가?
2. **영역 분류**: 유효한 경우, 내부인가 외부인가?
3. **검수 부위 및 오염 평가**: 가이드라인에 정의된 검수 부위와 오염 항목만 평가

# OOD (Out-of-Distribution) 이미지 판별

다음과 같은 이미지는 **차량 세차 검수와 무관한 OOD 이미지**로 분류해야 합니다:

## OOD 판별 기준
- **차량이 없는 이미지**: 풍경, 건물, 사람만 있는 사진 등
- **차량 정비/수리 이미지**: 본넷을 열어서 엔진룸을 수리하는 장면, 워셔액 확인/보충 장면
- **차량 내부에서 바깥을 촬영한 이미지**: 차량 내부에서 유리창을 통해 바깥 풍경을 찍은 사진 (유리가 더러워 보일 수 있으나 OOD로 처리)
- **차량 외부가 아닌 내부 구조물 촬영**: 엔진룸 내부, 트렁크 내부 적재물만 촬영된 경우

## OOD 이미지 출력 형식
```json
{{
  "image_type": "OOD",
  "areas": []
}}
```

# 다중 차량 처리 규칙

**중요**: 이미지에 차량이 여러 대 있는 경우, **화면 중앙에 위치한 가장 크게 보이는 차량**만 평가 대상으로 삼습니다.
- 배경에 있는 다른 차량은 무시
- 부분적으로만 보이는 주변 차량은 무시
- 오직 주요 피사체인 한 대의 차량만 검수 부위와 오염을 평가

# 유리 오염 판별 시 주의사항 (핵심)

**중요**: 차량 유리에 비친 **그림자나 반사**를 오염으로 오인하지 마세요.

## 오염이 아닌 경우 (무시해야 함)
- 나뭇가지나 나뭇잎이 유리에 **비친/반사된** 경우
- 건물, 하늘, 구름이 유리에 반사된 경우
- 촬영자나 주변 사물의 그림자가 유리에 비친 경우
- 햇빛이나 조명이 유리에 반사되어 생긴 빛 반점

## 실제 오염인 경우 (평가해야 함)
- 유리 표면에 **물리적으로 붙어있는** 이물질 (먼지, 물때, 새 배설물 등)
- 유리 표면의 긁힘이나 손상
- 스티커나 부착물

**판별 팁**: 반사/그림자는 유리 너머 배경과 함께 투명하게 겹쳐 보이고, 실제 오염은 유리 표면에 불투명하게 붙어 있습니다.

# 검수 부위 정의

{vehicle_areas_table}

# 오염 가이드라인

{contamination_guidelines}

# 유효한 레벨 제한

**중요**: 아래 표에서 각 오염 항목별로 유효한 레벨만 사용할 수 있습니다. '-'로 표시된 레벨은 정의되지 않았으므로 **절대 해당 레벨로 예측하면 안 됩니다**.

{valid_levels_table}

# 출력 형식

**중요**: 반드시 유효한 JSON 형식으로만 응답해야 합니다. JSON 앞뒤에 설명 텍스트를 포함하지 마세요.

```json
{{
  "image_type": "내부" | "외부" | "OOD",
  "areas": [
    {{
      "area_name": "<검수 부위 (위 테이블 참조)>",
      "contaminations": [
        {{
          "contamination_type": "<오염 항목 (가이드라인 참조)>",
          "severity": "<유효한 레벨만 사용>"
        }}
      ]
    }}
  ]
}}
```

# 평가 규칙

## 1. 이미지 유형 분류
- 이미지가 OOD 판별 기준에 해당하면 "OOD"로 분류하고 areas는 빈 배열로 반환
- 유효한 차량 이미지인 경우 내부/외부 영역을 판단
- 여러 차량이 있으면 가운데 가장 큰 차량만 평가

## 2. 검수 부위 평가
- **가이드라인에 정의된 검수 부위만 평가**
- 외부 이미지: {exterior_parts_list}
- 내부 이미지: {interior_parts_list}
- **이미지에서 명확히 보이는 부위만 평가** - 보이지 않는 부분은 생략

## 3. 오염 항목 평가
- **가이드라인에 정의된 오염 항목만 평가**
- 외부 오염: {exterior_contaminations_list}
- 내부 오염: {interior_contaminations_list}
- **가이드라인에 없는 오염은 무시** - 예측하지 않음

## 4. 심각도 레벨 제한 (핵심 규칙)
- **각 오염 항목별로 정의된 레벨만 사용 가능**
- **'-'로 표시된 레벨은 존재하지 않으므로 절대 사용 금지**
- 예: "새 배설물"은 Level 1이 없으므로 Level 0, 2, 3, 4만 사용 가능
- 예: "스티커"는 Level 1이 없으므로 Level 0, 2, 3, 4만 사용 가능
- 예: "낙엽"은 Level 4가 없으므로 Level 0, 1, 2, 3만 사용 가능

## 5. 중요 금지 사항
- **가이드라인에 없는 검수 부위 예측 금지**
- **가이드라인에 없는 오염 항목 예측 금지**
- **정의되지 않은 레벨(-)로 예측 금지**
- 유효한 JSON만 출력 - 추가 텍스트나 설명 없음

# 출력 예시

**예시 1: OOD - 평가 대상이 아닌 이미지**
```json
{{
  "image_type": "OOD",
  "areas": []
}}
```
※ 차량이 없거나, 본넷 열고 정비 중이거나, 차량 내부에서 바깥을 찍은 경우

**예시 2: 외부 - 본넷에 새 배설물 발견**
```json
{{
  "image_type": "외부",
  "areas": [
    {{
      "area_name": "본넷",
      "contaminations": [
        {{
          "contamination_type": "새 배설물",
          "severity": "Level 2"
        }}
      ]
    }}
  ]
}}
```
※ 새 배설물은 Level 1이 없으므로 Level 2 사용

**예시 3: 외부 - 여러 부위 평가**
```json
{{
  "image_type": "외부",
  "areas": [
    {{
      "area_name": "본넷",
      "contaminations": [
        {{
          "contamination_type": "진흙 및 흙탕물",
          "severity": "Level 2"
        }},
        {{
          "contamination_type": "나무 수액",
          "severity": "Level 0"
        }}
      ]
    }},
    {{
      "area_name": "유리",
      "contaminations": [
        {{
          "contamination_type": "스티커",
          "severity": "Level 0"
        }}
      ]
    }},
    {{
      "area_name": "휠",
      "contaminations": [
        {{
          "contamination_type": "타이어 분진",
          "severity": "Level 1"
        }}
      ]
    }}
  ]
}}
```

**예시 4: 내부 - 시트 오염**
```json
{{
  "image_type": "내부",
  "areas": [
    {{
      "area_name": "시트",
      "contaminations": [
        {{
          "contamination_type": "동물 털",
          "severity": "Level 2"
        }},
        {{
          "contamination_type": "시트 얼룩",
          "severity": "Level 1"
        }}
      ]
    }},
    {{
      "area_name": "매트",
      "contaminations": [
        {{
          "contamination_type": "모래 및 흙",
          "severity": "Level 3"
        }}
      ]
    }}
  ]
}}
```

**예시 5: 내부 - 분실물 발견 (Level 1 없음)**
```json
{{
  "image_type": "내부",
  "areas": [
    {{
      "area_name": "시트",
      "contaminations": [
        {{
          "contamination_type": "분실물",
          "severity": "Level 2"
        }}
      ]
    }}
  ]
}}
```
※ 분실물은 Level 1이 없으므로 Level 2부터 사용

**예시 6: 깨끗한 상태**
```json
{{
  "image_type": "외부",
  "areas": [
    {{
      "area_name": "본넷",
      "contaminations": [
        {{
          "contamination_type": "새 배설물",
          "severity": "Level 0"
        }},
        {{
          "contamination_type": "진흙 및 흙탕물",
          "severity": "Level 0"
        }}
      ]
    }}
  ]
}}
```

이제 제공된 이미지를 분석하고 지정된 JSON 형식으로 평가를 반환하세요. **반드시 가이드라인에 정의된 검수 부위, 오염 항목, 유효한 레벨만 사용하세요.** JSON만 출력하고 다른 텍스트는 출력하지 마세요."""  # noqa: E501

    return PromptTemplate.from_template(template)


def generate_prompt(
    parsed_rows: list[dict],
    part_to_details: dict[str, list[tuple[str, str]]],
    exterior_parts: list[str],
    interior_parts: list[str],
    valid_levels: dict[str, list[str]],
) -> str:
    """
    Generate prompt template from parsed guideline data (v5 format).

    Args:
        parsed_rows: Parsed guideline data from parse_guideline()
        part_to_details: Mapping from part name to list of (detail, area_type) tuples
        exterior_parts: List of exterior part names
        interior_parts: List of interior part names
        valid_levels: Mapping from contamination type to list of valid levels

    Returns:
        Generated prompt text
    """
    # Group contamination guidelines by area type -> contamination -> severity
    # Structure: {contamination: {severity: description}}
    interior_guidelines: dict[str, dict[str, str]] = {}
    exterior_guidelines: dict[str, dict[str, str]] = {}

    for row in parsed_rows:
        contamination_type = row["오염항목"]
        area_type = row["내외부 구분"]
        severity = row["오염 기준"]
        description = row["기준 내용"]

        target_dict = interior_guidelines if area_type == "내부" else exterior_guidelines
        target_dict.setdefault(contamination_type, {})[severity] = description

    # Build vehicle areas table (검수 부위 정의)
    vehicle_areas_parts = []
    vehicle_areas_parts.append("## 외부 부위")
    vehicle_areas_parts.append("| 검수 부위 | 세부 포함 영역 |")
    vehicle_areas_parts.append("|-----------|---------------|")

    for part in exterior_parts:
        details = part_to_details.get(part, [])
        if details:
            detail_str = ", ".join([d[0] for d in details])
            vehicle_areas_parts.append(f"| {part} | {detail_str} |")
        else:
            vehicle_areas_parts.append(f"| {part} | - |")

    vehicle_areas_parts.append("")
    vehicle_areas_parts.append("## 내부 부위")
    vehicle_areas_parts.append("| 검수 부위 | 세부 포함 영역 |")
    vehicle_areas_parts.append("|-----------|---------------|")

    for part in interior_parts:
        details = part_to_details.get(part, [])
        if details:
            detail_str = ", ".join([d[0] for d in details])
            vehicle_areas_parts.append(f"| {part} | {detail_str} |")
        else:
            vehicle_areas_parts.append(f"| {part} | - |")

    vehicle_areas_table = "\n".join(vehicle_areas_parts)

    # Build contamination guidelines section
    guidelines_parts = []

    if exterior_guidelines:
        guidelines_parts.append("## 외부 오염 가이드라인")
        for contamination_type, severities in exterior_guidelines.items():
            guidelines_parts.append(f"\n### {contamination_type}")
            for severity in SEVERITY_LABELS:
                if severity in severities:
                    guidelines_parts.append(f"- **{severity}**: {severities[severity]}")
                else:
                    guidelines_parts.append(f"- **{severity}**: -")

    if interior_guidelines:
        if guidelines_parts:
            guidelines_parts.append("")
        guidelines_parts.append("## 내부 오염 가이드라인")
        for contamination_type, severities in interior_guidelines.items():
            guidelines_parts.append(f"\n### {contamination_type}")
            for severity in SEVERITY_LABELS:
                if severity in severities:
                    guidelines_parts.append(f"- **{severity}**: {severities[severity]}")
                else:
                    guidelines_parts.append(f"- **{severity}**: -")

    contamination_guidelines = "\n".join(guidelines_parts)

    # Build valid levels table
    valid_levels_parts = []
    valid_levels_parts.append("| 오염 항목 | 유효한 레벨 | 사용 불가 레벨 |")
    valid_levels_parts.append("|-----------|-------------|---------------|")

    for contamination_type, levels in valid_levels.items():
        valid_str = ", ".join(levels)
        invalid_levels = [lvl for lvl in SEVERITY_LABELS if lvl not in levels]
        invalid_str = ", ".join(invalid_levels) if invalid_levels else "없음"
        valid_levels_parts.append(f"| {contamination_type} | {valid_str} | {invalid_str} |")

    valid_levels_table = "\n".join(valid_levels_parts)

    # Build parts lists
    exterior_parts_list = ", ".join(exterior_parts) if exterior_parts else "없음"
    interior_parts_list = ", ".join(interior_parts) if interior_parts else "없음"

    # Build contamination lists
    exterior_contaminations = list(exterior_guidelines.keys())
    interior_contaminations = list(interior_guidelines.keys())
    exterior_contaminations_list = ", ".join(exterior_contaminations) if exterior_contaminations else "없음"
    interior_contaminations_list = ", ".join(interior_contaminations) if interior_contaminations else "없음"

    # Create and format prompt
    prompt_template = _create_prompt_template()

    formatted_prompt = prompt_template.format(
        vehicle_areas_table=vehicle_areas_table,
        contamination_guidelines=contamination_guidelines,
        valid_levels_table=valid_levels_table,
        exterior_parts_list=exterior_parts_list,
        interior_parts_list=interior_parts_list,
        exterior_contaminations_list=exterior_contaminations_list,
        interior_contaminations_list=interior_contaminations_list,
    )

    return str(formatted_prompt)
