"""Batch inference module for vehicle contamination classification."""

import csv
import json
import time
from pathlib import Path
from typing import Optional

from src.api import VLMClient


# Fixed vehicle areas (Korean)
INTERIOR_AREAS = ["운전석", "조수석", "컵홀더", "뒷좌석"]
EXTERIOR_AREAS = ["전면", "조수석_방향", "운전석_방향", "후면"]
ALL_AREAS = INTERIOR_AREAS + EXTERIOR_AREAS


def load_prompt(prompt_path: Path) -> str:
    """Load prompt from file."""
    with open(prompt_path, encoding="utf-8") as f:
        return f.read()


def _normalize_inference_result(result: dict, image_name: str) -> dict:
    """
    Normalize and validate inference result to ensure consistent format.
    Converts Korean values to English for CSV output while preserving JSON keys.

    Args:
        result: Raw inference result dictionary
        image_name: Name of the image file for logging

    Returns:
        Normalized result dictionary with English values
    """
    # Mapping of Korean area names to English codes for CSV
    area_name_mapping = {
        "운전석": "driver_seat",
        "조수석": "passenger_seat",
        "컵홀더": "cup_holder",
        "뒷좌석": "back_seat",
        "전면": "front",
        "조수석_방향": "passenger_side",
        "운전석_방향": "driver_side",
        "후면": "rear",
    }

    # Mapping of Korean contamination types to English codes for CSV
    contamination_type_mapping = {
        # Template v3 Korean codes
        "오염_때_부스러기": "dirt_debris",
        "동물_털": "animal_hair",
        "시트_얼룩": "seat_stain",
        "컵홀더_오염": "cup_holder_dirt",
        "쓰레기_내부": "trash_interior",
        "담배": "cigarette",
        "내부위생불량": "interior_hygiene",
        "물때_오염": "water_stain",
        "특수_오염_벌레_새배설물": "special_contamination",
        "휠_타이어_오염": "wheel_tire_dirt",
        "외부위생불량_유리_경고장": "exterior_hygiene",
        "깨끗함": "clean",
        # Legacy variations
        "컵홀더": "cup_holder_dirt",
        "특수_오염_외부": "special_contamination",
        "특수오염": "special_contamination",
        "물때": "water_stain",
        "휠타이어": "wheel_tire_dirt",
        "오염": "dirt_debris",
        "동물털": "animal_hair",
        "시트얼룩": "seat_stain",
        "쓰레기": "trash_interior",
        "외부위생불량": "exterior_hygiene",
        # Mixed Korean/English
        "특수_오염_(외부)": "special_contamination",
        "쓰레기_(내부)": "trash_interior",
        "물때/오염_(외부)": "water_stain",
        "오염/때_(부스러기)": "dirt_debris",
    }

    # Mapping of Korean image_type to English
    image_type_mapping = {
        "내부": "interior",
        "외부": "exterior",
        "관련없음": "ood",
    }

    # Mapping of Korean severity to English
    severity_mapping = {
        "양호": "good",
        "보통": "normal",
        "심각": "critical",
        "깨끗함": "clean",
        "해당없음": "not_applicable",
    }

    # Normalize image_type
    if "image_type" not in result:
        print(f"Warning: Missing image_type for {image_name}, defaulting to 'ood'")
        result["image_type"] = "ood"
    elif result["image_type"] in image_type_mapping:
        original_type = result["image_type"]
        result["image_type"] = image_type_mapping[original_type]
        print(f"Normalized image_type for {image_name}: {original_type} -> {result['image_type']}")

    # Ensure areas array exists
    if "areas" not in result or not isinstance(result["areas"], list):
        print(f"Warning: Missing or invalid areas for {image_name}")
        result["areas"] = []
        return result

    # Normalize each area
    for area in result["areas"]:
        # Normalize area_name (v3 template uses Korean area names)
        if "area_name" in area:
            original_area = area["area_name"]
            if original_area in area_name_mapping:
                area["area_name"] = area_name_mapping[original_area]
                print(f"Normalized area_name for {image_name}: {original_area} -> {area['area_name']}")

        # Normalize contamination_type
        if "contamination_type" in area:
            original_type = area["contamination_type"]

            # Check if it's in Korean or needs mapping
            if original_type in contamination_type_mapping:
                area["contamination_type"] = contamination_type_mapping[original_type]
                print(
                    f"Normalized contamination_type for {image_name}: {original_type} -> {area['contamination_type']}"
                )
            elif original_type and original_type not in ["clean", "unknown"]:
                # Unknown contamination type
                print(f"Warning: Unknown contamination_type '{original_type}' for {image_name}, setting to 'unknown'")
                area["contamination_type"] = "unknown"

        # Normalize severity
        if "severity" in area:
            original_severity = area["severity"]
            if original_severity in severity_mapping:
                area["severity"] = severity_mapping[original_severity]
                print(f"Normalized severity for {image_name}: {original_severity} -> {area['severity']}")
        else:
            area["severity"] = "not_applicable"

        # Note: contamination_detected field is optional in v3 template
        # If present, keep it; if not, don't add it (v3 only includes contaminated areas)

    return result


def process_image(
    client: VLMClient,
    image_path: Path,
    prompt: str,
    max_tokens: int = 1000,
    temperature: float = 0.0,
) -> dict:
    """
    Process a single image with VLM inference.

    Args:
        client: VLM client instance
        image_path: Path to image file
        prompt: Prompt text
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Dictionary containing inference results and metadata
    """
    start_time = time.time()

    try:
        response = client.query(
            image_input=str(image_path),
            prompt_input=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        latency = time.time() - start_time

        # Extract response content
        if "choices" in response and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"]

            # Parse JSON response
            try:
                # Try to extract JSON from markdown code blocks
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    json_str = content[json_start:json_end].strip()
                elif "```" in content:
                    json_start = content.find("```") + 3
                    json_end = content.find("```", json_start)
                    json_str = content[json_start:json_end].strip()
                else:
                    json_str = content.strip()

                result = json.loads(json_str)

                # Note: Normalization disabled - keep Korean values as-is
                # result = _normalize_inference_result(result, image_path.name)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON for {image_path.name}: {e}")
                print(f"Raw content: {content[:200]}...")
                result = {"error": "json_parse_error", "raw_content": content}

            return {
                "success": True,
                "result": result,
                "latency": latency,
                "raw_response": response,
            }
        else:
            return {
                "success": False,
                "error": "no_choices_in_response",
                "latency": latency,
                "raw_response": response,
            }

    except Exception as e:
        latency = time.time() - start_time
        return {
            "success": False,
            "error": str(e),
            "latency": latency,
        }


def create_csv_row(image_path: Path, inference_result: dict, model_name: str) -> dict:
    """
    Create a CSV row from inference result.

    Args:
        image_path: Path to image file
        inference_result: Inference result dictionary
        model_name: Name of the model used

    Returns:
        Dictionary representing a CSV row
    """
    row = {
        "image_name": image_path.name,
        "model": model_name,
        "latency_seconds": f"{inference_result['latency']:.3f}",
        "success": inference_result["success"],
    }

    if not inference_result["success"]:
        row["error"] = inference_result.get("error", "unknown_error")
        # Add empty columns for all areas
        for area in ALL_AREAS:
            row[f"{area}_contamination_type"] = ""
            row[f"{area}_severity"] = ""
        row["image_type"] = ""
        return row

    result = inference_result.get("result", {})

    # Add image type
    row["image_type"] = result.get("image_type", "")

    # Create a lookup dict for areas
    # In v3 template, areas array only contains contaminated areas
    areas_dict = {}
    for area_info in result.get("areas", []):
        area_name = area_info.get("area_name", "")
        areas_dict[area_name] = area_info

    # Add columns for each area
    # For v3 template: if area not in areas_dict, it means it's clean
    for area in ALL_AREAS:
        if area in areas_dict:
            area_info = areas_dict[area]
            contamination_type = area_info.get("contamination_type", "")
            severity = area_info.get("severity", "")
        else:
            # Area not in results means it's clean (for v3 template)
            # For legacy templates, empty string is appropriate
            contamination_type = "청결" if result.get("areas", None) is not None else ""
            severity = "청결" if result.get("areas", None) is not None else ""

        row[f"{area}_contamination_type"] = contamination_type
        row[f"{area}_severity"] = severity

    return row


def load_ground_truth(gt_csv_path: Path) -> dict[str, dict]:
    """
    Load ground truth data from CSV file.

    Args:
        gt_csv_path: Path to ground truth CSV file

    Returns:
        Dictionary mapping filename to GT data
    """
    gt_data = {}
    with open(gt_csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row.get("file_name", "")
            if file_name:
                gt_data[file_name] = {
                    "gt_contamination_area": row.get("gt_contamination_area", ""),
                    "gt_contamination_type": row.get("gt_contamination_type", ""),
                }
    return gt_data


def run_batch_inference(
    input_csv: Path,
    images_dir: Path,
    prompt_path: Path,
    output_csv: Path,
    api_url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    limit: Optional[int] = None,
) -> dict:
    """
    Run batch inference on images specified in CSV file.

    Args:
        input_csv: CSV file containing image filenames and ground truth data
        images_dir: Directory containing images
        prompt_path: Path to prompt file
        output_csv: Output CSV file path
        api_url: VLM API endpoint URL
        model: Model name
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        limit: Maximum number of images to process (default: all)

    Returns:
        Dictionary with summary statistics
    """
    # Load prompt
    prompt = load_prompt(prompt_path)

    # Initialize client
    client = VLMClient(api_url=api_url, model=model)

    # Load input CSV with ground truth data
    print(f"Loading input CSV from {input_csv}")
    image_data = []
    with open(input_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row.get("file_name", "")
            if file_name:
                image_data.append(
                    {
                        "file_name": file_name,
                        "gt_contamination_area": row.get("gt_contamination_area", ""),
                        "gt_contamination_type": row.get("gt_contamination_type", ""),
                    }
                )

    if not image_data:
        raise ValueError(f"No valid entries found in {input_csv}")

    print(f"Loaded {len(image_data)} entries from CSV")

    # Apply limit if specified
    if limit is not None and limit > 0:
        image_data = image_data[:limit]
        print(f"Processing first {len(image_data)} images (limit applied)")

    # Process images
    results = []
    for i, data in enumerate(image_data, 1):
        file_name = data["file_name"]
        image_path = images_dir / file_name

        print(f"[{i}/{len(image_data)}] Processing {file_name}...")

        # Check if image exists
        if not image_path.exists():
            print(f"  ✗ File not found: {image_path}")
            row = {
                "image_name": file_name,
                "gt_contamination_area": data["gt_contamination_area"],
                "gt_contamination_type": data["gt_contamination_type"],
                "model": model,
                "latency_seconds": "0.000",
                "success": False,
                "error": "file_not_found",
                "image_type": "",
            }
            # Add empty columns for all areas
            for area in ALL_AREAS:
                row[f"{area}_contamination_type"] = ""
                row[f"{area}_severity"] = ""
            results.append(row)
            continue

        inference_result = process_image(
            client=client,
            image_path=image_path,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        row = create_csv_row(image_path, inference_result, model)

        # Add ground truth columns from CSV
        row["gt_contamination_area"] = data["gt_contamination_area"]
        row["gt_contamination_type"] = data["gt_contamination_type"]

        results.append(row)

        if inference_result["success"]:
            print(f"  ✓ Success (latency: {inference_result['latency']:.3f}s)")
        else:
            print(f"  ✗ Failed: {inference_result.get('error', 'unknown')}")

    # Save results
    fieldnames = [
        "image_name",
        "gt_contamination_area",
        "gt_contamination_type",
        "model",
        "latency_seconds",
        "success",
        "error",
        "image_type",
    ]
    for area in ALL_AREAS:
        fieldnames.append(f"{area}_contamination_type")
        fieldnames.append(f"{area}_severity")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Calculate summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    avg_latency = sum(float(r["latency_seconds"]) for r in results) / len(results)

    return {
        "total": len(results),
        "successful": successful,
        "failed": failed,
        "avg_latency": avg_latency,
        "output_path": output_csv,
    }
