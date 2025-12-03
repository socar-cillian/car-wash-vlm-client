"""Batch inference module for vehicle contamination classification."""

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

from src.api import VLMClient


# Fixed vehicle areas (Korean)
INTERIOR_AREAS = ["운전석", "조수석", "컵홀더", "뒷좌석"]
EXTERIOR_AREAS = ["전면", "조수석 방향", "운전석 방향", "후면"]
ALL_AREAS = INTERIOR_AREAS + EXTERIOR_AREAS

# Valid severity levels
SEVERITY_LEVELS = ["Level 0", "Level 1", "Level 2", "Level 3", "Level 4"]


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
    image_path: Path | str,
    prompt: str,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    image_name: str | None = None,
) -> dict:
    """
    Process a single image with VLM inference.

    Args:
        client: VLM client instance
        image_path: Path to image file or URL string
        prompt: Prompt text
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        image_name: Optional image name for logging (used when image_path is a URL)

    Returns:
        Dictionary containing inference results and metadata
    """
    start_time = time.time()

    try:
        # Determine image name for logging
        if image_name is None:
            if isinstance(image_path, Path):
                image_name = image_path.name
            else:
                # Extract from URL
                parsed = urlparse(str(image_path))
                image_name = Path(parsed.path).name

        response = client.query(
            image_input=str(image_path),
            prompt_input=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            image_name=image_name,
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
                # result = _normalize_inference_result(result, image_name)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON for {image_name}: {e}")
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


def create_csv_row(image_path: Path, inference_result: dict, model_name: str, prompt_version: str = "") -> dict:
    """
    Create a CSV row from inference result.
    Supports new v4.1 format with area_name, sub_area, and contaminations array.

    Args:
        image_path: Path to image file
        inference_result: Inference result dictionary
        model_name: Name of the model used
        prompt_version: Name of the prompt file/version used

    Returns:
        Dictionary representing a CSV row
    """
    row = {
        "image_name": image_path.name,
        "model": model_name,
        "prompt_version": prompt_version,
        "latency_seconds": f"{inference_result['latency']:.3f}",
        "success": inference_result["success"],
    }

    if not inference_result["success"]:
        row["error"] = inference_result.get("error", "unknown_error")
        row["image_type"] = ""
        row["area_name"] = ""
        row["sub_area"] = ""
        row["contamination_type"] = ""
        row["severity"] = ""
        row["is_in_guideline"] = ""
        row["max_severity"] = ""
        row["raw_response"] = ""
        return row

    result = inference_result.get("result", {})

    # Extract image_type
    row["image_type"] = result.get("image_type", "")

    # Extract areas array (new format)
    areas = result.get("areas", [])

    # Flatten areas into rows - collect all contaminations
    all_contaminations = []
    max_severity_level = -1
    max_severity_str = "Level 0"

    for area in areas:
        area_name = area.get("area_name", "")
        sub_area = area.get("sub_area", "")

        # New format: contaminations is an array
        contaminations = area.get("contaminations", [])

        for contamination in contaminations:
            cont_type = contamination.get("contamination_type", "")
            severity = contamination.get("severity", "Level 0")
            is_in_guideline = contamination.get("is_in_guideline", True)

            all_contaminations.append(
                {
                    "area_name": area_name,
                    "sub_area": sub_area,
                    "contamination_type": cont_type,
                    "severity": severity,
                    "is_in_guideline": is_in_guideline,
                }
            )

            # Track max severity (Level 0 ~ Level 4)
            try:
                level_num = int(severity.replace("Level ", ""))
                if level_num > max_severity_level:
                    max_severity_level = level_num
                    max_severity_str = severity
            except (ValueError, AttributeError):
                pass

    # If no contaminations found, use empty values
    if not all_contaminations:
        row["area_name"] = ""
        row["sub_area"] = ""
        row["contamination_type"] = ""
        row["severity"] = ""
        row["is_in_guideline"] = ""
        row["max_severity"] = "Level 0"
        row["raw_response"] = str(result)
        return row

    # Aggregate all contaminations into comma-separated strings
    area_names = sorted({c["area_name"] for c in all_contaminations if c["area_name"]})
    sub_areas = sorted({c["sub_area"] for c in all_contaminations if c["sub_area"]})
    cont_types = sorted({c["contamination_type"] for c in all_contaminations if c["contamination_type"]})
    severities = sorted({c["severity"] for c in all_contaminations if c["severity"]})

    # Check if any contamination is not in guideline
    has_non_guideline = any(not c["is_in_guideline"] for c in all_contaminations)

    row["area_name"] = ", ".join(area_names)
    row["sub_area"] = ", ".join(sub_areas)
    row["contamination_type"] = ", ".join(cont_types)
    row["severity"] = ", ".join(severities)
    row["is_in_guideline"] = "N" if has_non_guideline else "Y"
    row["max_severity"] = max_severity_str
    row["raw_response"] = str(result)

    return row


def create_csv_row_legacy(image_path: Path, inference_result: dict, model_name: str, prompt_version: str = "") -> dict:
    """
    Create a CSV row from inference result (legacy format for area-based inference).

    Args:
        image_path: Path to image file
        inference_result: Inference result dictionary
        model_name: Name of the model used
        prompt_version: Name of the prompt file/version used

    Returns:
        Dictionary representing a CSV row
    """
    row = {
        "image_name": image_path.name,
        "model": model_name,
        "prompt_version": prompt_version,
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
    limit: int | None = None,
    max_workers: int = 16,
) -> dict:
    """
    Run batch inference on images specified in CSV file with parallel processing.

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
        max_workers: Number of parallel workers (default: 4)

    Returns:
        Dictionary with summary statistics
    """
    # Load prompt
    prompt = load_prompt(prompt_path)

    # Initialize client
    client = VLMClient(api_url=api_url, model=model)

    # Extract prompt version from filename
    prompt_version = prompt_path.stem

    # Load input CSV with all columns
    print(f"Loading input CSV from {input_csv}")
    image_data = []
    input_csv_columns = []
    with open(input_csv, encoding="utf-8-sig") as f:  # utf-8-sig handles BOM
        reader = csv.DictReader(f)
        input_csv_columns = reader.fieldnames or []
        for row in reader:
            # Get file_name - try multiple column names: file_name, image_name
            file_name = row.get("file_name") or row.get("image_name", "")

            if not file_name:
                # Try filename column (may contain path like /000565/56542/202511/uuid.jpg)
                filename_val = row.get("filename", "")
                if filename_val:
                    file_name = Path(filename_val).name

            if not file_name:
                # Try file_url first, then image_url
                url = row.get("file_url") or row.get("image_url")
                if url:
                    parsed = urlparse(url)
                    file_name = Path(parsed.path).name

            if file_name:
                row["file_name"] = file_name

            if file_name:
                # Store all columns from the original CSV
                image_data.append(dict(row))

    if not image_data:
        raise ValueError(f"No valid entries found in {input_csv}")

    print(f"Loaded {len(image_data)} entries from CSV")

    # Apply limit if specified
    if limit is not None and limit > 0:
        image_data = image_data[:limit]
        print(f"Processing first {len(image_data)} images (limit applied)")

    # Check local image availability
    local_count = 0
    missing_count = 0
    for data in image_data:
        file_name = data["file_name"]
        image_path = images_dir / file_name
        if image_path.exists():
            local_count += 1
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"Image sources: {local_count} local files found, {missing_count} missing")
    else:
        print(f"Image sources: {local_count} local files")

    print(f"Starting parallel inference with {max_workers} workers...")
    print()

    # Start timer for total processing time
    total_start_time = time.time()

    # Prepare fieldnames for CSV output
    # Start with original CSV columns
    fieldnames = list(input_csv_columns)

    # Add inference result columns (v4.1 format)
    inference_columns = [
        "image_name",
        "label",  # Include label column for evaluation
        "model",
        "prompt_version",
        "latency_seconds",
        "success",
        "error",
        "image_type",
        "area_name",
        "sub_area",
        "contamination_type",
        "severity",
        "max_severity",
        "is_in_guideline",
        "raw_response",
    ]
    for col in inference_columns:
        if col not in fieldnames:
            fieldnames.append(col)

    # Create output directory and open CSV file for streaming write
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Counters for summary (instead of storing all results in memory)
    successful_count = 0
    failed_count = 0
    total_latency = 0.0

    # Process images in parallel with streaming CSV write
    futures_to_data = {}

    with open(output_csv, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Use a single Progress context for both submitting and processing
        with (
            Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                TextColumn("• {task.fields[status]}"),
                refresh_per_second=30,  # Faster refresh for large datasets
            ) as progress,
            ThreadPoolExecutor(max_workers=max_workers) as executor,
        ):
            # Submit all tasks with progress
            submit_task = progress.add_task("Submitting tasks", total=len(image_data), status="starting...")
            progress.refresh()  # Force immediate render

            for data in image_data:
                file_name = data["file_name"]

                # Only use local files
                image_path = images_dir / file_name
                if not image_path.exists():
                    # Local file not found - write immediately and skip
                    row = {
                        **data,  # Include all original CSV columns
                        "model": model,
                        "prompt_version": prompt_version,
                        "latency_seconds": "0.000",
                        "success": False,
                        "error": "file_not_found",
                        "image_type": "",
                        "area_name": "",
                        "sub_area": "",
                        "contamination_type": "",
                        "severity": "",
                        "max_severity": "",
                        "is_in_guideline": "",
                        "raw_response": "",
                    }
                    writer.writerow(row)
                    failed_count += 1
                    progress.update(submit_task, advance=1, status=f"skip: {file_name}")
                    continue

                # Use local file
                image_source = str(image_path)
                image_display_path = image_path

                # Submit inference task
                future = executor.submit(
                    process_image,
                    client,
                    image_source,  # Can be URL string or local path string
                    prompt,
                    max_tokens,
                    temperature,
                    file_name,  # Pass filename for logging
                )
                futures_to_data[future] = (data, image_display_path)
                progress.update(submit_task, advance=1, status=f"{file_name}")

            # Mark submit task as complete
            progress.update(submit_task, visible=False)

            # Process completed tasks with progress bar - write immediately to CSV
            process_task = progress.add_task("Processing images", total=len(futures_to_data), status="")

            for future in as_completed(futures_to_data):
                data, image_display_path = futures_to_data[future]

                try:
                    inference_result = future.result()
                    row = create_csv_row(image_display_path, inference_result, model, prompt_version)

                    # Add all original CSV columns
                    for col in input_csv_columns:
                        if col not in row:
                            row[col] = data.get(col, "")

                    # Write immediately to CSV (streaming write)
                    writer.writerow(row)

                    # Update counters
                    latency = float(row["latency_seconds"])
                    total_latency += latency
                    if inference_result["success"]:
                        successful_count += 1
                        progress.update(
                            process_task,
                            advance=1,
                            status=f"✓ {image_display_path.name} ({latency:.2f}s)",
                        )
                    else:
                        failed_count += 1
                        progress.update(process_task, advance=1, status=f"✗ {image_display_path.name}")

                except Exception as e:
                    # Handle unexpected errors
                    row = {
                        **data,  # Include all original CSV columns
                        "model": model,
                        "prompt_version": prompt_version,
                        "latency_seconds": "0.000",
                        "success": False,
                        "error": str(e),
                        "image_type": "",
                        "area_name": "",
                        "sub_area": "",
                        "contamination_type": "",
                        "severity": "",
                        "max_severity": "",
                        "is_in_guideline": "",
                        "raw_response": "",
                    }
                    # Write immediately to CSV (streaming write)
                    writer.writerow(row)
                    failed_count += 1
                    progress.update(process_task, advance=1, status=f"✗ {image_display_path.name} (error)")

                # Flush periodically to ensure data is written to disk
                if (successful_count + failed_count) % 100 == 0:
                    csv_file.flush()

    # Calculate total processing time
    total_elapsed_time = time.time() - total_start_time

    # Calculate summary using counters (no need to store all results in memory)
    total_count = successful_count + failed_count
    avg_latency = total_latency / total_count if total_count > 0 else 0

    # Calculate per-image time (wall clock time divided by number of images)
    avg_time_per_image = total_elapsed_time / total_count if total_count > 0 else 0

    return {
        "total": total_count,
        "successful": successful_count,
        "failed": failed_count,
        "avg_latency": avg_latency,
        "total_time": total_elapsed_time,
        "avg_time_per_image": avg_time_per_image,
        "output_path": output_csv,
    }
