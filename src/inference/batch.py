"""Batch inference module for vehicle contamination classification."""

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from src.api import VLMClient
from src.monitoring import get_langfuse_monitor


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
            image_name=image_path.name,
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


def create_csv_row(image_path: Path, inference_result: dict, model_name: str, prompt_version: str = "") -> dict:
    """
    Create a CSV row from inference result.

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
        row["classification"] = ""
        row["contamination_types"] = ""
        row["contamination_parts"] = ""
        row["raw_response"] = ""
        return row

    result = inference_result.get("result", {})

    # Add classification
    row["classification"] = result.get("classification", "")

    # Add contamination types (convert list to comma-separated string)
    contamination_types = result.get("contamination_types", [])
    if isinstance(contamination_types, list):
        row["contamination_types"] = ", ".join(contamination_types) if contamination_types else ""
    else:
        row["contamination_types"] = str(contamination_types)

    # Add contamination parts (convert list to comma-separated string)
    contamination_parts = result.get("contamination_parts", [])
    if isinstance(contamination_parts, list):
        row["contamination_parts"] = ", ".join(contamination_parts) if contamination_parts else ""
    else:
        row["contamination_parts"] = str(contamination_parts)

    # Add raw response for debugging (optional, can be removed if not needed)
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


def run_simple_batch_inference(
    images_dir: Path,
    prompt_path: Path,
    output_csv: Path,
    api_url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    limit: int | None = None,
    max_workers: int = 4,
    enable_langfuse: bool = True,
    metadata_csv: Path | None = None,
) -> dict:
    """
    Run batch inference on all images in a directory with parallel processing.

    Args:
        images_dir: Directory containing images
        prompt_path: Path to prompt file
        output_csv: Output CSV file path
        api_url: VLM API endpoint URL
        model: Model name
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        limit: Maximum number of images to process (default: all)
        max_workers: Number of parallel workers (default: 4)
        enable_langfuse: Whether to enable Langfuse monitoring (default: True)
        metadata_csv: Optional metadata CSV file to join with results

    Returns:
        Dictionary with summary statistics
    """
    # Initialize Langfuse monitor
    langfuse_monitor = get_langfuse_monitor(enabled=enable_langfuse)

    # Load prompt
    prompt = load_prompt(prompt_path)

    # Initialize client with langfuse monitoring
    client = VLMClient(api_url=api_url, model=model, langfuse_monitor=langfuse_monitor)

    # Extract prompt version from filename
    prompt_version = prompt_path.stem

    # Load metadata if provided
    metadata_dict = {}
    metadata_columns = []
    if metadata_csv is not None:
        print(f"Loading metadata from {metadata_csv}")
        with open(metadata_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            metadata_columns = reader.fieldnames or []
            for row in reader:
                # Try to get file_name from different possible columns
                file_name = row.get("file_name", "")
                if not file_name:
                    # If file_name is not present, try to extract from file_url
                    file_url = row.get("file_url", "")
                    if file_url:
                        # Extract filename from URL (e.g., "https://example.com/path/file.jpg" -> "file.jpg")
                        file_name = file_url.split("/")[-1]

                if file_name:
                    metadata_dict[file_name] = dict(row)
        print(f"Loaded metadata for {len(metadata_dict)} files")

    # Scan images directory for image files
    print(f"Scanning images directory: {images_dir}")
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    image_paths = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in image_extensions]

    if not image_paths:
        raise ValueError(f"No image files found in {images_dir}")

    print(f"Found {len(image_paths)} images")

    # Apply limit if specified
    if limit is not None and limit > 0:
        image_paths = image_paths[:limit]
        print(f"Processing first {len(image_paths)} images (limit applied)")

    print(f"Starting parallel inference with {max_workers} workers...")
    print()

    # Start timer for total processing time
    total_start_time = time.time()

    # Process images in parallel
    results = []
    futures_to_path = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        for image_path in image_paths:
            future = executor.submit(
                process_image,
                client,
                image_path,
                prompt,
                max_tokens,
                temperature,
            )
            futures_to_path[future] = image_path

        # Process completed tasks with progress bar
        with tqdm(total=len(futures_to_path), desc="Processing images", unit="img") as pbar:
            for future in as_completed(futures_to_path):
                image_path = futures_to_path[future]

                try:
                    inference_result = future.result()
                    row = create_csv_row(image_path, inference_result, model, prompt_version)

                    # Add metadata if available
                    if image_path.name in metadata_dict:
                        for col in metadata_columns:
                            if col not in row:
                                row[col] = metadata_dict[image_path.name].get(col, "")

                    results.append(row)

                    # Update progress bar description
                    if inference_result["success"]:
                        pbar.set_postfix_str(f"✓ {image_path.name} ({inference_result['latency']:.2f}s)")
                    else:
                        pbar.set_postfix_str(f"✗ {image_path.name}")

                except Exception as e:
                    # Handle unexpected errors
                    row = {
                        "image_name": image_path.name,
                        "model": model,
                        "prompt_version": prompt_version,
                        "latency_seconds": "0.000",
                        "success": False,
                        "error": str(e),
                        "classification": "",
                        "contamination_types": "",
                        "contamination_parts": "",
                        "raw_response": "",
                    }

                    # Add metadata if available
                    if image_path.name in metadata_dict:
                        for col in metadata_columns:
                            if col not in row:
                                row[col] = metadata_dict[image_path.name].get(col, "")

                    results.append(row)
                    pbar.set_postfix_str(f"✗ {image_path.name} (error)")

                pbar.update(1)

    # Save results
    # Start with metadata columns if available
    fieldnames = list(metadata_columns) if metadata_columns else []

    # Ensure image_name is in fieldnames
    if "image_name" not in fieldnames:
        fieldnames.insert(0, "image_name")

    # Add inference result columns
    inference_columns = [
        "model",
        "prompt_version",
        "latency_seconds",
        "success",
        "error",
        "classification",
        "contamination_types",
        "contamination_parts",
        "raw_response",
    ]
    for col in inference_columns:
        if col not in fieldnames:
            fieldnames.append(col)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Calculate total processing time
    total_elapsed_time = time.time() - total_start_time

    # Calculate summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    avg_latency = sum(float(r["latency_seconds"]) for r in results) / len(results)

    # Calculate per-image time (wall clock time divided by number of images)
    avg_time_per_image = total_elapsed_time / len(results) if results else 0
    speedup = avg_latency / avg_time_per_image if avg_time_per_image > 0 else 1

    # Log summary to Langfuse using trace
    if langfuse_monitor.enabled:
        trace = langfuse_monitor.start_trace_span(
            name="simple_batch_inference",
            input_data={
                "images_dir": str(images_dir),
                "model": model,
                "max_workers": max_workers,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            metadata={
                "total_images": len(results),
                "successful": successful,
                "failed": failed,
                "success_rate": successful / len(results) if results else 0,
            },
        )
        if trace:
            with trace as span:
                span.update(
                    output={
                        "total_images": len(results),
                        "successful": successful,
                        "failed": failed,
                        "avg_latency": avg_latency,
                        "total_time": total_elapsed_time,
                        "avg_time_per_image": avg_time_per_image,
                        "speedup": speedup,
                        "output_path": str(output_csv),
                    }
                )

        # Flush all events to Langfuse
        langfuse_monitor.flush()

    return {
        "total": len(results),
        "successful": successful,
        "failed": failed,
        "avg_latency": avg_latency,
        "total_time": total_elapsed_time,
        "avg_time_per_image": avg_time_per_image,
        "output_path": output_csv,
    }


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
    max_workers: int = 4,
    enable_langfuse: bool = True,
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
        enable_langfuse: Whether to enable Langfuse monitoring (default: True)

    Returns:
        Dictionary with summary statistics
    """
    # Initialize Langfuse monitor
    langfuse_monitor = get_langfuse_monitor(enabled=enable_langfuse)

    # Load prompt
    prompt = load_prompt(prompt_path)

    # Initialize client with langfuse monitoring
    client = VLMClient(api_url=api_url, model=model, langfuse_monitor=langfuse_monitor)

    # Extract prompt version from filename
    prompt_version = prompt_path.stem

    # Load input CSV with all columns
    print(f"Loading input CSV from {input_csv}")
    image_data = []
    input_csv_columns = []
    with open(input_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        input_csv_columns = reader.fieldnames or []
        for row in reader:
            file_name = row.get("file_name", "")
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

    print(f"Starting parallel inference with {max_workers} workers...")
    print()

    # Start timer for total processing time
    total_start_time = time.time()

    # Process images in parallel
    results = []
    futures_to_data = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        for data in image_data:
            file_name = data["file_name"]
            image_path = images_dir / file_name

            # Check if image exists before submitting
            if not image_path.exists():
                row = {
                    **data,  # Include all original CSV columns
                    "model": model,
                    "prompt_version": prompt_version,
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

            # Submit inference task
            future = executor.submit(
                process_image,
                client,
                image_path,
                prompt,
                max_tokens,
                temperature,
            )
            futures_to_data[future] = (data, image_path)

        # Process completed tasks with progress bar
        with tqdm(total=len(futures_to_data), desc="Processing images", unit="img") as pbar:
            for future in as_completed(futures_to_data):
                data, image_path = futures_to_data[future]

                try:
                    inference_result = future.result()
                    row = create_csv_row_legacy(image_path, inference_result, model, prompt_version)

                    # Add all original CSV columns
                    for col in input_csv_columns:
                        if col not in row:
                            row[col] = data.get(col, "")

                    results.append(row)

                    # Update progress bar description
                    if inference_result["success"]:
                        pbar.set_postfix_str(f"✓ {image_path.name} ({inference_result['latency']:.2f}s)")
                    else:
                        pbar.set_postfix_str(f"✗ {image_path.name}")

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
                    }
                    for area in ALL_AREAS:
                        row[f"{area}_contamination_type"] = ""
                        row[f"{area}_severity"] = ""
                    results.append(row)
                    pbar.set_postfix_str(f"✗ {image_path.name} (error)")

                pbar.update(1)

    # Save results - combine original CSV columns with inference results
    # Start with original CSV columns
    fieldnames = list(input_csv_columns)

    # Add inference result columns
    inference_columns = [
        "model",
        "prompt_version",
        "latency_seconds",
        "success",
        "error",
        "image_type",
    ]
    for col in inference_columns:
        if col not in fieldnames:
            fieldnames.append(col)

    # Add area-specific columns
    for area in ALL_AREAS:
        fieldnames.append(f"{area}_contamination_type")
        fieldnames.append(f"{area}_severity")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Calculate total processing time
    total_elapsed_time = time.time() - total_start_time

    # Calculate summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    avg_latency = sum(float(r["latency_seconds"]) for r in results) / len(results)

    # Calculate per-image time (wall clock time divided by number of images)
    avg_time_per_image = total_elapsed_time / len(results) if results else 0
    speedup = avg_latency / avg_time_per_image if avg_time_per_image > 0 else 1

    # Log summary to Langfuse using trace
    if langfuse_monitor.enabled:
        trace = langfuse_monitor.start_trace_span(
            name="batch_inference",
            input_data={
                "input_csv": str(input_csv),
                "images_dir": str(images_dir),
                "model": model,
                "max_workers": max_workers,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            metadata={
                "total_images": len(results),
                "successful": successful,
                "failed": failed,
                "success_rate": successful / len(results) if results else 0,
            },
        )
        if trace:
            with trace as span:
                span.update(
                    output={
                        "total_images": len(results),
                        "successful": successful,
                        "failed": failed,
                        "avg_latency": avg_latency,
                        "total_time": total_elapsed_time,
                        "avg_time_per_image": avg_time_per_image,
                        "speedup": speedup,
                        "output_path": str(output_csv),
                    }
                )

        # Flush all events to Langfuse
        langfuse_monitor.flush()

    return {
        "total": len(results),
        "successful": successful,
        "failed": failed,
        "avg_latency": avg_latency,
        "total_time": total_elapsed_time,
        "avg_time_per_image": avg_time_per_image,
        "output_path": output_csv,
    }
