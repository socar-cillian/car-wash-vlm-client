"""Batch inference module for vehicle contamination classification."""

import csv
import json
import time
from pathlib import Path

from src.client import VLMClient


# Fixed vehicle areas
INTERIOR_AREAS = ["driver_seat", "passenger_seat", "cup_holder", "back_seat"]
EXTERIOR_AREAS = ["front", "passenger_side", "driver_side", "rear"]
ALL_AREAS = INTERIOR_AREAS + EXTERIOR_AREAS


def load_prompt(prompt_path: Path) -> str:
    """Load prompt from file."""
    with open(prompt_path, encoding="utf-8") as f:
        return f.read()


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
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON for {image_path.name}: {e}")
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
    areas_dict = {}
    for area_info in result.get("areas", []):
        area_name = area_info.get("area_name", "")
        areas_dict[area_name] = area_info

    # Add columns for each area
    for area in ALL_AREAS:
        if area in areas_dict:
            area_info = areas_dict[area]
            contamination_type = area_info.get("contamination_type", "")
            severity = area_info.get("severity", "")
        else:
            contamination_type = ""
            severity = ""

        row[f"{area}_contamination_type"] = contamination_type
        row[f"{area}_severity"] = severity

    return row


def run_batch_inference(
    images_dir: Path,
    prompt_path: Path,
    output_csv: Path,
    api_url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    limit: int | None = None,
) -> dict:
    """
    Run batch inference on all images in directory.

    Args:
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

    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
    image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions]

    if not image_files:
        raise ValueError(f"No images found in {images_dir}")

    # Apply limit if specified
    if limit is not None and limit > 0:
        image_files = image_files[:limit]
        print(f"Processing first {len(image_files)} images (limit applied)")

    # Process images
    results = []
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing {image_path.name}...")

        inference_result = process_image(
            client=client,
            image_path=image_path,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        row = create_csv_row(image_path, inference_result, model)
        results.append(row)

        if inference_result["success"]:
            print(f"  ✓ Success (latency: {inference_result['latency']:.3f}s)")
        else:
            print(f"  ✗ Failed: {inference_result.get('error', 'unknown')}")

    # Save results
    fieldnames = ["image_name", "model", "latency_seconds", "success", "error", "image_type"]
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
