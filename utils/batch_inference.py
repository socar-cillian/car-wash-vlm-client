#!/usr/bin/env python3
"""
Batch inference script for vehicle contamination classification.

This script:
1. Loads images from a directory
2. Runs inference using VLM API with a specified prompt
3. Saves results to CSV with detailed metrics
"""

import argparse
import csv
import json
import sys
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


def batch_inference(
    images_dir: Path,
    prompt_path: Path,
    output_csv: Path,
    api_url: str,
    model: str,
    max_tokens: int,
    temperature: float,
):
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
    """
    # Load prompt
    print("=" * 60)
    print("Step 1: Loading prompt")
    print("=" * 60)
    prompt = load_prompt(prompt_path)
    print(f"Loaded prompt from: {prompt_path}")
    print(f"Prompt length: {len(prompt)} characters")

    # Initialize client
    print("\n" + "=" * 60)
    print("Step 2: Initializing VLM client")
    print("=" * 60)
    client = VLMClient(api_url=api_url, model=model)
    print(f"API URL: {api_url}")
    print(f"Model: {model}")

    # Get all image files
    print("\n" + "=" * 60)
    print("Step 3: Collecting images")
    print("=" * 60)
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
    image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions]
    print(f"Found {len(image_files)} images in {images_dir}")

    if not image_files:
        print("No images found. Exiting.")
        return

    # Process images
    print("\n" + "=" * 60)
    print("Step 4: Running inference")
    print("=" * 60)

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
    print("\n" + "=" * 60)
    print("Step 5: Saving results")
    print("=" * 60)

    # Define column order
    fieldnames = ["image_name", "model", "latency_seconds", "success", "error", "image_type"]
    for area in ALL_AREAS:
        fieldnames.append(f"{area}_contamination_type")
        fieldnames.append(f"{area}_severity")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to: {output_csv}")
    print(f"Total images processed: {len(results)}")

    # Print summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    avg_latency = sum(float(r["latency_seconds"]) for r in results) / len(results)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Average latency: {avg_latency:.3f}s")


def main():
    """Main function."""
    # Get the project root directory
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Batch inference for vehicle contamination classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python utils/batch_inference.py \\
    --images images/sample_images/images \\
    --prompt prompts/car_contamination_classification_prompt_v2.txt \\
    --output results/inference_results.csv \\
    --model qwen3-vl-4b-instruct
        """,
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing images to process",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        required=True,
        help="Path to prompt file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://vllm.mlops.socarcorp.co.kr/v1/chat/completions",
        help="VLM API endpoint URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-vl-4b-instruct",
        help="Model name to use",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )

    args = parser.parse_args()

    # Make paths absolute if relative
    images_dir = args.images if args.images.is_absolute() else project_root / args.images
    prompt_path = args.prompt if args.prompt.is_absolute() else project_root / args.prompt
    output_csv = args.output if args.output.is_absolute() else project_root / args.output

    # Validate paths
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}", file=sys.stderr)
        sys.exit(1)

    if not prompt_path.exists():
        print(f"Error: Prompt file not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)

    try:
        batch_inference(
            images_dir=images_dir,
            prompt_path=prompt_path,
            output_csv=output_csv,
            api_url=args.api_url,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        print("\n" + "=" * 60)
        print("✓ All steps completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
