#!/usr/bin/env python3
"""
Merge multiple CSV files and download images from URLs.

This script:
1. Merges all CSV files from a specified directory
2. Renames columns according to the mapping
3. Downloads images from image_url column
4. Saves the merged CSV with file_name column added
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests


def merge_csv_files(csv_dir: Path) -> list[dict]:
    """
    Merge all CSV files in the specified directory.

    Args:
        csv_dir: Directory containing CSV files

    Returns:
        List of dictionaries containing all rows from all CSV files
    """
    all_rows = []
    csv_files = list(csv_dir.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {csv_dir}")

    print(f"Found {len(csv_files)} CSV file(s) to merge")

    for csv_file in csv_files:
        print(f"Reading {csv_file.name}...")
        with open(csv_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            all_rows.extend(rows)
            print(f"  Added {len(rows)} rows")

    print(f"Total rows merged: {len(all_rows)}")
    return all_rows


def rename_columns(rows: list[dict]) -> list[dict]:
    """
    Rename columns according to the mapping and add gt_contamination_area.

    Mapping:
        lesion_id -> id
        car_id -> car_id (unchanged)
        car_num -> car_num (unchanged)
        lesion_type -> gt_contamination_type
        file_url -> file_url (unchanged)
        확인용 -> image_url

    Additionally adds:
        gt_contamination_area -> "exterior" if lesion_type contains "외부", else "interior"

    Args:
        rows: List of dictionaries with original column names

    Returns:
        List of dictionaries with renamed columns
    """
    column_mapping = {
        "lesion_id": "id",
        "car_id": "car_id",
        "car_num": "car_num",
        "lesion_type": "gt_contamination_type",
        "file_url": "file_url",
        "확인용": "image_url",
    }

    renamed_rows = []
    for row in rows:
        new_row = {}
        for old_col, new_col in column_mapping.items():
            new_row[new_col] = row.get(old_col, "")

        # Add gt_contamination_area based on lesion_type
        lesion_type = row.get("lesion_type", "")
        new_row["gt_contamination_area"] = "exterior" if "외부" in lesion_type else "interior"

        renamed_rows.append(new_row)

    return renamed_rows


def download_image(url: str, output_dir: Path) -> str | None:
    """
    Download image from URL and save to output directory.

    Args:
        url: Image URL to download
        output_dir: Directory to save the image

    Returns:
        Filename of the downloaded image, or None if download failed
    """
    if not url or url.strip() == "":
        return None

    try:
        # Parse URL to get filename
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)

        # If no filename in URL, generate one based on URL hash
        if not filename or "." not in filename:
            filename = f"{abs(hash(url))}.jpg"

        output_path = output_dir / filename

        # Skip if already exists
        if output_path.exists():
            print(f"  Skipping {filename} (already exists)")
            return filename

        # Download image
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Save to file
        with open(output_path, "wb") as f:
            f.write(response.content)

        print(f"  Downloaded {filename}")
        return filename

    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        return None


def download_all_images(rows: list[dict], output_dir: Path) -> list[dict]:
    """
    Download all images from image_url column and add file_name column.

    Args:
        rows: List of dictionaries with image_url column
        output_dir: Directory to save images

    Returns:
        List of dictionaries with file_name column added
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading images to {output_dir}...")

    updated_rows = []
    for i, row in enumerate(rows, 1):
        image_url = row.get("image_url", "")
        print(f"[{i}/{len(rows)}] Processing {image_url[:60]}...")

        filename = download_image(image_url, output_dir)
        row["file_name"] = filename if filename else ""

        updated_rows.append(row)

    return updated_rows


def save_merged_csv(rows: list[dict], output_path: Path):
    """
    Save merged data to CSV file.

    Args:
        rows: List of dictionaries to save
        output_path: Path to output CSV file
    """
    if not rows:
        raise ValueError("No rows to save")

    # Define column order
    fieldnames = [
        "id",
        "car_id",
        "car_num",
        "gt_contamination_area",
        "gt_contamination_type",
        "file_url",
        "image_url",
        "file_name",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nMerged CSV saved to {output_path}")
    print(f"Total rows: {len(rows)}")


def main():
    """Main function to merge CSV files and download images."""
    # Get the project root directory (parent of utils/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Merge CSV files and download images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python utils/merge_csv_and_download_images.py \\
    --csv-dir images/sample_images/csv \\
    --output merged_data.csv \\
    --image-dir images/sample_images/images
        """,
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=project_root / "images/sample_images/csv",
        help="Directory containing CSV files to merge (default: images/sample_images/csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "images/sample_images/csv/merged_data.csv",
        help="Output path for merged CSV file (default: images/sample_images/csv/merged_data.csv)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=project_root / "images/sample_images/images",
        help="Directory to save downloaded images (default: images/sample_images/images)",
    )

    args = parser.parse_args()

    try:
        # Step 1: Merge CSV files
        print("=" * 60)
        print("Step 1: Merging CSV files")
        print("=" * 60)
        rows = merge_csv_files(args.csv_dir)

        # Step 2: Rename columns
        print("\n" + "=" * 60)
        print("Step 2: Renaming columns")
        print("=" * 60)
        rows = rename_columns(rows)
        print("Columns renamed successfully")

        # Step 3: Download images
        print("\n" + "=" * 60)
        print("Step 3: Downloading images")
        print("=" * 60)
        rows = download_all_images(rows, args.image_dir)

        # Step 4: Save merged CSV
        print("\n" + "=" * 60)
        print("Step 4: Saving merged CSV")
        print("=" * 60)
        save_merged_csv(rows, args.output)

        print("\n" + "=" * 60)
        print("✓ All steps completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
