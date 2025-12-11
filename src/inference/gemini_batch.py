"""Gemini Batch API inference module for vehicle contamination classification.

This module uses Google's Gemini Batch API to process images at 50% of standard API costs.
Reference: https://ai.google.dev/gemini-api/docs/batch-api

Supported modes:
- Vertex AI: Requires GCS bucket for input/output (gcs_uri)
- Gemini Developer API: Supports inline requests (no GCS needed)
"""

import base64
import csv
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse


if TYPE_CHECKING:
    from google.cloud import storage

from dotenv import load_dotenv
from google import genai
from google.genai import types
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm
from rich.table import Table

from src.inference.pricing import (
    AVERAGE_OUTPUT_TOKENS,
    TOKENS_PER_IMAGE_ESTIMATE,
    CostInfo,
)
from src.inference.pricing import (
    estimate_image_tokens as pricing_estimate_image_tokens,
)


# Load environment variables
load_dotenv()

console = Console()


# =============================================================================
# GCS UTILITIES (for Vertex AI)
# =============================================================================


def _resolve_credentials_path(creds_path: str) -> Path | None:
    """
    Resolve credentials path, handling relative paths relative to project root.

    Args:
        creds_path: Path from environment variable (can be relative or absolute)

    Returns:
        Resolved Path if file exists, None otherwise
    """
    path = Path(creds_path)

    # If absolute path and exists, return it
    if path.is_absolute() and path.exists():
        return path

    # If relative path, try to resolve relative to project root
    # Find project root by looking for common markers (pyproject.toml, .env, etc.)
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".env").exists():
            project_root = parent
            resolved = (project_root / creds_path).resolve()
            if resolved.exists():
                return resolved
            break

    # Try relative to current working directory as fallback
    resolved = path.resolve()
    if resolved.exists():
        return resolved

    return None


def _get_storage_client() -> "storage.Client":
    """
    Get a GCS storage client with credentials from environment.

    Uses GOOGLE_APPLICATION_CREDENTIALS environment variable.
    Handles relative paths relative to project root.
    """
    from google.cloud import storage
    from google.oauth2 import service_account

    creds_path_str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not creds_path_str:
        # Fallback to Application Default Credentials
        return storage.Client()

    # Resolve path (handles relative paths)
    creds_path = _resolve_credentials_path(creds_path_str)

    if creds_path and creds_path.exists():
        console.print(f"[dim]Using service account: {creds_path}[/dim]")
        credentials = service_account.Credentials.from_service_account_file(str(creds_path))
        return storage.Client(credentials=credentials, project=credentials.project_id)
    else:
        # Fallback to Application Default Credentials
        console.print(f"[yellow]Warning: Service account file not found: {creds_path_str}[/yellow]")
        console.print(
            f"[yellow]  Resolved path: {creds_path}[/yellow]"
            if creds_path
            else "[yellow]  Could not resolve path[/yellow]"
        )
        console.print("[yellow]Falling back to Application Default Credentials[/yellow]")
        return storage.Client()


def upload_to_gcs(local_path: Path, bucket_name: str, blob_name: str) -> str:
    """
    Upload a local file to Google Cloud Storage.

    Args:
        local_path: Path to local file
        bucket_name: GCS bucket name
        blob_name: Destination blob name (path within bucket)

    Returns:
        GCS URI (gs://bucket/blob_name)
    """
    client = _get_storage_client()

    # Check if bucket exists
    try:
        bucket = client.bucket(bucket_name)
        # Try to get bucket metadata to verify it exists and we have access
        bucket.reload()
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "Not Found" in error_msg:
            # Get credentials path for error message
            creds_path_str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "not set")
            raise ValueError(
                f"Bucket '{bucket_name}' not found or you don't have access. "
                f"Please verify:\n"
                f"  1. Bucket name is correct: {bucket_name}\n"
                f"  2. Service account has 'Storage Object Admin' or 'Storage Admin' role\n"
                f"  3. GOOGLE_APPLICATION_CREDENTIALS is set correctly: {creds_path_str}\n"
                f"  4. Service account has access to the bucket"
            ) from e
        raise

    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))

    return f"gs://{bucket_name}/{blob_name}"


def download_from_gcs(gcs_uri: str, local_path: Path) -> None:
    """
    Download a file from Google Cloud Storage.

    Args:
        gcs_uri: GCS URI (gs://bucket/blob_name)
        local_path: Destination local path
    """
    # Parse gs:// URI
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")

    parts = gcs_uri[5:].split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else ""

    client = _get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(str(local_path))


def delete_gcs_blob(gcs_uri: str) -> None:
    """
    Delete a blob from Google Cloud Storage.

    Args:
        gcs_uri: GCS URI (gs://bucket/blob_name)
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")

    parts = gcs_uri[5:].split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else ""

    client = _get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete()


def list_gcs_images(gcs_path: str) -> list[str]:
    """
    List image files in a GCS path.

    Args:
        gcs_path: GCS path (gs://bucket/path or bucket/path)

    Returns:
        List of GCS URIs for image files
    """
    # Normalize path
    if gcs_path.startswith("gs://"):
        gcs_path = gcs_path[5:]

    parts = gcs_path.split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    # Ensure prefix ends with /
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    client = _get_storage_client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
    image_uris = []

    for blob in blobs:
        ext = Path(blob.name).suffix.lower()
        if ext in image_extensions:
            image_uris.append(f"gs://{bucket_name}/{blob.name}")

    return image_uris


def parse_gcs_path(path: str) -> tuple[str, str]:
    """
    Parse a GCS path into bucket and prefix.

    Args:
        path: GCS path (gs://bucket/path or bucket/path)

    Returns:
        Tuple of (bucket_name, prefix)
    """
    if path.startswith("gs://"):
        path = path[5:]

    parts = path.split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    return bucket_name, prefix


def is_gcs_path(path: str) -> bool:
    """Check if a path is a GCS path."""
    return path.startswith("gs://") or "/" in path and not path.startswith("/")


def get_mime_type_from_extension(file_name: str) -> str:
    """Get MIME type from file extension."""
    suffix = Path(file_name).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mime_types.get(suffix, "image/jpeg")


def get_genai_client() -> tuple[genai.Client, bool]:
    """
    Initialize and return a Google GenAI client.

    Returns:
        Tuple of (client, use_vertex_ai flag)
    """
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "FALSE").upper() == "TRUE"

    if use_vertex:
        # Vertex AI mode - uses Application Default Credentials or service account
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        client = genai.Client(vertexai=True, project=project, location=location)
    else:
        # Gemini Developer API mode
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        client = genai.Client(api_key=api_key)

    return client, use_vertex


def estimate_image_tokens(image_path: Path, model: str = "") -> int:
    """
    Estimate tokens for an image based on model and resolution.

    This is a wrapper around pricing.estimate_image_tokens that accepts a Path
    for backwards compatibility with the existing codebase.

    See src/inference/pricing.py for detailed documentation on image tokenization.
    """
    return pricing_estimate_image_tokens(model, resolution="medium")


def count_tokens_for_content(
    client: genai.Client,
    model: str,
    prompt: str,
    image_path: Path | None = None,
) -> int:
    """
    Count tokens for content using the Gemini API.

    Args:
        client: GenAI client
        model: Model name
        prompt: Text prompt
        image_path: Optional path to image file

    Returns:
        Token count from API
    """
    try:
        contents: list[Any] = []

        if image_path and image_path.exists():
            # Build content with image
            image_data, mime_type = encode_image_to_base64(image_path)
            contents = [
                types.Content(
                    parts=[
                        types.Part.from_bytes(data=base64.b64decode(image_data), mime_type=mime_type),
                        types.Part.from_text(text=prompt),
                    ]
                )
            ]
        else:
            # Text only
            contents = [prompt]

        response = client.models.count_tokens(
            model=model,
            contents=contents,
        )

        return int(response.total_tokens) if hasattr(response, "total_tokens") and response.total_tokens else 0

    except Exception as e:
        console.print(f"[yellow]Warning: Could not count tokens via API: {e}[/yellow]")
        # Fallback to estimation
        text_tokens = len(prompt) // 4
        image_tokens = estimate_image_tokens(image_path, model) if image_path else 0
        return text_tokens + image_tokens


def count_tokens_for_batch(
    client: genai.Client,
    model: str,
    prompt: str,
    image_paths: list[Path],
    sample_size: int = 5,
) -> dict[str, Any]:
    """
    Count tokens for a batch of images using sampling.

    For efficiency, samples a few images and extrapolates to full batch.

    Args:
        client: GenAI client
        model: Model name
        prompt: Text prompt (same for all images)
        image_paths: List of image paths
        sample_size: Number of images to sample for accurate estimation

    Returns:
        Dictionary with token statistics
    """
    if not image_paths:
        return {
            "total_input_tokens": 0,
            "avg_tokens_per_image": 0,
            "prompt_tokens": len(prompt) // 4,
            "sampled": False,
        }

    # Get prompt-only token count
    prompt_only_tokens = 0
    try:
        response = client.models.count_tokens(model=model, contents=prompt)
        if hasattr(response, "total_tokens") and response.total_tokens:
            prompt_only_tokens = int(response.total_tokens)
        else:
            prompt_only_tokens = len(prompt) // 4
    except Exception:
        prompt_only_tokens = len(prompt) // 4

    # Sample images for accurate token count
    sample_paths = image_paths[: min(sample_size, len(image_paths))]
    sampled_image_tokens = []

    console.print(f"[cyan]Counting tokens for {len(sample_paths)} sample images...[/cyan]")

    for img_path in sample_paths:
        if img_path.exists():
            try:
                total_tokens = count_tokens_for_content(client, model, prompt, img_path)
                # Image tokens = total - prompt tokens
                image_only_tokens = max(0, total_tokens - prompt_only_tokens)
                sampled_image_tokens.append(image_only_tokens)
            except Exception:
                # Use estimation
                sampled_image_tokens.append(estimate_image_tokens(img_path, model))

    # Calculate average
    if sampled_image_tokens:
        avg_image_tokens = sum(sampled_image_tokens) / len(sampled_image_tokens)
    else:
        avg_image_tokens = TOKENS_PER_IMAGE_ESTIMATE

    # Extrapolate to full batch
    total_input_tokens = int((prompt_only_tokens + avg_image_tokens) * len(image_paths))
    tokens_per_request = int(prompt_only_tokens + avg_image_tokens)

    return {
        "total_input_tokens": total_input_tokens,
        "tokens_per_request": tokens_per_request,
        "prompt_tokens": prompt_only_tokens,
        "avg_image_tokens": int(avg_image_tokens),
        "sampled_count": len(sampled_image_tokens),
        "total_images": len(image_paths),
        "sampled": len(image_paths) > sample_size,
    }


# Note: Pricing tables, constants, and CostInfo class have been moved to
# src/inference/pricing.py for better organization and documentation.
# See that file for detailed pricing information and calculation logic.


@dataclass
class BatchTimingInfo:
    """Timing information for batch job."""

    job_submitted_at: datetime | None = None  # When we submitted the job
    job_started_at: datetime | None = None  # When batch actually started processing
    job_completed_at: datetime | None = None  # When batch finished

    def to_dict(self) -> dict[str, str]:
        """Convert timing info to dictionary with ISO format strings."""
        return {
            "job_submitted_at": (self.job_submitted_at.isoformat() if self.job_submitted_at else ""),
            "job_started_at": (self.job_started_at.isoformat() if self.job_started_at else ""),
            "job_completed_at": (self.job_completed_at.isoformat() if self.job_completed_at else ""),
            "total_duration_seconds": (str(self.total_duration_seconds) if self.total_duration_seconds else ""),
            "actual_processing_seconds": (
                str(self.actual_processing_seconds) if self.actual_processing_seconds else ""
            ),
        }

    @property
    def total_duration_seconds(self) -> float | None:
        """Total time from submission to completion."""
        if self.job_submitted_at and self.job_completed_at:
            return (self.job_completed_at - self.job_submitted_at).total_seconds()
        return None

    @property
    def actual_processing_seconds(self) -> float | None:
        """Actual processing time (from start to completion)."""
        if self.job_started_at and self.job_completed_at:
            return (self.job_completed_at - self.job_started_at).total_seconds()
        return None


# CostInfo class has been moved to src/inference/pricing.py


def estimate_batch_cost_with_api(
    client: genai.Client,
    model: str,
    prompt: str,
    image_paths: list[Path],
    output_tokens_per_image: int = AVERAGE_OUTPUT_TOKENS,
    use_vertex_ai: bool = True,
    sample_size: int = 5,
) -> dict[str, Any]:
    """
    Estimate the cost of a batch job using actual API token counting.

    This method samples a few images and uses the Gemini count_tokens API
    for accurate estimation.

    Args:
        client: GenAI client
        model: Model name
        prompt: Full prompt text
        image_paths: List of image paths to process
        output_tokens_per_image: Estimated output tokens per image
        use_vertex_ai: Whether using Vertex AI pricing
        sample_size: Number of images to sample for token counting

    Returns:
        Dictionary with estimated costs
    """
    # Use API to count tokens
    token_stats = count_tokens_for_batch(
        client=client,
        model=model,
        prompt=prompt,
        image_paths=image_paths,
        sample_size=sample_size,
    )

    total_input_tokens = token_stats["total_input_tokens"]
    total_output_tokens = output_tokens_per_image * len(image_paths)

    # Create CostInfo and calculate
    cost_info = CostInfo(
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        model=model,
        is_batch=True,
        use_vertex_ai=use_vertex_ai,
    )
    cost_info.calculate_costs()

    return {
        "num_images": len(image_paths),
        "estimated_input_tokens": total_input_tokens,
        "estimated_output_tokens": total_output_tokens,
        "estimated_total_tokens": total_input_tokens + total_output_tokens,
        "tokens_per_request": token_stats.get("tokens_per_request", 0),
        "prompt_tokens": token_stats.get("prompt_tokens", 0),
        "avg_image_tokens": token_stats.get("avg_image_tokens", 0),
        "output_tokens_per_image": output_tokens_per_image,
        "estimated_input_cost_usd": cost_info.input_cost_usd,
        "estimated_output_cost_usd": cost_info.output_cost_usd,
        "estimated_total_cost_usd": cost_info.total_cost_usd,
        "estimated_total_cost_krw": cost_info.total_cost_krw,
        "model": model,
        "use_vertex_ai": use_vertex_ai,
        "batch_discount_applied": True,
        "estimation_method": "api_sampling",
        "sampled_images": token_stats.get("sampled_count", 0),
    }


def load_prompt(prompt_path: Path) -> str:
    """Load prompt from file."""
    with open(prompt_path, encoding="utf-8") as f:
        return f.read()


def encode_image_to_base64(image_path: Path) -> tuple[str, str]:
    """Encode image to base64 and return with mime type."""
    suffix = image_path.suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    mime_type = mime_types.get(suffix, "image/jpeg")

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    return image_data, mime_type


def create_batch_request(
    image_path: Path,
    prompt: str,
    request_key: str,
    max_tokens: int = 1000,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """
    Create a single batch request for Gemini API (JSONL file format) with base64 image.

    Args:
        image_path: Path to the image file
        prompt: Prompt text
        request_key: Unique key for this request (used to match responses)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Dictionary representing a batch request in JSONL format
    """
    image_data, mime_type = encode_image_to_base64(image_path)

    return {
        "key": request_key,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"inline_data": {"mime_type": mime_type, "data": image_data}},
                        {"text": prompt},
                    ],
                }
            ],
            "generation_config": {
                "max_output_tokens": max_tokens,
                "temperature": temperature,
                "response_mime_type": "application/json",
            },
        },
    }


def create_batch_request_with_gcs_uri(
    gcs_uri: str,
    prompt: str,
    request_key: str,
    max_tokens: int = 1000,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """
    Create a single batch request for Gemini API using GCS URI (no base64 encoding).

    This is more efficient for images already in GCS as it avoids downloading
    and re-encoding the images.

    Args:
        gcs_uri: GCS URI of the image (gs://bucket/path/to/image.jpg)
        prompt: Prompt text
        request_key: Unique key for this request (used to match responses)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Dictionary representing a batch request in JSONL format
    """
    mime_type = get_mime_type_from_extension(gcs_uri)

    return {
        "key": request_key,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"file_data": {"mime_type": mime_type, "file_uri": gcs_uri}},
                        {"text": prompt},
                    ],
                }
            ],
            "generation_config": {
                "max_output_tokens": max_tokens,
                "temperature": temperature,
                "response_mime_type": "application/json",
            },
        },
    }


def create_inline_batch_request(
    image_path: Path,
    prompt: str,
    max_tokens: int = 1000,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """
    Create a single inline batch request for Vertex AI Batch API.

    This format is used when passing requests directly to client.batches.create(src=[...])
    instead of uploading a JSONL file.

    Args:
        image_path: Path to the image file
        prompt: Prompt text
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Dictionary representing an inline batch request
    """
    image_data, mime_type = encode_image_to_base64(image_path)

    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": mime_type, "data": image_data}},
                    {"text": prompt},
                ],
            }
        ],
        "config": {
            "response_modalities": ["TEXT"],
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "response_mime_type": "application/json",
        },
    }


def prepare_inline_batch_requests(
    image_data: list[dict[str, str]],
    images_dir: Path,
    prompt: str,
    max_tokens: int = 1000,
    temperature: float = 0.0,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """
    Prepare inline batch requests for Vertex AI Batch API.

    Args:
        image_data: List of dictionaries with image metadata
        images_dir: Directory containing images
        prompt: Prompt text
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Tuple of (list of inline requests, list of request keys, list of skipped file names)
    """
    inline_requests = []
    request_keys = []
    skipped_files = []

    for data in image_data:
        file_name = data.get("file_name", "")
        if not file_name:
            continue

        image_path = images_dir / file_name
        if not image_path.exists():
            skipped_files.append(file_name)
            continue

        request = create_inline_batch_request(
            image_path=image_path,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        inline_requests.append(request)
        request_keys.append(file_name)

    return inline_requests, request_keys, skipped_files


def prepare_batch_requests_file(
    image_data: list[dict[str, str]],
    images_dir: Path,
    prompt: str,
    output_jsonl: Path,
    max_tokens: int = 1000,
    temperature: float = 0.0,
) -> tuple[list[str], list[str]]:
    """
    Prepare batch requests JSONL file.

    Args:
        image_data: List of dictionaries with image metadata
        images_dir: Directory containing images
        prompt: Prompt text
        output_jsonl: Output path for JSONL file
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Tuple of (list of request keys, list of skipped file names)
    """
    request_keys = []
    skipped_files = []

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for data in image_data:
            file_name = data.get("file_name", "")
            if not file_name:
                continue

            image_path = images_dir / file_name
            if not image_path.exists():
                skipped_files.append(file_name)
                continue

            request_key = file_name  # Use filename as key for easy matching
            request = create_batch_request(
                image_path=image_path,
                prompt=prompt,
                request_key=request_key,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            f.write(json.dumps(request) + "\n")
            request_keys.append(request_key)

    return request_keys, skipped_files


def prepare_batch_requests_file_from_gcs(
    gcs_image_uris: list[str],
    prompt: str,
    output_jsonl: Path,
    max_tokens: int = 1000,
    temperature: float = 0.0,
) -> list[str]:
    """
    Prepare batch requests JSONL file using GCS image URIs.

    This is more efficient for images already in GCS as it uses file_uri
    instead of downloading and base64 encoding the images.

    Args:
        gcs_image_uris: List of GCS URIs for images
        prompt: Prompt text
        output_jsonl: Output path for JSONL file
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        List of request keys (file names)
    """
    request_keys = []

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for gcs_uri in gcs_image_uris:
            # Use filename as key for easy matching
            file_name = Path(gcs_uri).name
            request_key = file_name

            request = create_batch_request_with_gcs_uri(
                gcs_uri=gcs_uri,
                prompt=prompt,
                request_key=request_key,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            f.write(json.dumps(request) + "\n")
            request_keys.append(request_key)

    return request_keys


def poll_batch_job(
    client: genai.Client,
    job_name: str,
    poll_interval: int = 30,
    max_wait: int = 86400,  # 24 hours
) -> tuple[Any, BatchTimingInfo]:
    """
    Poll batch job status until completion.

    Args:
        client: Gemini client
        job_name: Name of the batch job
        poll_interval: Seconds between status checks
        max_wait: Maximum seconds to wait

    Returns:
        Tuple of (final job object, timing info)
    """
    timing = BatchTimingInfo(job_submitted_at=datetime.now(UTC))

    completed_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }

    elapsed = 0
    job = None
    job_started = False

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=1,
    ) as progress:
        task = progress.add_task("Waiting for batch job...", total=100)

        while elapsed < max_wait:
            try:
                job = client.batches.get(name=job_name)
                state = str(job.state) if hasattr(job, "state") else "UNKNOWN"

                # Check if job started processing
                if not job_started and state == "JOB_STATE_RUNNING":
                    timing.job_started_at = datetime.now(UTC)
                    job_started = True
                    progress.update(task, description="Batch job processing...")

                # Check for completion
                if state in completed_states:
                    timing.job_completed_at = datetime.now(UTC)
                    progress.update(task, completed=100, description=f"Batch job {state}")
                    break

                # Update progress based on elapsed time (estimate)
                progress_pct = min(95, (elapsed / max_wait) * 100)
                progress.update(
                    task,
                    completed=progress_pct,
                    description=f"Batch job {state} ({elapsed}s elapsed)",
                )

            except Exception as e:
                console.print(f"[yellow]Warning: Error checking job status: {e}[/yellow]")

            time.sleep(poll_interval)
            elapsed += poll_interval

    return job, timing


def parse_batch_results(
    client: genai.Client,
    job: Any,
    request_keys: list[str],
    use_vertex_ai: bool = True,
) -> tuple[dict[str, dict[str, Any]], CostInfo]:
    """
    Parse batch job results.

    Args:
        client: Gemini client
        job: Completed batch job object
        request_keys: List of request keys to match (used for inline requests without keys)
        use_vertex_ai: Whether using Vertex AI (affects pricing)

    Returns:
        Tuple of (results dict mapping key to result, cost info)
    """
    results: dict[str, dict[str, Any]] = {}
    cost_info = CostInfo(
        model=str(job.model) if hasattr(job, "model") else "unknown",
        use_vertex_ai=use_vertex_ai,
    )

    # Get results from the job
    if hasattr(job, "dest") and job.dest:
        # Results are stored in destination
        dest = job.dest
        if hasattr(dest, "inlined_responses") and dest.inlined_responses:
            # For inline requests, responses may not have keys
            # We match by index with request_keys
            for idx, response in enumerate(dest.inlined_responses):
                # Try to get key from response, fallback to request_keys by index
                key = ""
                if hasattr(response, "key") and response.key:
                    key = response.key
                elif idx < len(request_keys):
                    key = request_keys[idx]
                else:
                    key = f"request_{idx}"

                result_data: dict[str, Any] = {
                    "success": False,
                    "result": None,
                    "error": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }

                if hasattr(response, "response") and response.response:
                    resp = response.response
                    # Extract text content
                    if hasattr(resp, "candidates") and resp.candidates:
                        candidate = resp.candidates[0]
                        if hasattr(candidate, "content") and candidate.content:
                            parts = candidate.content.parts
                            if parts:
                                text = parts[0].text if hasattr(parts[0], "text") else ""
                                try:
                                    # Parse JSON response
                                    result_data["result"] = json.loads(text)
                                    result_data["success"] = True
                                except json.JSONDecodeError:
                                    result_data["result"] = {"raw_text": text}
                                    result_data["success"] = True

                    # Extract usage metadata
                    if hasattr(resp, "usage_metadata") and resp.usage_metadata:
                        usage = resp.usage_metadata
                        result_data["input_tokens"] = getattr(usage, "prompt_token_count", 0) or 0
                        result_data["output_tokens"] = getattr(usage, "candidates_token_count", 0) or 0
                        cost_info.input_tokens += result_data["input_tokens"]
                        cost_info.output_tokens += result_data["output_tokens"]

                if hasattr(response, "error") and response.error:
                    result_data["error"] = str(response.error)
                    result_data["success"] = False

                results[key] = result_data

    # Calculate costs
    cost_info.calculate_costs()

    return results, cost_info


def create_csv_row_from_result(
    file_name: str,
    original_data: dict[str, str],
    result: dict[str, Any],
    model_name: str,
    prompt_version: str,
) -> dict[str, Any]:
    """Create a CSV row from batch result."""
    row: dict[str, Any] = {
        **original_data,
        "file_name": file_name,
        "model": model_name,
        "prompt_version": prompt_version,
        "success": result.get("success", False),
        "input_tokens": result.get("input_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
    }

    if not result.get("success"):
        row["error"] = result.get("error", "unknown_error")
        row["image_type"] = ""
        row["area_name"] = ""
        row["sub_area"] = ""
        row["contamination_type"] = ""
        row["max_severity"] = ""
        row["raw_response"] = ""
        return row

    parsed = result.get("result", {})

    # Extract image_type
    row["image_type"] = parsed.get("image_type", "")

    # Extract areas array
    areas = parsed.get("areas", [])

    # Flatten contaminations
    all_contaminations = []
    max_severity_level = -1
    max_severity_str = "Level 0"

    for area in areas:
        area_name = area.get("area_name", "")
        sub_area = area.get("sub_area", "")
        contaminations = area.get("contaminations", [])

        for contamination in contaminations:
            cont_type = contamination.get("contamination_type", "")
            severity = contamination.get("severity", "Level 0")

            all_contaminations.append(
                {
                    "area_name": area_name,
                    "sub_area": sub_area,
                    "contamination_type": cont_type,
                    "severity": severity,
                }
            )

            try:
                level_num = int(severity.replace("Level ", ""))
                if level_num > max_severity_level:
                    max_severity_level = level_num
                    max_severity_str = severity
            except (ValueError, AttributeError):
                pass

    if not all_contaminations:
        row["area_name"] = ""
        row["sub_area"] = ""
        row["contamination_type"] = ""
        row["max_severity"] = "Level 0"
        row["raw_response"] = str(parsed)
        return row

    # Aggregate
    area_names = sorted({c["area_name"] for c in all_contaminations if c["area_name"]})
    sub_areas = sorted({c["sub_area"] for c in all_contaminations if c["sub_area"]})
    cont_types = sorted({c["contamination_type"] for c in all_contaminations if c["contamination_type"]})

    row["area_name"] = ", ".join(area_names)
    row["sub_area"] = ", ".join(sub_areas)
    row["contamination_type"] = ", ".join(cont_types)
    row["max_severity"] = max_severity_str
    row["raw_response"] = str(parsed)

    return row


def run_gemini_batch_inference(
    input_csv: Path | None,
    images_dir: Path | str,
    prompt_path: Path,
    output_csv: Path,
    model: str | None = None,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    skip_confirmation: bool = False,
    limit: int | None = None,
    poll_interval: int = 30,
    gcs_bucket: str | None = None,
) -> dict[str, Any]:
    """
    Run batch inference using Gemini Batch API.

    Args:
        input_csv: CSV file containing image filenames (optional if using GCS path)
        images_dir: Directory containing images (local path or GCS path like gs://bucket/path)
        prompt_path: Path to prompt file
        output_csv: Output CSV file path
        model: Model name (defaults to env MODEL_NAME)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        limit: Maximum number of images to process
        poll_interval: Seconds between status checks
        gcs_bucket: GCS bucket name for Vertex AI batch jobs (defaults to env GCS_BUCKET_NAME)

    Returns:
        Dictionary with summary statistics

    Notes:
        - If images_dir is a GCS path (gs://... or bucket/path), images will be read directly from GCS
        - If input_csv is None and images_dir is a GCS path, all images in the GCS path will be processed
        - GCS images use file_uri in batch requests (more efficient than base64)
    """
    # Get configuration from environment
    api_key = os.getenv("GOOGLE_API_KEY")
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "FALSE").upper() == "TRUE"

    if model is None:
        model = os.getenv("MODEL_NAME", "gemini-2.5-flash")

    # Initialize client
    if use_vertex:
        # Vertex AI mode
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        # Gemini 3 models require global endpoint
        # Reference: https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/3-pro
        if model and model.startswith("gemini-3"):
            location = "global"
            console.print(f"[yellow]Note: {model} requires global endpoint, using location='global'[/yellow]")

        # Try to get project from service account if not set
        if not project:
            # Try multiple possible environment variable names
            creds_path_str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if creds_path_str:
                creds_path = _resolve_credentials_path(creds_path_str)
                if creds_path and creds_path.exists():
                    try:
                        with open(creds_path, encoding="utf-8") as f:
                            creds_data = json.load(f)
                            project = creds_data.get("project_id")
                    except (OSError, json.JSONDecodeError):
                        pass  # Ignore credential file read errors, will check project below

        if not project:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT environment variable is required for Vertex AI. "
                "Set it in your .env file or ensure your service account JSON has project_id."
            )

        client = genai.Client(vertexai=True, project=project, location=location)
        console.print(f"[cyan]Using Vertex AI (project: {project}, location: {location})[/cyan]")
    else:
        # Gemini Developer API mode
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        client = genai.Client(api_key=api_key)
        console.print("[cyan]Using Gemini Developer API[/cyan]")

    # Load prompt
    prompt = load_prompt(prompt_path)
    prompt_version = prompt_path.stem

    # Check if images_dir is a GCS path
    images_dir_str = str(images_dir)
    use_gcs_images = images_dir_str.startswith("gs://") or (
        "/" in images_dir_str and not images_dir_str.startswith("/") and not Path(images_dir_str).exists()
    )

    # Variables for GCS mode
    gcs_image_uris: list[str] = []
    image_data: list[dict[str, str]] = []
    input_csv_columns: list[str] = []

    if use_gcs_images:
        # GCS mode: list images from GCS bucket
        console.print(f"[cyan]Loading images from GCS: {images_dir_str}[/cyan]")

        # Normalize GCS path
        if not images_dir_str.startswith("gs://"):
            images_dir_str = f"gs://{images_dir_str}"

        gcs_image_uris = list_gcs_images(images_dir_str)
        console.print(f"[green]Found {len(gcs_image_uris)} images in GCS[/green]")

        if not gcs_image_uris:
            raise ValueError(f"No images found in GCS path: {images_dir_str}")

        # If input_csv is provided, filter images by CSV
        if input_csv is not None and input_csv.exists():
            console.print(f"[cyan]Filtering images using CSV: {input_csv}[/cyan]")
            with open(input_csv, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                input_csv_columns = list(reader.fieldnames or [])
                csv_file_names = set()
                for row in reader:
                    file_name = row.get("file_name") or row.get("image_name", "")
                    if not file_name:
                        filename_val = row.get("filename", "")
                        if filename_val:
                            file_name = Path(filename_val).name
                    if file_name:
                        csv_file_names.add(file_name)
                        image_data.append({"file_name": file_name, **row})

            # Filter GCS URIs by CSV file names
            gcs_uri_map = {Path(uri).name: uri for uri in gcs_image_uris}
            filtered_uris = []
            for file_name in csv_file_names:
                if file_name in gcs_uri_map:
                    filtered_uris.append(gcs_uri_map[file_name])
            gcs_image_uris = filtered_uris
            console.print(f"[green]Filtered to {len(gcs_image_uris)} images matching CSV[/green]")
        else:
            # Use all GCS images
            image_data = [{"file_name": Path(uri).name} for uri in gcs_image_uris]
            input_csv_columns = ["file_name"]

    else:
        # Local mode: load from CSV and local directory
        if input_csv is None:
            raise ValueError("input_csv is required when using local images directory")

        images_dir = Path(images_dir)

        console.print(f"[cyan]Loading input CSV from {input_csv}[/cyan]")

        with open(input_csv, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            input_csv_columns = list(reader.fieldnames or [])
            for row in reader:
                file_name = row.get("file_name") or row.get("image_name", "")
                if not file_name:
                    filename_val = row.get("filename", "")
                    if filename_val:
                        file_name = Path(filename_val).name
                if not file_name:
                    url = row.get("file_url") or row.get("image_url", "")
                    if url:
                        parsed = urlparse(url)
                        file_name = Path(parsed.path).name
                if file_name:
                    row["file_name"] = file_name
                    image_data.append(dict(row))

    if not image_data and not gcs_image_uris:
        raise ValueError(f"No valid entries found in {input_csv}")

    if use_gcs_images:
        console.print(f"[green]Found {len(gcs_image_uris)} images to process from GCS[/green]")
    else:
        console.print(f"[green]Loaded {len(image_data)} entries from CSV[/green]")

    # Apply limit
    if limit is not None and limit > 0:
        if use_gcs_images:
            gcs_image_uris = gcs_image_uris[:limit]
            image_data = image_data[:limit] if image_data else [{"file_name": Path(uri).name} for uri in gcs_image_uris]
        else:
            image_data = image_data[:limit]
        console.print(f"[yellow]Processing first {limit} images (limit applied)[/yellow]")

    # Get valid image paths (files that exist) - only for local mode
    if use_gcs_images:
        # For GCS mode, we don't need to check local paths
        valid_image_paths: list[Path] = []
        num_images = len(gcs_image_uris)
    else:
        # images_dir is guaranteed to be Path here (converted in local mode branch above)
        images_dir_path = images_dir if isinstance(images_dir, Path) else Path(images_dir)
        valid_image_paths = [
            images_dir_path / data.get("file_name", "")
            for data in image_data
            if (images_dir_path / data.get("file_name", "")).exists()
        ]
        num_images = len(valid_image_paths)

    # Estimate and display cost before proceeding
    console.print()

    if use_gcs_images:
        # For GCS images, use simple estimation (can't sample without downloading)
        console.print("[bold yellow]━━━ Estimated Cost (Simple Estimation) ━━━[/bold yellow]")
        from src.inference.pricing import estimate_batch_cost

        # Get prompt token count
        try:
            response = client.models.count_tokens(model=model, contents=prompt)
            if hasattr(response, "total_tokens") and response.total_tokens:
                prompt_tokens = int(response.total_tokens)
            else:
                prompt_tokens = len(prompt) // 4
        except Exception:
            prompt_tokens = len(prompt) // 4

        cost_estimate = estimate_batch_cost(
            num_images=num_images,
            model=model,
            prompt_tokens=prompt_tokens,
            output_tokens_per_image=AVERAGE_OUTPUT_TOKENS,
            use_vertex_ai=use_vertex,
        )
        cost_estimate["sampled_images"] = 0
    else:
        # For local images, use API-based token counting with sampling
        console.print("[bold yellow]━━━ Estimated Cost (API Token Counting) ━━━[/bold yellow]")
        cost_estimate = estimate_batch_cost_with_api(
            client=client,
            model=model,
            prompt=prompt,
            image_paths=valid_image_paths,
            output_tokens_per_image=AVERAGE_OUTPUT_TOKENS,
            use_vertex_ai=use_vertex,
            sample_size=5,
        )

    estimate_table = Table(show_header=False, box=None)
    estimate_table.add_column("Key", style="cyan", width=30)
    estimate_table.add_column("Value", style="white")

    estimate_table.add_row("📷 Images to process", str(cost_estimate["num_images"]))
    estimate_table.add_row("📝 Prompt tokens", f"{cost_estimate['prompt_tokens']:,}")
    estimate_table.add_row("🖼️  Avg image tokens", f"{cost_estimate['avg_image_tokens']:,}")
    estimate_table.add_row("📥 Tokens per request", f"{cost_estimate['tokens_per_request']:,}")
    estimate_table.add_row("📤 Output tokens/image", f"~{cost_estimate['output_tokens_per_image']:,}")
    estimate_table.add_row("", "")
    estimate_table.add_row("📊 Total input tokens", f"{cost_estimate['estimated_input_tokens']:,}")
    estimate_table.add_row("📊 Total output tokens", f"~{cost_estimate['estimated_output_tokens']:,}")
    estimate_table.add_row("", "")
    estimate_table.add_row(
        "💵 Estimated Input Cost",
        f"[yellow]${cost_estimate['estimated_input_cost_usd']:.4f}[/yellow]",
    )
    estimate_table.add_row(
        "💵 Estimated Output Cost",
        f"[yellow]${cost_estimate['estimated_output_cost_usd']:.4f}[/yellow]",
    )
    estimate_table.add_row(
        "💰 Estimated Total (USD)",
        f"[bold green]${cost_estimate['estimated_total_cost_usd']:.4f}[/bold green]",
    )
    estimate_table.add_row(
        "💴 Estimated Total (KRW)",
        f"[bold green]₩{cost_estimate['estimated_total_cost_krw']:,.0f}[/bold green]",
    )
    estimate_table.add_row("", "")
    estimate_table.add_row(
        "🏷️  Pricing",
        "Vertex AI Batch (50% discount)" if use_vertex else "Developer API Batch",
    )
    estimate_table.add_row("🤖 Model", model)
    sampled_info = f"(sampled {cost_estimate.get('sampled_images', 0)} images)"
    estimate_table.add_row("📏 Estimation method", f"API token counting {sampled_info}")

    console.print(estimate_table)
    console.print()
    console.print("[dim]Note: Output token estimate is approximate. Actual costs from API response.[/dim]")
    console.print()

    # Ask for confirmation unless skipped
    if not skip_confirmation and not Confirm.ask(
        "[bold yellow]Do you want to proceed with the batch job?[/bold yellow]",
        default=True,
    ):
        console.print("[yellow]Batch job cancelled by user.[/yellow]")
        return {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "output_path": output_csv,
            "timing": {},
            "cost": {},
            "model": model,
            "cancelled": True,
        }

    console.print("[green]Proceeding with batch job...[/green]")
    console.print()

    # Prepare batch requests
    console.print("[cyan]Preparing batch requests...[/cyan]")

    # Variables for cleanup
    gcs_input_uri: str | None = None
    temp_jsonl_path: Path | None = None

    # skipped_files tracking
    skipped_files: list[str] = []

    if use_vertex:
        # Vertex AI: Must use GCS for input/output
        # 1. Create JSONL file locally
        # 2. Upload to GCS
        # 3. Create batch job with gcs_uri

        # Use provided gcs_bucket or fallback to environment variable
        if gcs_bucket is None:
            gcs_bucket = os.getenv("GCS_BUCKET_NAME")
        if not gcs_bucket:
            raise ValueError(
                "GCS bucket name is required for Vertex AI batch jobs. "
                "Provide it via --gcs-bucket option or set GCS_BUCKET_NAME in your .env file"
            )

        # Clean up bucket name - remove gs:// prefix and extract path if included
        if gcs_bucket.startswith("gs://"):
            gcs_bucket = gcs_bucket[5:]

        # Extract bucket prefix if user included path (e.g., "bucket/path" -> bucket="bucket", prefix="path")
        gcs_prefix = ""
        if "/" in gcs_bucket:
            parts = gcs_bucket.split("/", 1)
            gcs_bucket = parts[0]
            gcs_prefix = parts[1].rstrip("/") + "/"
            console.print(f"[dim]Using bucket '{gcs_bucket}' with prefix '{gcs_prefix}'[/dim]")

        # Create temp directory for JSONL file
        temp_dir = Path(tempfile.gettempdir()) / "gemini_batch"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique job ID
        job_id = f"batch_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        temp_jsonl_path = temp_dir / f"{job_id}_input.jsonl"

        if use_gcs_images:
            # GCS images mode: use file_uri in requests (no base64 encoding)
            console.print("[cyan]Using GCS file URIs for images (efficient mode)[/cyan]")
            request_keys = prepare_batch_requests_file_from_gcs(
                gcs_image_uris=gcs_image_uris,
                prompt=prompt,
                output_jsonl=temp_jsonl_path,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            # Local images mode: encode to base64
            # images_dir is already converted to Path in local mode branch (line 1309)
            images_dir_path = images_dir if isinstance(images_dir, Path) else Path(images_dir)
            request_keys, skipped_files = prepare_batch_requests_file(
                image_data=image_data,
                images_dir=images_dir_path,
                prompt=prompt,
                output_jsonl=temp_jsonl_path,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        if skipped_files:
            console.print(f"[yellow]Skipped {len(skipped_files)} files (not found)[/yellow]")

        console.print(f"[green]Prepared {len(request_keys)} batch requests in JSONL[/green]")

        # Upload to GCS
        gcs_blob_name = f"{gcs_prefix}gemini-batch/{job_id}/input.jsonl"
        console.print(f"[cyan]Uploading JSONL to GCS: gs://{gcs_bucket}/{gcs_blob_name}[/cyan]")
        gcs_input_uri = upload_to_gcs(temp_jsonl_path, gcs_bucket, gcs_blob_name)
        console.print(f"[green]Uploaded to {gcs_input_uri}[/green]")

        # Create batch job with GCS URI
        console.print("[cyan]Creating batch job...[/cyan]")
        batch_job = client.batches.create(
            model=model,
            src=gcs_input_uri,
        )

    else:
        # Gemini Developer API: Supports inline requests
        if use_gcs_images:
            raise ValueError(
                "GCS images are only supported with Vertex AI (GOOGLE_GENAI_USE_VERTEXAI=TRUE). "
                "For Gemini Developer API, use local images."
            )

        # images_dir is already converted to Path in local mode branch (line 1309)
        images_dir_path = images_dir if isinstance(images_dir, Path) else Path(images_dir)
        inline_requests, request_keys, skipped_files = prepare_inline_batch_requests(
            image_data=image_data,
            images_dir=images_dir_path,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if skipped_files:
            console.print(f"[yellow]Skipped {len(skipped_files)} files (not found)[/yellow]")

        console.print(f"[green]Prepared {len(request_keys)} batch requests[/green]")

        # Create batch job with inline requests
        console.print("[cyan]Creating batch job...[/cyan]")
        batch_job = client.batches.create(
            model=model,
            src=inline_requests,
        )

    job_name = batch_job.name
    if not job_name:
        raise RuntimeError("Batch job creation failed: no job name returned")
    console.print(f"[green]Created batch job: {job_name}[/green]")

    # Save job info to file for later resume
    job_info_file = output_csv.with_suffix(".job.json")
    job_info = {
        "job_name": job_name,
        "model": model,
        "output_csv": str(output_csv),
        "prompt_version": prompt_version,
        "num_requests": len(request_keys),
        "request_keys": request_keys,
        "created_at": datetime.now(UTC).isoformat(),
        "use_vertex_ai": use_vertex,
    }
    with open(job_info_file, "w", encoding="utf-8") as f:
        json.dump(job_info, f, indent=2)
    console.print(f"[cyan]Job info saved to: {job_info_file}[/cyan]")
    console.print(f"[cyan]To resume later: carwash resume-job {job_info_file}[/cyan]")

    # Poll for completion
    console.print("[cyan]Waiting for batch job to complete (this may take up to 24 hours)...[/cyan]")
    final_job, timing_info = poll_batch_job(
        client=client,
        job_name=job_name,
        poll_interval=poll_interval,
    )

    # Check job status
    job_state = str(final_job.state) if hasattr(final_job, "state") else "UNKNOWN"
    if job_state != "JOB_STATE_SUCCEEDED":
        console.print(f"[red]Batch job failed with state: {job_state}[/red]")
        raise RuntimeError(f"Batch job failed: {job_state}")

    console.print("[green]Batch job completed successfully![/green]")

    # Parse results
    console.print("[cyan]Parsing batch results...[/cyan]")
    results, cost_info = parse_batch_results(client, final_job, request_keys, use_vertex_ai=use_vertex)

    # Display actual costs from API response
    console.print()
    console.print("[bold green]━━━ Actual Costs (from API Response) ━━━[/bold green]")

    actual_cost_table = Table(show_header=False, box=None)
    actual_cost_table.add_column("Key", style="cyan", width=30)
    actual_cost_table.add_column("Value", style="white")

    actual_cost_table.add_row("📊 Total input tokens", f"{cost_info.input_tokens:,}")
    actual_cost_table.add_row("📊 Total output tokens", f"{cost_info.output_tokens:,}")
    actual_cost_table.add_row("📊 Total tokens", f"{cost_info.input_tokens + cost_info.output_tokens:,}")
    actual_cost_table.add_row("", "")
    actual_cost_table.add_row(
        "💵 Input Cost",
        f"[yellow]${cost_info.input_cost_usd:.6f}[/yellow]",
    )
    actual_cost_table.add_row(
        "💵 Output Cost",
        f"[yellow]${cost_info.output_cost_usd:.6f}[/yellow]",
    )
    actual_cost_table.add_row(
        "💰 Total Cost (USD)",
        f"[bold green]${cost_info.total_cost_usd:.6f}[/bold green]",
    )
    actual_cost_table.add_row(
        "💴 Total Cost (KRW)",
        f"[bold green]₩{cost_info.total_cost_krw:,.0f}[/bold green]",
    )

    # Compare with estimate
    actual_cost_table.add_row("", "")
    actual_cost_table.add_row("[bold]━━━ Comparison with Estimate ━━━[/bold]", "")
    estimated_input = cost_estimate["estimated_input_tokens"]
    estimated_output = cost_estimate["estimated_output_tokens"]
    actual_input = cost_info.input_tokens
    actual_output = cost_info.output_tokens

    input_diff = actual_input - estimated_input
    output_diff = actual_output - estimated_output
    cost_diff = cost_info.total_cost_usd - cost_estimate["estimated_total_cost_usd"]

    input_diff_pct = (input_diff / estimated_input * 100) if estimated_input > 0 else 0
    output_diff_pct = (output_diff / estimated_output * 100) if estimated_output > 0 else 0
    estimated_cost_usd = cost_estimate["estimated_total_cost_usd"]
    cost_diff_pct = (cost_diff / estimated_cost_usd * 100) if estimated_cost_usd > 0 else 0

    input_diff_str = f"+{input_diff:,}" if input_diff >= 0 else f"{input_diff:,}"
    output_diff_str = f"+{output_diff:,}" if output_diff >= 0 else f"{output_diff:,}"
    cost_diff_str = f"+${cost_diff:.6f}" if cost_diff >= 0 else f"-${abs(cost_diff):.6f}"

    actual_cost_table.add_row(
        "📥 Input tokens diff",
        f"{input_diff_str} ({input_diff_pct:+.1f}%)",
    )
    actual_cost_table.add_row(
        "📤 Output tokens diff",
        f"{output_diff_str} ({output_diff_pct:+.1f}%)",
    )
    actual_cost_table.add_row(
        "💰 Cost diff",
        f"{cost_diff_str} ({cost_diff_pct:+.1f}%)",
    )

    console.print(actual_cost_table)
    console.print()

    # Create output CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Prepare fieldnames
    fieldnames = list(input_csv_columns)
    inference_columns = [
        "file_name",
        "model",
        "prompt_version",
        "success",
        "error",
        "image_type",
        "area_name",
        "sub_area",
        "contamination_type",
        "max_severity",
        "input_tokens",
        "output_tokens",
        "raw_response",
    ]
    for col in inference_columns:
        if col not in fieldnames:
            fieldnames.append(col)

    # Write results
    successful_count = 0
    failed_count = 0

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Create lookup for original data
        data_lookup = {d["file_name"]: d for d in image_data}

        for key, result in results.items():
            original_data = data_lookup.get(key, {"file_name": key})
            row = create_csv_row_from_result(
                file_name=key,
                original_data=original_data,
                result=result,
                model_name=model,
                prompt_version=prompt_version,
            )
            writer.writerow(row)

            if result.get("success"):
                successful_count += 1
            else:
                failed_count += 1

        # Write skipped files
        for file_name in skipped_files:
            original_data = data_lookup.get(file_name, {"file_name": file_name})
            row = {
                **original_data,
                "file_name": file_name,
                "model": model,
                "prompt_version": prompt_version,
                "success": False,
                "error": "file_not_found",
                "image_type": "",
                "area_name": "",
                "sub_area": "",
                "contamination_type": "",
                "max_severity": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "raw_response": "",
            }
            writer.writerow(row)
            failed_count += 1

    # Cleanup temporary files
    if temp_jsonl_path and temp_jsonl_path.exists():
        try:
            temp_jsonl_path.unlink()
            console.print(f"[dim]Cleaned up temp file: {temp_jsonl_path}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not delete temp file: {e}[/yellow]")

    # Optionally cleanup GCS input file (keep for debugging by default)
    # Uncomment below to auto-delete GCS input after successful completion
    # if gcs_input_uri:
    #     try:
    #         delete_gcs_blob(gcs_input_uri)
    #         console.print(f"[dim]Cleaned up GCS input: {gcs_input_uri}[/dim]")
    #     except Exception as e:
    #         console.print(f"[yellow]Warning: Could not delete GCS input: {e}[/yellow]")

    # Prepare summary
    return {
        "total": len(request_keys) + len(skipped_files),
        "successful": successful_count,
        "failed": failed_count,
        "output_path": output_csv,
        "timing": timing_info.to_dict(),
        "cost": cost_info.to_dict(),
        "model": model,
    }
