"""Parse Gemini Batch API output JSONL file.

Extracts only the response data, discarding the bulky request (base64 images).

Usage as CLI:
    python -m src.inference.batch_output_parser input.jsonl output.jsonl
    python -m src.inference.batch_output_parser input.jsonl  # prints to stdout

Usage as module:
    from src.inference.batch_output_parser import parse_batch_output
    results = parse_batch_output(input_path, output_path)
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ParsedBatchResult:
    """Parsed result from a single batch API response."""

    key: str
    status: str
    processed_time: str
    success: bool
    finish_reason: str
    result: dict | None
    error: str | None
    input_tokens: int
    output_tokens: int


@dataclass
class ParseSummary:
    """Summary of batch output parsing."""

    total_count: int
    success_count: int
    failed_count: int
    input_size_bytes: int
    output_size_bytes: int | None

    @property
    def reduction_percent(self) -> float | None:
        """Calculate size reduction percentage."""
        if self.output_size_bytes is not None and self.input_size_bytes > 0:
            return (1 - self.output_size_bytes / self.input_size_bytes) * 100
        return None


def parse_batch_output_line(data: dict) -> ParsedBatchResult:
    """Parse a single line of batch output.

    Args:
        data: Parsed JSON object from a single line of batch output JSONL

    Returns:
        ParsedBatchResult with extracted data
    """
    key = data.get("key", "")
    status = data.get("status", "")
    processed_time = data.get("processed_time", "")
    success = False
    finish_reason = ""
    result = None
    error = None
    input_tokens = 0
    output_tokens = 0

    # Extract response (the actual model output)
    response = data.get("response", {})
    if response:
        candidates = response.get("candidates", [])
        if candidates:
            candidate = candidates[0]
            finish_reason = candidate.get("finishReason", "")

            # Check if response was truncated
            if finish_reason == "MAX_TOKENS":
                error = "response_truncated"
            elif finish_reason == "STOP":
                # Get the generated text
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                if parts and "text" in parts[0]:
                    text = parts[0]["text"]
                    try:
                        result = json.loads(text)
                        success = True
                    except json.JSONDecodeError:
                        result = {"raw_text": text}
                        success = True
            else:
                error = f"unexpected_finish_reason: {finish_reason}"

        # Get usage metadata
        usage = response.get("usageMetadata", {})
        input_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)
    else:
        error = "no_response"

    return ParsedBatchResult(
        key=key,
        status=status,
        processed_time=processed_time,
        success=success,
        finish_reason=finish_reason,
        result=result,
        error=error,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def parse_batch_output(
    input_path: Path,
    output_path: Path | None = None,
    *,
    verbose: bool = True,
) -> tuple[list[ParsedBatchResult], ParseSummary]:
    """Parse batch output and extract only essential data.

    Args:
        input_path: Path to input JSONL file from Gemini Batch API
        output_path: Optional path to write parsed output JSONL
        verbose: Whether to print progress messages

    Returns:
        Tuple of (list of parsed results, parse summary)
    """
    results: list[ParsedBatchResult] = []
    success_count = 0
    failed_count = 0

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)
            parsed = parse_batch_output_line(data)
            results.append(parsed)

            if parsed.success:
                success_count += 1
            else:
                failed_count += 1

    input_size = input_path.stat().st_size
    output_size = None

    # Output
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            for item in results:
                item_dict = {
                    "key": item.key,
                    "status": item.status,
                    "processed_time": item.processed_time,
                    "success": item.success,
                    "finish_reason": item.finish_reason,
                    "result": item.result,
                    "error": item.error,
                    "input_tokens": item.input_tokens,
                    "output_tokens": item.output_tokens,
                }
                f.write(json.dumps(item_dict, ensure_ascii=False) + "\n")

        output_size = output_path.stat().st_size

        if verbose:
            reduction = (1 - output_size / input_size) * 100 if input_size > 0 else 0
            print(f"Parsed {len(results)} results -> {output_path}")
            print(f"  Success: {success_count}, Failed: {failed_count}")
            print(
                f"  Size: {input_size / 1024 / 1024:.1f}MB -> "
                f"{output_size / 1024 / 1024:.1f}MB ({reduction:.1f}% reduction)"
            )
    elif verbose:
        for item in results:
            item_dict = {
                "key": item.key,
                "status": item.status,
                "processed_time": item.processed_time,
                "success": item.success,
                "finish_reason": item.finish_reason,
                "result": item.result,
                "error": item.error,
                "input_tokens": item.input_tokens,
                "output_tokens": item.output_tokens,
            }
            print(json.dumps(item_dict, ensure_ascii=False, indent=2))

    summary = ParseSummary(
        total_count=len(results),
        success_count=success_count,
        failed_count=failed_count,
        input_size_bytes=input_size,
        output_size_bytes=output_size,
    )

    return results, summary


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    parse_batch_output(input_file, output_file)


if __name__ == "__main__":
    main()
