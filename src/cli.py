"""Command-line interface for the VLM client."""

import argparse
import json
import sys

import requests

from .client import VLMClient
from .exceptions import VLMClientError


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description="VLM Client for querying vision-language models")
    parser.add_argument("image", help="Image URL or local path to image file")
    parser.add_argument("prompt", help="Text prompt or path to .txt file containing prompt")
    parser.add_argument(
        "--api-url",
        default="http://vllm.mlops.socarcorp.co.kr/v1/chat/completions",
        help="VLM API endpoint URL",
    )
    parser.add_argument("--model", default="qwen3-vl-4b-instruct", help="Model name to use")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument(
        "--output",
        help="Output file path (optional, prints to stdout if not specified)",
    )
    parser.add_argument(
        "--json-mode",
        action="store_true",
        help="Enable JSON mode to force response in JSON format",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print full API response to stderr")

    return parser


def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    try:
        # Create client and send query
        client = VLMClient(api_url=args.api_url, model=args.model)
        print(f"Sending query to {args.api_url}...", file=sys.stderr)

        # Prepare response format
        response_format = {"type": "json_object"} if args.json_mode else None

        response = client.query(
            image_input=args.image,
            prompt_input=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            response_format=response_format,
        )

        # Extract response text
        response_text = response["choices"][0]["message"]["content"]

        # Output result
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(response_text)
            print(f"Response saved to {args.output}", file=sys.stderr)
        else:
            print("\n--- Response ---")
            print(response_text)

        # Print full JSON response to stderr if verbose
        if args.verbose:
            print("\n--- Full API Response ---", file=sys.stderr)
            print(json.dumps(response, indent=2, ensure_ascii=False), file=sys.stderr)

    except VLMClientError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"API Error: {e}", file=sys.stderr)
        if hasattr(e.response, "text"):
            print(f"Response: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Network Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"Unexpected API response format: missing key {e}", file=sys.stderr)
        print(f"Response: {json.dumps(response, indent=2)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
