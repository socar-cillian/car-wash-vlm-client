"""VLM Client for interacting with Vision Language Model API."""

from __future__ import annotations

import base64
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .exceptions import ImageNotFoundError, InvalidImageFormatError, PromptNotFoundError


class VLMClient:
    """Client for interacting with VLM (Vision Language Model) API."""

    SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

    def __init__(
        self,
        api_url: str,
        model: str = "qwen3-vl-8b-instruct",
    ):
        self.api_url = api_url
        self.model = model

        # Setup session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,  # Total number of retries
            backoff_factor=1,  # Wait 1, 2, 4 seconds between retries
            status_forcelist=[500, 502, 503, 504],  # Retry on server errors
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def check_health(self, timeout: int = 10) -> dict:
        """
        Check if the VLM API server is healthy and accessible.

        Args:
            timeout: Timeout in seconds for the health check

        Returns:
            Dictionary with health check results:
            {
                "healthy": bool,
                "status_code": int or None,
                "error": str or None,
                "response_time": float (in seconds)
            }

        Raises:
            Does not raise exceptions - returns error information in dict
        """

        start_time = time.time()

        try:
            # Try to get the models endpoint first (common in OpenAI-compatible APIs)
            base_url = self.api_url.rsplit("/v1/", 1)[0] if "/v1/" in self.api_url else self.api_url
            health_endpoints = [
                f"{base_url}/health",
                f"{base_url}/v1/models",
                f"{base_url}/healthz",
            ]

            last_error = None
            for endpoint in health_endpoints:
                try:
                    response = self.session.get(endpoint, timeout=timeout)
                    response_time = time.time() - start_time

                    if response.status_code == 200:
                        return {
                            "healthy": True,
                            "status_code": response.status_code,
                            "error": None,
                            "response_time": response_time,
                            "endpoint": endpoint,
                        }
                    last_error = f"Status code: {response.status_code}"
                except requests.exceptions.RequestException as e:
                    last_error = str(e)
                    continue

            # If all endpoints failed, return the last error
            response_time = time.time() - start_time
            return {
                "healthy": False,
                "status_code": None,
                "error": last_error or "All health check endpoints failed",
                "response_time": response_time,
            }

        except Exception as e:
            response_time = time.time() - start_time
            return {
                "healthy": False,
                "status_code": None,
                "error": str(e),
                "response_time": response_time,
            }

    def _load_prompt(self, prompt_input: str) -> str:
        """Load prompt from text file or use as direct string."""
        # Check if it's a very long string (likely already loaded prompt text)
        # or contains newlines (likely direct prompt text)
        if len(prompt_input) > 500 or "\n" in prompt_input:
            return prompt_input

        # If it looks like a file path, try to load it
        if "/" in prompt_input or "\\" in prompt_input or prompt_input.endswith(".txt"):
            prompt_path = Path(prompt_input)
            if not prompt_path.exists():
                raise PromptNotFoundError(prompt_input)
            if not prompt_path.is_file():
                raise PromptNotFoundError(prompt_input)

            with open(prompt_path, encoding="utf-8") as f:
                return f.read().strip()

        # Otherwise, treat it as a direct string
        return prompt_input

    def _prepare_image(self, image_input: str) -> str:
        """Prepare image URL or convert local file to base64 data URL."""
        # Check if it's a URL
        if image_input.startswith(("http://", "https://")):
            return image_input

        # Handle local file
        image_path = Path(image_input)
        if not image_path.exists():
            raise ImageNotFoundError(image_input)

        if not image_path.is_file():
            raise ImageNotFoundError(image_input)

        # Validate image format
        suffix = image_path.suffix.lower()
        if suffix not in self.SUPPORTED_IMAGE_FORMATS:
            raise InvalidImageFormatError(image_input, list(self.SUPPORTED_IMAGE_FORMATS))

        # Read and encode image to base64
        with open(image_path, "rb") as f:
            image_data = f.read()

        base64_image = base64.b64encode(image_data).decode("utf-8")

        # Determine MIME type
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime_type = mime_types.get(suffix, "image/jpeg")

        return f"data:{mime_type};base64,{base64_image}"

    def query(
        self,
        image_input: str,
        prompt_input: str,
        max_tokens: int = 8192,
        temperature: float = 0.1,
        response_format: dict = None,
        image_name: str | None = None,
    ) -> dict:
        """
        Send a query to the VLM API.

        Args:
            image_input: URL or local path to image
            prompt_input: Text prompt or path to .txt file containing prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            response_format: Response format configuration (e.g., {"type": "json_object"})
            image_name: Name of the image for logging purposes

        Returns:
            API response as dictionary

        Raises:
            ImageNotFoundError: If image file doesn't exist
            PromptNotFoundError: If prompt file doesn't exist
            InvalidImageFormatError: If image format is not supported
            requests.exceptions.RequestException: If API request fails
        """
        # Prepare image and prompt
        image_url = self._prepare_image(image_input)
        prompt = self._load_prompt(prompt_input)

        # Get image name for logging
        if image_name is None:
            image_name = Path(image_input).name if not image_input.startswith("http") else "url_image"

        # Build request payload
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Add response format if specified
        if response_format is not None:
            payload["response_format"] = response_format

        # Send request with retry and increased timeout
        headers = {"Content-Type": "application/json"}

        try:
            response = self.session.post(self.api_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result
        except requests.exceptions.HTTPError as e:
            # Try to extract error details from response
            error_detail = "Unknown error"
            try:
                if response.text:
                    error_json = response.json()
                    error_detail = error_json.get("detail", error_json.get("message", response.text[:200]))
            except Exception:
                error_detail = response.text[:200] if response.text else str(e)

            # Re-raise with more detailed error message
            raise requests.exceptions.HTTPError(
                f"{response.status_code} {response.reason}: {error_detail}",
                response=response,
            ) from e
