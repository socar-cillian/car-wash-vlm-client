"""VLM Client for interacting with Vision Language Model API."""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

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

        # Setup session with retry logic and connection pooling for high throughput
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,  # Total number of retries
            backoff_factor=0.5,  # Wait 0.5, 1, 2 seconds between retries (faster)
            status_forcelist=[500, 502, 503, 504],  # Retry on server errors
        )
        # Increase connection pool size for parallel requests
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=32,  # Number of connection pools
            pool_maxsize=32,  # Max connections per pool
        )
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

    def get_model_info(self, timeout: int = 10) -> dict[str, Any] | None:
        """
        Get model information including max_model_len from the server.

        Args:
            timeout: Timeout in seconds for the request

        Returns:
            Dictionary with model info or None if request fails
        """
        try:
            base_url = self.api_url.rsplit("/v1/", 1)[0] if "/v1/" in self.api_url else self.api_url
            response = self.session.get(f"{base_url}/v1/models", timeout=timeout)
            if response.status_code == 200:
                data: dict[str, Any] = response.json()
                if "data" in data and len(data["data"]) > 0:
                    return dict(data["data"][0])
            return None
        except Exception:
            return None

    def get_server_replicas(self, namespace: str = "vllm-test", timeout: int = 5) -> int:
        """
        Get the number of available server replicas by querying Kubernetes endpoints.

        This method tries multiple approaches:
        1. Query Kubernetes API directly (if running inside K8s with proper RBAC)
        2. Make multiple health checks to estimate available backends

        Args:
            namespace: Kubernetes namespace where vLLM is deployed
            timeout: Timeout in seconds for each request

        Returns:
            Number of available replicas (minimum 1)
        """
        try:
            # Method 1: Try to query Kubernetes API (works inside K8s with proper ServiceAccount)
            import os

            token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"  # nosec B105
            ca_path = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"

            if os.path.exists(token_path) and os.path.exists(ca_path):
                with open(token_path) as f:
                    token = f.read().strip()

                k8s_host = os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
                k8s_port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")

                # Get deployment replicas
                service_name = f"{namespace}-qwen3-vl-8b-engine-service"
                endpoints_url = f"https://{k8s_host}:{k8s_port}/api/v1/namespaces/{namespace}/endpoints/{service_name}"

                response = self.session.get(
                    endpoints_url,
                    headers={"Authorization": f"Bearer {token}"},
                    verify=ca_path,
                    timeout=timeout,
                )

                if response.status_code == 200:
                    data = response.json()
                    # Count addresses in all subsets
                    total_addresses = 0
                    for subset in data.get("subsets", []):
                        total_addresses += len(subset.get("addresses", []))
                    if total_addresses > 0:
                        return total_addresses

        except Exception:  # nosec B110
            pass  # Fallback to default if K8s API not available

        # Method 2: Fallback - assume 1 replica if we can reach the server
        health = self.check_health(timeout=timeout)
        return 1 if health.get("healthy") else 0

    def get_recommended_workers(
        self, namespace: str = "vllm-test", workers_per_replica: int = 4, timeout: int = 5
    ) -> int:
        """
        Get the recommended number of workers based on server replicas.

        Args:
            namespace: Kubernetes namespace where vLLM is deployed
            workers_per_replica: Number of workers to allocate per GPU/replica
            timeout: Timeout in seconds for replica check

        Returns:
            Recommended number of workers (minimum 1)
        """
        replicas = self.get_server_replicas(namespace=namespace, timeout=timeout)
        recommended = max(1, replicas * workers_per_replica)
        return recommended

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
        response_format: dict[str, str] | None = None,
        image_name: str | None = None,
        prompt_mode: str = "system",
    ) -> dict[str, Any]:
        """
        Send a query to the VLM API.

        Args:
            image_input: URL or local path to image
            prompt_input: Text prompt or path to .txt file containing prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            response_format: Response format configuration (e.g., {"type": "json_object"})
            image_name: Name of the image for logging purposes
            prompt_mode: Where to place the prompt - "system" (enables vLLM prefix caching)
                        or "user" (traditional mode, prompt with image in user message)

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

        # Build request payload based on prompt_mode
        if prompt_mode == "system":
            # System prompt mode: enables vLLM prefix caching optimization
            # Prompt goes to system message, only image in user message
            messages = [
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ]
        else:
            # User prompt mode: traditional mode without prefix caching
            # Both prompt and image in user message
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

        payload = {
            "model": self.model,
            "messages": messages,
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
            result: dict[str, Any] = response.json()
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
