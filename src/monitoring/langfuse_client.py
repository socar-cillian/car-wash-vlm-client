"""Langfuse client for monitoring VLM inference."""

import os

from dotenv import load_dotenv
from langfuse import Langfuse


class LangfuseMonitor:
    """Langfuse monitoring client for VLM inference tracking."""

    def __init__(self, enabled: bool = True):
        """
        Initialize Langfuse monitor.

        Args:
            enabled: Whether to enable langfuse monitoring
        """
        self.enabled = enabled
        self.client: Langfuse | None = None

        if enabled:
            self._initialize_client()

    def _initialize_client(self):
        """Initialize Langfuse client with environment variables."""
        # Load environment variables
        load_dotenv()

        host = os.getenv("LANGFUSE_HOST")
        secret_key = os.getenv("LANGFUSE_CAR_WASH_VLM_SECRET_KEY")
        public_key = os.getenv("LANGFUSE_CAR_WASH_VLM_PUBLIC_KEY")

        # Check if all required environment variables are set
        if not all([host, secret_key, public_key]):
            print("Warning: Langfuse environment variables not set. Monitoring disabled.")
            self.enabled = False
            return

        try:
            self.client = Langfuse(
                host=host,
                secret_key=secret_key,
                public_key=public_key,
            )
            print("✓ Langfuse monitoring enabled")
        except Exception as e:
            print(f"Warning: Failed to initialize Langfuse client: {e}")
            self.enabled = False

    def start_trace_span(self, name: str, input_data: dict = None, metadata: dict = None):
        """
        Start a new trace span using context manager.

        Args:
            name: Name of the trace/span
            input_data: Input data for the trace
            metadata: Additional metadata

        Returns:
            Context manager for the span
        """
        if not self.enabled or not self.client:
            return None

        try:
            return self.client.start_as_current_span(
                name=name,
                input=input_data or {},
                metadata=metadata or {},
            )
        except Exception as e:
            print(f"Warning: Failed to start trace span: {e}")
            return None

    def start_generation(
        self,
        name: str,
        model: str,
        input_data: dict,
        model_parameters: dict = None,
        metadata: dict = None,
    ):
        """
        Start a generation using context manager.

        Args:
            name: Name of the generation
            model: Model name
            input_data: Input data
            model_parameters: Model parameters (temperature, max_tokens, etc.)
            metadata: Additional metadata

        Returns:
            Context manager for the generation
        """
        if not self.enabled or not self.client:
            return None

        try:
            params = {
                "name": name,
                "model": model,
                "input": input_data,
            }

            if model_parameters:
                params["model_parameters"] = model_parameters

            if metadata:
                params["metadata"] = metadata

            return self.client.start_as_current_generation(**params)
        except Exception as e:
            print(f"Warning: Failed to start generation: {e}")
            return None

    def flush(self):
        """Flush all pending events to Langfuse."""
        if self.enabled and self.client:
            try:
                self.client.flush()
            except Exception as e:
                print(f"Warning: Failed to flush Langfuse events: {e}")


# Global instance
_langfuse_monitor: LangfuseMonitor | None = None


def get_langfuse_monitor(enabled: bool = True) -> LangfuseMonitor:
    """
    Get or create the global Langfuse monitor instance.

    Args:
        enabled: Whether to enable monitoring

    Returns:
        LangfuseMonitor instance
    """
    global _langfuse_monitor

    if _langfuse_monitor is None:
        _langfuse_monitor = LangfuseMonitor(enabled=enabled)

    return _langfuse_monitor
