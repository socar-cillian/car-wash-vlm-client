"""Custom exceptions for the VLM client."""


class VLMClientError(Exception):
    """Base exception for VLM client errors."""

    pass


class ImageNotFoundError(VLMClientError):
    """Raised when an image file is not found."""

    def __init__(self, image_path: str):
        self.image_path = image_path
        super().__init__(f"Image file not found: {image_path}")


class PromptNotFoundError(VLMClientError):
    """Raised when a prompt file is not found."""

    def __init__(self, prompt_path: str):
        self.prompt_path = prompt_path
        super().__init__(f"Prompt file not found: {prompt_path}")


class InvalidImageFormatError(VLMClientError):
    """Raised when an image file has an unsupported format."""

    def __init__(self, image_path: str, supported_formats: list[str]):
        self.image_path = image_path
        self.supported_formats = supported_formats
        super().__init__(f"Invalid image format: {image_path}. Supported formats: {', '.join(supported_formats)}")
