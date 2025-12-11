"""
Vertex AI Gemini Pricing Module

This module contains pricing tables and cost calculation logic for Vertex AI Gemini models.
All prices are per 1 million (1M) tokens unless otherwise specified.

References:
- Vertex AI Pricing: https://cloud.google.com/vertex-ai/generative-ai/pricing
- Gemini 3 Pro Preview: https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/3-pro
- Image Understanding: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-understanding

Last updated: 2024-12 (verify prices at the reference URL before production use)

Key Pricing Rules:
1. Batch API provides 50% discount on BOTH input AND output tokens for Vertex AI
2. If input context exceeds 200K tokens, ALL tokens (input + output) are charged at long context rates
3. Images are tokenized and charged at the same rate as text tokens
4. Gemini 3 Pro uses variable sequence length for images (not Pan and Scan like older models)
"""

from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# EXCHANGE RATE
# =============================================================================
# USD to KRW exchange rate (approximate)
# Update this value as needed for accurate KRW estimates
USD_TO_KRW = 1450.0


# =============================================================================
# IMAGE TOKENIZATION
# =============================================================================
# Gemini 3 Pro uses "Variable Sequence Length" for images:
# - MEDIA_RESOLUTION_HIGH: 1120 tokens per image
# - MEDIA_RESOLUTION_MEDIUM: 560 tokens per image (default)
# - MEDIA_RESOLUTION_LOW: 280 tokens per image
#
# Earlier models (Gemini 2.x and below) use "Pan and Scan":
# - Fixed 258 tokens per image regardless of size
#
# Reference: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-understanding

GEMINI_3_IMAGE_TOKENS = {
    "high": 1120,  # MEDIA_RESOLUTION_HIGH
    "medium": 560,  # MEDIA_RESOLUTION_MEDIUM (default)
    "low": 280,  # MEDIA_RESOLUTION_LOW
}

GEMINI_2_IMAGE_TOKENS = 258  # Pan and Scan: fixed 258 tokens per image


def estimate_image_tokens(model: str, resolution: str = "medium") -> int:
    """
    Estimate tokens for an image based on model and resolution.

    Gemini 3 Pro uses variable sequence length:
    - MEDIA_RESOLUTION_HIGH: 1120 tokens
    - MEDIA_RESOLUTION_MEDIUM: 560 tokens (default)
    - MEDIA_RESOLUTION_LOW: 280 tokens

    Earlier models (Gemini 2.x) use Pan and Scan:
    - Fixed 258 tokens per image

    Args:
        model: Model name (e.g., "gemini-3-pro-preview", "gemini-2.5-flash")
        resolution: Image resolution level ("high", "medium", "low")

    Returns:
        Estimated token count for the image

    Reference:
        https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-understanding
    """
    # Gemini 3 Pro uses variable sequence length
    if model.startswith("gemini-3"):
        return GEMINI_3_IMAGE_TOKENS.get(resolution, GEMINI_3_IMAGE_TOKENS["medium"])

    # Earlier models (Gemini 2.x, etc.) use Pan and Scan: 258 tokens per image
    return GEMINI_2_IMAGE_TOKENS


# =============================================================================
# VERTEX AI PRICING TABLE (per 1M tokens)
# =============================================================================
# Reference: https://cloud.google.com/vertex-ai/generative-ai/pricing
#
# Important Notes:
# 1. Batch API provides 50% discount on BOTH input AND output for Vertex AI
# 2. If context exceeds 200K tokens, ALL tokens use "long context" rates
# 3. Only requests returning 200 response codes are charged
#
# Pricing Structure:
# - Flash models: flat rate regardless of context length
# - Pro models: tiered pricing based on context length (≤200K vs >200K)

VERTEX_AI_PRICING = {
    # =========================================================================
    # Gemini 3 Pro Preview
    # =========================================================================
    # Reference: https://cloud.google.com/vertex-ai/generative-ai/pricing
    #
    # Online API:
    #   Input (≤200K): $2.00/1M tokens
    #   Input (>200K): $4.00/1M tokens
    #   Output (≤200K): $12.00/1M tokens
    #   Output (>200K): $18.00/1M tokens
    #
    # Batch API (50% discount):
    #   Input (≤200K): $1.00/1M tokens
    #   Input (>200K): $2.00/1M tokens
    #   Output: 50% discount applied to online rates
    #
    "gemini-3-pro-preview": {
        # Online API rates
        "input_under_200k": 2.00,  # $/1M tokens for ≤200K context
        "input_over_200k": 4.00,  # $/1M tokens for >200K context
        "output_under_200k": 12.00,  # $/1M tokens for ≤200K context
        "output_over_200k": 18.00,  # $/1M tokens for >200K context
        # Batch API discounts (50% off for both input and output)
        "batch_input_discount": 0.5,  # Batch input: $1.00/1M (≤200K), $2.00/1M (>200K)
        "batch_output_discount": 0.5,  # Batch output: $6.00/1M (≤200K), $9.00/1M (>200K)
        # Image output rate (for image generation)
        "image_output": 120.00,  # $/1M tokens for image generation output
        # Model characteristics
        "has_tiered_pricing": True,  # Uses ≤200K / >200K pricing tiers
        "context_threshold": 200_000,  # Token threshold for tier switch
    },
    # =========================================================================
    # Gemini 2.5 Flash
    # =========================================================================
    # Reference: https://cloud.google.com/vertex-ai/generative-ai/pricing
    #
    # Online API:
    #   Input (text/image/video): $0.30/1M tokens (flat rate, no tier difference)
    #   Audio input: $1.00/1M tokens
    #   Output (text): $2.50/1M tokens
    #   Output (image): $30.00/1M tokens
    #
    # Batch API (50% discount):
    #   Input: $0.15/1M tokens
    #   Audio: $0.50/1M tokens
    #   Output: $1.25/1M tokens (50% discount applied)
    #
    "gemini-2.5-flash": {
        # Online API rates
        "input_text_image_video": 0.30,  # $/1M tokens for text/image/video input
        "input_audio": 1.00,  # $/1M tokens for audio input
        "output_text": 2.50,  # $/1M tokens for text output
        "output_image": 30.00,  # $/1M tokens for image generation output
        # Batch API discounts (50% off for both input and output)
        "batch_input_discount": 0.5,  # Batch input: $0.15/1M
        "batch_output_discount": 0.5,  # Batch output: $1.25/1M
        # Model characteristics
        "has_tiered_pricing": False,  # Flat rate, no context-based tiers
    },
}


# =============================================================================
# COST CALCULATION DATACLASS
# =============================================================================


@dataclass
class CostInfo:
    """
    Cost information for API requests.

    This class calculates costs based on token counts and model pricing.
    Supports both online and batch API pricing.

    Attributes:
        input_tokens: Number of input tokens consumed
        output_tokens: Number of output tokens generated
        model: Model name used for pricing lookup
        is_batch: Whether batch API pricing should be applied (50% discount)
        use_vertex_ai: Whether using Vertex AI (True) or Developer API (False)

    Example usage:
        cost = CostInfo(
            input_tokens=100000,
            output_tokens=5000,
            model="gemini-3-pro-preview",
            is_batch=True,
            use_vertex_ai=True,
        )
        cost.calculate_costs()
        print(f"Total: ${cost.total_cost_usd:.4f}")
    """

    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    is_batch: bool = True
    use_vertex_ai: bool = True

    # Calculated costs (populated by calculate_costs())
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    total_cost_krw: float = 0.0

    # Per-image breakdown (optional)
    per_image_costs: list[dict[str, Any]] = field(default_factory=list)

    def calculate_costs(self) -> None:
        """
        Calculate costs based on token counts and model pricing.

        Cost Calculation Logic:
        1. Look up model pricing from VERTEX_AI_PRICING table
        2. Determine input rate based on model type:
           - Tiered models (Pro): Use ≤200K or >200K rate based on context
           - Flat models (Flash): Use single rate regardless of context
        3. Determine output rate similarly
        4. Apply batch discount if is_batch=True (50% off both input and output)
        5. Calculate: cost = (tokens / 1,000,000) * rate

        Formula:
            Input Cost = (input_tokens / 1M) × input_rate × batch_discount
            Output Cost = (output_tokens / 1M) × output_rate × batch_discount
            Total Cost = Input Cost + Output Cost

        Example for Gemini 3 Pro Preview Batch (≤200K context):
            - Input: 100,000 tokens × ($2.00/1M × 0.5) = $0.10
            - Output: 5,000 tokens × ($12.00/1M × 0.5) = $0.03
            - Total: $0.13
        """
        # Get pricing for the model (fallback to gemini-2.5-flash if not found)
        pricing = VERTEX_AI_PRICING.get(self.model, VERTEX_AI_PRICING.get("gemini-2.5-flash", {}))

        # Determine input rate based on model pricing structure
        if pricing.get("has_tiered_pricing", False):
            # Tiered pricing (Pro models): choose rate based on context length
            # Note: For simplicity, we use ≤200K rate as default
            # In practice, you'd check if total context exceeds 200K
            input_rate = pricing.get("input_under_200k", 2.00)
            output_rate = pricing.get("output_under_200k", 12.00)
        else:
            # Flat pricing (Flash models): single rate
            input_rate = pricing.get("input_text_image_video", 0.30)
            output_rate = pricing.get("output_text", 2.50)

        # Apply batch discounts if using batch API
        # Vertex AI Batch: 50% discount on BOTH input AND output
        if self.is_batch:
            batch_input_discount = pricing.get("batch_input_discount", 0.5)
            batch_output_discount = pricing.get("batch_output_discount", 0.5)
            input_rate *= batch_input_discount
            output_rate *= batch_output_discount

        # Calculate costs (rate is per 1M tokens)
        # Formula: cost = (tokens / 1,000,000) * rate_per_million
        self.input_cost_usd = (self.input_tokens / 1_000_000) * input_rate
        self.output_cost_usd = (self.output_tokens / 1_000_000) * output_rate
        self.total_cost_usd = self.input_cost_usd + self.output_cost_usd
        self.total_cost_krw = self.total_cost_usd * USD_TO_KRW

    def to_dict(self) -> dict[str, Any]:
        """Convert cost info to dictionary for serialization."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "input_cost_usd": f"${self.input_cost_usd:.6f}",
            "output_cost_usd": f"${self.output_cost_usd:.6f}",
            "total_cost_usd": f"${self.total_cost_usd:.6f}",
            "total_cost_krw": f"₩{self.total_cost_krw:.2f}",
            "model": self.model,
            "is_batch_pricing": self.is_batch,
        }


# =============================================================================
# ESTIMATION DEFAULTS
# =============================================================================
# Default values for estimation when API token counting is unavailable

# Average output tokens (estimated for JSON response with contamination data)
# Typical response includes: image_type, areas array with contaminations
AVERAGE_OUTPUT_TOKENS = 500

# Default image tokens for estimation fallback
# Uses Gemini 3 Pro MEDIUM resolution as default
TOKENS_PER_IMAGE_ESTIMATE = GEMINI_3_IMAGE_TOKENS["medium"]  # 560 tokens


# =============================================================================
# COST ESTIMATION FUNCTIONS
# =============================================================================


def estimate_batch_cost(
    num_images: int,
    model: str,
    prompt_tokens: int,
    image_tokens_per_image: int | None = None,
    output_tokens_per_image: int = AVERAGE_OUTPUT_TOKENS,
    use_vertex_ai: bool = True,
) -> dict[str, Any]:
    """
    Estimate the cost of a batch job.

    This function calculates estimated costs without making API calls.
    For more accurate estimation, use the count_tokens API.

    Args:
        num_images: Number of images to process
        model: Model name (e.g., "gemini-3-pro-preview")
        prompt_tokens: Tokens in the text prompt
        image_tokens_per_image: Tokens per image (auto-detected if None)
        output_tokens_per_image: Expected output tokens per image
        use_vertex_ai: Whether using Vertex AI pricing

    Returns:
        Dictionary with estimated costs and token breakdown

    Example:
        estimate = estimate_batch_cost(
            num_images=100,
            model="gemini-3-pro-preview",
            prompt_tokens=2000,
            output_tokens_per_image=500,
        )
        print(f"Estimated cost: ${estimate['estimated_total_cost_usd']:.4f}")
    """
    # Auto-detect image tokens based on model
    if image_tokens_per_image is None:
        image_tokens_per_image = estimate_image_tokens(model, resolution="medium")

    # Calculate total tokens
    # Each request: prompt_tokens + image_tokens
    tokens_per_request = prompt_tokens + image_tokens_per_image
    total_input_tokens = tokens_per_request * num_images
    total_output_tokens = output_tokens_per_image * num_images

    # Calculate costs
    cost_info = CostInfo(
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        model=model,
        is_batch=True,
        use_vertex_ai=use_vertex_ai,
    )
    cost_info.calculate_costs()

    return {
        "num_images": num_images,
        "estimated_input_tokens": total_input_tokens,
        "estimated_output_tokens": total_output_tokens,
        "estimated_total_tokens": total_input_tokens + total_output_tokens,
        "tokens_per_request": tokens_per_request,
        "prompt_tokens": prompt_tokens,
        "image_tokens": image_tokens_per_image,
        "output_tokens_per_image": output_tokens_per_image,
        "estimated_input_cost_usd": cost_info.input_cost_usd,
        "estimated_output_cost_usd": cost_info.output_cost_usd,
        "estimated_total_cost_usd": cost_info.total_cost_usd,
        "estimated_total_cost_krw": cost_info.total_cost_krw,
        "model": model,
        "use_vertex_ai": use_vertex_ai,
        "batch_discount_applied": True,
        "estimation_method": "simple",
    }


def get_pricing_summary(model: str) -> dict[str, Any]:
    """
    Get a human-readable pricing summary for a model.

    Args:
        model: Model name

    Returns:
        Dictionary with pricing details for display
    """
    pricing = VERTEX_AI_PRICING.get(model, {})

    if pricing.get("has_tiered_pricing"):
        return {
            "model": model,
            "pricing_type": "tiered",
            "online_input_under_200k": f"${pricing.get('input_under_200k', 0):.2f}/1M",
            "online_input_over_200k": f"${pricing.get('input_over_200k', 0):.2f}/1M",
            "online_output_under_200k": f"${pricing.get('output_under_200k', 0):.2f}/1M",
            "online_output_over_200k": f"${pricing.get('output_over_200k', 0):.2f}/1M",
            "batch_input_under_200k": (
                f"${pricing.get('input_under_200k', 0) * pricing.get('batch_input_discount', 0.5):.2f}/1M"
            ),
            "batch_output_under_200k": (
                f"${pricing.get('output_under_200k', 0) * pricing.get('batch_output_discount', 0.5):.2f}/1M"
            ),
            "batch_discount": "50% on both input and output",
        }
    else:
        return {
            "model": model,
            "pricing_type": "flat",
            "online_input": f"${pricing.get('input_text_image_video', 0):.2f}/1M",
            "online_output": f"${pricing.get('output_text', 0):.2f}/1M",
            "batch_input": (
                f"${pricing.get('input_text_image_video', 0) * pricing.get('batch_input_discount', 0.5):.2f}/1M"
            ),
            "batch_output": (f"${pricing.get('output_text', 0) * pricing.get('batch_output_discount', 0.5):.2f}/1M"),
            "batch_discount": "50% on both input and output",
        }
