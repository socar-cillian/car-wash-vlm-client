"""Simple script to run car contamination inference."""

from pathlib import Path

from src.api import VLMClient


def main():
    # Configuration
    prompt_path = Path("prompts/prompt_v4.1.txt")
    image_path = input("🖼️  Image path (local file or URL): ").strip()

    if not image_path:
        print("❌ No image path provided")
        return

    # Load prompt
    if not prompt_path.exists():
        print(f"❌ Prompt file not found: {prompt_path}")
        return

    prompt = prompt_path.read_text(encoding="utf-8")

    # Initialize client
    api_url = "https://vllm.mlops.socarcorp.co.kr/v1/chat/completions"
    model = "qwen3-vl-8b-instruct"

    print(f"\n🔧 API: {api_url}")
    print(f"🤖 Model: {model}")
    print(f"📝 Prompt: {prompt_path}")
    print(f"🖼️  Image: {image_path}")
    print()

    client = VLMClient(api_url=api_url, model=model)

    # Health check
    print("⏳ Checking server health...")
    health = client.check_health(timeout=10)
    if not health["healthy"]:
        print(f"❌ Server unhealthy: {health.get('error')}")
        return
    print(f"✅ Server healthy ({health['response_time']:.2f}s)")
    print()

    # Run inference
    print("⏳ Running inference...")
    try:
        response = client.query(
            image_input=image_path,
            prompt_input=prompt,
            max_tokens=2000,
            temperature=0.0,
        )

        if "choices" in response and response["choices"]:
            result = response["choices"][0]["message"]["content"]
            print("\n" + "=" * 60)
            print("📊 Result:")
            print("=" * 60)
            print(result)
        else:
            print("❌ No response from model")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
