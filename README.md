# Car Wash VLM Client

Python client for querying Vision Language Models (VLM) via API. This tool allows you to send images (from URLs or local files) along with text prompts to analyze visual content.

## Features

- Support for both image URLs and local image files
- Load prompts from text files or pass them directly
- Base64 encoding for local images
- Configurable model parameters (max tokens, temperature)
- Save responses to file or print to stdout

## Installation

```bash
# Install dependencies
pip install -e .
```

## Usage

### Basic Usage

```bash
# Using image URL and direct prompt
python main.py "https://example.com/image.jpg" "What do you see in this image?"

# Using local image and prompt file
python main.py "./images/car.jpg" "./prompts/example_korean.txt"
```

### With Options

```bash
# Specify model and parameters
python main.py \
  "./images/car.jpg" \
  "./prompts/car_wash_inspection.txt" \
  --model "qwen3-vl-4b-instruct" \
  --max-tokens 500 \
  --temperature 0.7

# Save response to file
python main.py \
  "https://example.com/image.jpg" \
  "Describe this image" \
  --output response.txt

# Use different API endpoint
python main.py \
  "./image.jpg" \
  "What do you see?" \
  --api-url "http://localhost:8000/v1/chat/completions"
```

### Command Line Arguments

```
positional arguments:
  image                 Image URL or local path to image file
  prompt                Text prompt or path to .txt file containing prompt

optional arguments:
  --api-url            VLM API endpoint URL (default: http://vllm.mlops.socarcorp.co.kr/v1/chat/completions)
  --model              Model name to use (default: qwen3-vl-4b-instruct)
  --max-tokens         Maximum tokens to generate (default: 200)
  --temperature        Sampling temperature (default: 0.7)
  --output             Output file path (optional, prints to stdout if not specified)
```

## Example Prompts

The `prompts/` directory contains example prompt files:

- `example_korean.txt` - Simple Korean prompt
- `example_english.txt` - Simple English prompt
- `car_wash_inspection.txt` - Detailed car inspection prompt

## Using as a Library

```python
from main import VLMClient

# Initialize client
client = VLMClient(
    api_url="http://vllm.mlops.socarcorp.co.kr/v1/chat/completions",
    model="qwen3-vl-4b-instruct"
)

# Query with image URL
response = client.query(
    image_input="https://example.com/image.jpg",
    prompt_input="What do you see in this image?",
    max_tokens=200
)

# Query with local image and prompt file
response = client.query(
    image_input="./images/car.jpg",
    prompt_input="./prompts/car_wash_inspection.txt",
    max_tokens=500,
    temperature=0.7
)

# Extract response text
result = response['choices'][0]['message']['content']
print(result)
```

## Image Format Support

Supported image formats:
- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- WebP (.webp)
- BMP (.bmp)

Local images are automatically converted to base64 data URLs before sending to the API.

## Error Handling

The client handles common errors:
- File not found (for local images or prompt files)
- Network errors (connection timeout, HTTP errors)
- API errors (invalid response format)

All errors are reported with descriptive messages to help with debugging.

## Original cURL Command

This Python client replicates the following cURL command:

```bash
curl -L -X POST http://vllm.mlops.socarcorp.co.kr/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-vl-4b-instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "https://example.com/image.jpg"
            }
          },
          {
            "type": "text",
            "text": "What do you see in this image?"
          }
        ]
      }
    ],
    "max_tokens": 200
  }'
```
