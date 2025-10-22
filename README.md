# Car Contamination Classification

Python client for car contamination classification using Vision Language Models (VLM).

## Installation

```bash
# Install dependencies using uv
uv sync
```

## Commands

### 1. Single Image Inference (`infer`)

Run inference on a single image to classify contamination.

```bash
python main.py infer <image_path> <prompt_path> [OPTIONS]
```

**Example:**
```bash
# Basic usage
python main.py infer \
  images/sample_images/images/car1.jpg \
  prompts/car_contamination_classification_prompt_v5.txt

# Save output to file
python main.py infer \
  images/sample_images/images/car1.jpg \
  prompts/car_contamination_classification_prompt_v5.txt \
  --output results/car1_result.json

# Use different model
python main.py infer \
  images/sample_images/images/car1.jpg \
  prompts/car_contamination_classification_prompt_v5.txt \
  --model qwen3-vl-4b-instruct \
  --temperature 0.0
```

**Options:**
- `--api-url`: VLM API endpoint URL (default: http://vllm.mlops.socarcorp.co.kr/v1/chat/completions)
- `--model`: Model name (default: qwen3-vl-4b-instruct)
- `--max-tokens`: Maximum tokens to generate (default: 1000)
- `--temperature`: Sampling temperature (default: 0.0)
- `--output`: Output file path (optional)

---

### 2. Batch Inference (`batch-infer`)

Run inference on multiple images and save results to CSV.

```bash
python main.py batch-infer <images_directory> <prompt_path> <output_csv> [OPTIONS]
```

**Example:**
```bash
# Process all images in a directory
python main.py batch-infer \
  images/sample_images/images \
  prompts/car_contamination_classification_prompt_v5.txt \
  results/inference_results.csv

# Process only first 10 images
python main.py batch-infer \
  images/sample_images/images \
  prompts/car_contamination_classification_prompt_v5.txt \
  results/inference_results.csv \
  --limit 10

# With custom parameters
python main.py batch-infer \
  images/sample_images/images \
  prompts/car_contamination_classification_prompt_v5.txt \
  results/inference_results.csv \
  --model qwen3-vl-4b-instruct \
  --max-tokens 1000 \
  --temperature 0.0
```

**Output CSV Format:**
The CSV includes:
- `image_name`: Image filename
- `model`: Model used for inference
- `latency_seconds`: Processing time
- `success`: Whether inference succeeded
- `image_type`: interior/exterior/ood
- `{area}_contamination_type`: Contamination type for each area
- `{area}_severity`: Severity level for each area

**Options:**
- `--api-url`: VLM API endpoint URL
- `--model`: Model name (default: qwen3-vl-4b-instruct)
- `--max-tokens`: Maximum tokens to generate (default: 1000)
- `--temperature`: Sampling temperature (default: 0.0)
- `--limit`: Maximum number of images to process (default: all)

---

### 3. Generate Prompt (`generate-prompt`)

Generate a prompt template from a guideline CSV file using Jinja2 templates.

```bash
python main.py generate-prompt <guideline_csv> [output_path] [OPTIONS]
```

**Example:**
```bash
# Auto-generate with version numbering
python main.py generate-prompt guideline/guideline_v1.csv

# Save to specific location
python main.py generate-prompt \
  guideline/guideline_v1.csv \
  prompts/custom_prompt.txt

# Save transformed guideline
python main.py generate-prompt \
  guideline/guideline_v1.csv \
  --save-transformed

# Use specific template version
python main.py generate-prompt \
  guideline/guideline_v1.csv \
  --template-version 1
```

**Options:**
- `--save-transformed`: Save transformed guideline CSV
- `--template-version`: Template version to use (default: 1)

**Guideline CSV Format:**
```csv
오염항목,내/외부 구분,양호 (Good),보통 (Normal),심각 (Critical)
오염/때 (부스러기),내부,Description for good,Description for normal,Description for critical
```

The command transforms this into a structured prompt using Jinja2 templates located in `prompts/templates/`.

---

## Help

For detailed help on any command:

```bash
# General help
python main.py --help

# Command-specific help
python main.py infer --help
python main.py batch-infer --help
python main.py generate-prompt --help
```

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- WebP (.webp)
- BMP (.bmp)

Local images are automatically converted to base64 data URLs before sending to the API.
