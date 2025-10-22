# Car Contamination Classification

Python client for car contamination classification using Vision Language Models (VLM).

## Installation

```bash
# Install dependencies using uv
uv sync
```

## Commands

### 1. Dashboard (`dashboard`)

Launch an interactive web dashboard to visualize inference results.

```bash
streamlit run src/dashboard/app.py
```

The dashboard provides:
- 📊 Summary statistics (total images, success rate, interior/exterior counts)
- 🔍 Filtering by image type, GT area, and inference success
- 🖼️ Image-by-image navigation with side-by-side comparison
- 📋 Ground truth vs prediction visualization
- 🎨 Color-coded severity levels (🔴 심각, 🟡 보통, 🟢 양호)
- 💾 Download filtered results as CSV

**Configuration:**
- Default CSV path: `results/inference_results.csv`
- Default images directory: `images/sample_images/images`
- Both paths can be customized in the sidebar

**Usage Tips:**
- Use the slider to navigate through images
- Apply filters to focus on specific subsets
- Hover over areas to see detailed contamination info
- Download filtered results for further analysis

---

### 2. Single Image Inference (`infer`)

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
  --model qwen3-vl-8b-instruct-fp8 \
  --temperature 0.0
```

**Options:**
- `--api-url`: VLM API endpoint URL (default: http://vllm.mlops.socarcorp.co.kr/v1/chat/completions)
- `--model`: Model name (default: qwen25-vl-7b-instruct-awq)
- `--max-tokens`: Maximum tokens to generate (default: 1000)
- `--temperature`: Sampling temperature (default: 0.0)
- `--output`: Output file path (optional)

---

### 3. Batch Inference (`batch-infer`)

Run inference on multiple images specified in a CSV file and save results to CSV.

```bash
python main.py batch-infer <input_csv> <images_directory> <prompt_path> <output_csv> [OPTIONS]
```

**Input CSV Format:**
The input CSV must contain the following columns:
- `file_name`: Image filename (e.g., "car1.jpg")
- `gt_contamination_area`: Ground truth area (interior/exterior)
- `gt_contamination_type`: Ground truth contamination type

**Example:**
```bash
# Process images specified in CSV with ground truth data
python main.py batch-infer \
  images/sample_images/csv/merged_data.csv \
  images/sample_images/images \
  prompts/car_contamination_classification_prompt_v5.txt \
  results/inference_results.csv

# Process only first 10 images
python main.py batch-infer \
  images/sample_images/csv/merged_data.csv \
  images/sample_images/images \
  prompts/car_contamination_classification_prompt_v5.txt \
  results/inference_results.csv \
  --limit 10

# With custom model (default is qwen25-vl-7b-instruct-awq)
python main.py batch-infer \
  images/sample_images/csv/merged_data.csv \
  images/sample_images/images \
  prompts/car_contamination_classification_prompt_v5.txt \
  results/inference_results.csv \
  --model qwen3-vl-8b-instruct-fp8 \
  --max-tokens 1000 \
  --temperature 0.0
```

**Output CSV Format:**
The CSV includes:
- `image_name`: Image filename
- `gt_contamination_area`: Ground truth area (interior/exterior) from input CSV
- `gt_contamination_type`: Ground truth contamination type from input CSV
- `model`: Model used for inference
- `latency_seconds`: Processing time
- `success`: Whether inference succeeded
- `image_type`: interior/exterior/ood
- `{area}_contamination_type`: Contamination type for each area
- `{area}_severity`: Severity level for each area

**Options:**
- `--api-url`: VLM API endpoint URL
- `--model`: Model name (default: qwen25-vl-7b-instruct-awq)
- `--max-tokens`: Maximum tokens to generate (default: 1000)
- `--temperature`: Sampling temperature (default: 0.0)
- `--limit`: Maximum number of images to process (default: all)

---

### 4. Generate Prompt (`generate-prompt`)

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

# Use template version 3 (Korean prompt with English keys)
python main.py generate-prompt \
  guideline/guideline_v1.csv \
  --template-version 3
```

**Template Versions:**
- **v1**: English instructions with Korean examples
- **v2**: English instructions with strict English output codes
- **v3**: Korean instructions with Korean values (auto-converted to English in CSV output)

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
