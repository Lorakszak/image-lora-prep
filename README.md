# image-lora-prep

Utilities to prepare image datasets for LoRA training. Includes:

- Image scaling to accepted resolutions/aspect ratios with optional min/max constraints
- Automated center-cropping to target shapes
- Manual cropping GUI with draggable preset rectangles

## Installation (conda)

```bash
conda env create -f environment.yml
conda activate image-lora-prep
```

Alternatively (pip/venv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Preset aspect ratios and sizes

Defined in `config.py` via `ACCEPTED_SHAPES`:

- 1:1 → 1024x1024
- 3:4 → 896x1152
- 5:8 → 832x1216
- 9:16 → 768x1344
- 9:21 → 640x1536
- Reverse orientations: 4:3, 8:5, 16:9, 21:9

## CLI

```
python main.py --help
python main.py scale --help
python main.py crop-auto --help
python main.py crop-manual --help
python main.py caption --help
python main.py rename --help
```

Examples:

```bash
# Scale with center-crop to 3:4 (896x1152)
python main.py scale --input-path ./images --output-dir ./out --shape 3:4 \
  --max-width 2048 --max-height 2048 --allow-upscale --resample lanczos

# Automated crop to 16:9 landscape
python main.py crop-auto --input-path ./images --output-dir ./out16x9 --shape 16:9

# Manual cropping GUI, starting with 1:1 selection
python main.py crop-manual --input-dir ./images --output-dir ./out_manual --shape 1:1

# Caption all images in a directory using OpenAI; captions saved as .txt files
export OPENAI_API_KEY=sk-...  # set your key
python main.py caption --input-path ./images --output-dir ./captions \
  --prompt "Describe succinctly for training." --model gpt-5 --max-tokens 256

# Rename images sequentially with optional prefix
python main.py rename --directory ./images --prefix sample_
```

## Configuration

See `config.py` for global settings for paths, behavior, and shapes. CLI options override relevant settings at runtime.

### Captioning prompts

Prompts are configured in `config.py` under `Captioning`:
- The system prompt and user prompt are tuned to produce a detailed, natural-language, single-line description without labeled sections like `Subject:` or `Style:` and without newlines or tabs.

