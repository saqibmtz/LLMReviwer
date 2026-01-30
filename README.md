## LLM Paper Evaluator

Minimal pipeline to extract text from PDFs, evaluate with an LLM (OpenAI or D-LLM) using a prompt template, and save results as JSON.

## Project structure

```
LLMReviewer/
├── Code/
│   ├── run.py                # Batch runner (discovers PDFs, builds prompts, calls model, saves JSON)
│   ├── evaluate_pdfs.py      # PDF extraction + model call helpers
│   └── barebones.py          # Minimal standalone example
├── prompts/
│   └── evaluation_prompt.txt # Prompt template
├── rawdata/                  # Input PDFs
├── processed/
│   └── evaluation/           # Output JSON
├── .env                      # Environment configuration
├── requirements.txt          # Dependencies
└── README.md
```

## Quick start

### 1) Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure your LLM provider

Edit `.env` or export environment variables:

**For OpenAI (default):**

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY="your-api-key-here"
export MODEL=gpt-5
```

**For D-LLM:**

```bash
export LLM_PROVIDER=dllm
export OPENAI_API_KEY="your-api-key-here"
export DLLM_BASE_URL="http://your-dllm-gateway:8080/v1"
export MODEL="Qwen/Qwen3-32B"
```

### 3) Run the batch evaluator

```bash
cd Code
python run.py
```

Outputs are written to `processed/evaluation/<PDF_STEM>_<PROMPT_STEM>.json`.

## Configuration (env vars)

### Provider settings

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider to use (`openai` or `dllm`) | `openai` |
| `OPENAI_API_KEY` | API key for authentication | (required) |
| `DLLM_BASE_URL` | D-LLM gateway URL (required when using `dllm`) | - |
| `MODEL` | Model name | `gpt-5` |

### Path settings

| Variable | Description | Default |
|----------|-------------|---------|
| `PROMPT_PATH` | Prompt template path | `../prompts/evaluation_prompt.txt` |
| `INPUT_DIR` | Where to read PDFs | `../rawdata` |
| `OUTPUT_DIR` | Where to write JSON | `../processed/evaluation` |

### Rate limiting

| Variable | Description | Default |
|----------|-------------|---------|
| `TPM_LIMIT` | Tokens-per-minute pacing limit | `30000` |
| `MAX_RETRIES` | Retry attempts on failures | `5` |
| `INITIAL_BACKOFF_S` | Initial backoff seconds | `5` |
| `MAX_CONTENT_CHARS` | Truncate long papers to this many chars | `300000` |

## Example usage

**OpenAI with custom model:**

```bash
LLM_PROVIDER=openai MODEL=gpt-4o-mini python Code/run.py
```

**D-LLM:**

```bash
LLM_PROVIDER=dllm DLLM_BASE_URL=http://localhost:8080/v1 MODEL=Qwen/Qwen3-32B python Code/run.py
```

## Notes

- **Extraction**: Uses PyMuPDF (`fitz`) to read PDFs and heuristics to find title, abstract, and main text.
- **Pacing**: Simple TPM pacing and exponential backoff are applied to limit rate errors.
- **Truncation**: Long content is truncated to `MAX_CONTENT_CHARS` and recorded in `truncated.txt`.
- **Idempotency**: Existing outputs are skipped to allow resumable runs.
- **D-LLM**: Uses OpenAI-compatible API format. See [D-LLM docs](https://llm-d.ai/docs/usage/getting-started-inferencing) for setup.

## Troubleshooting

- **Missing OPENAI_API_KEY**: Set the environment variable as shown above.
- **Missing DLLM_BASE_URL**: Required when `LLM_PROVIDER=dllm`. Set to your D-LLM gateway endpoint.
- **ModuleNotFoundError: fitz**: Run `pip install -r requirements.txt` inside the virtualenv.
- **Unknown provider**: Ensure `LLM_PROVIDER` is either `openai` or `dllm`.
