# Qwen Spam/Ham Classifier (binary)

A tiny repo that “decapitates” a causal LM (Qwen3-0.6B-Base) and fine-tunes a new 2-way classification head for **spam vs ham**.
It trains on three CSV files, writes versioned checkpoints, and includes a simple CLI for inference.

---

## Quick start

This project uses **uv** for dependency management and execution.

1) **Install deps**
   ```bash
   uv sync
   ```

2) **Train**
   ```bash
   uv run python train_qwen_classifier.py
   ```

3) **Classify some text (uses the `best` checkpoint symlink)**
   ```bash
   uv run python run_qwen_classifier.py "free $$$ click here NOW"
   # -> Ham: 03.12% || Spam: 96.88%
   ```

You can also point at a specific checkpoint directory name:
```bash
uv run python run_qwen_classifier.py "hello there, just checking in" 20251024Z223015
```

---

## What’s in here?

- `create_model.py` — Loads **Qwen/Qwen3-0.6B-Base**, then replaces `lm_head` with a fresh `torch.nn.Linear(..., out_features=2)` (binary classes). Uses `dtype="auto"` and `device_map="auto"` so the model tries to place itself on available GPU/CPU automatically.

- `train_qwen_classifier.py` — Minimal trainer:
  - Builds datasets from CSV files (see **Data format** below).
  - Tokenizes with the Qwen tokenizer and **pads/truncates to a fixed max length** (computed from the longest training sample unless you pass `max_length` to the dataset ctor).
  - Uses **cross-entropy** on the **last-token logits** for the two classes.
  - AdamW (lr `5e-5`, weight decay `0.1`), batch size `8`, `num_epochs=5`.
  - Prints loss every 50 steps (evaluating over 5 mini-batches) and accuracy per epoch (again over a small eval window).
  - Saves a timestamped checkpoint each eval; updates a `checkpoints/best` **symlink** when validation loss improves.

- `persistence.py` — Checkpoint I/O:
  - Saves **`safetensors`** state dict to `checkpoints/YYYYMMDDZHHMMSS/model.safetensors` + `meta.json`.
  - Maintains the `checkpoints/best` symlink to the best-so-far run.
  - `load_model(checkpoint)` restores weights into a freshly constructed model.

- `run_qwen_classifier.py` — Tiny CLI using `click`; prints **Ham** and **Spam** probabilities for an input string using a chosen checkpoint (default: `best`).

---

## Data format

Place three CSV files at the repo root:

- `classification-train.csv`
- `classification-validation.csv`
- `classification-test.csv`

Each must have the columns:

- **`Label`** — integer class id (`0` for ham, `1` for spam)
- **`Text`** — the raw input string

Example (`classification-train.csv`):

```csv
Label,Text
0,"Hello! Are we still on for lunch?"
1,"CONGRATULATIONS! You've won a FREE cruise. Click now!"
0,"Invoice attached. Thanks."
```

**Notes**
- By default, the training set’s **longest tokenized sample length** becomes the fixed sequence length. Validation and test are padded/truncated to that length for consistency.
- Padding uses the tokenizer’s pad token id (`50256` in the current code).

```

## Checkpoints & layout

After/while training you’ll have:

```text
checkpoints/
  20251024Z223015/
    model.safetensors
    meta.json        # {"epoch": ..., "train_loss": ..., "val_loss": ...}
  20251024Z231742/
    model.safetensors
    meta.json
  best -> 20251024Z231742   # symlink to the lowest val loss so far
```

Use the directory name (or `best`) with the CLI.

---

## Requirements

Managed by `uv`. Typical runtime stack:

- `torch`
- `transformers`
- `pandas`
- `safetensors`
- `click`
- (Optional but recommended) `accelerate` — improves `device_map="auto"` placement

Install via:

```bash
uv sync
```

All commands should be run with `uv run`, e.g.:

```bash
uv run python train_qwen_classifier.py
uv run python run_qwen_classifier.py "some text"
```

---

## How it works (short version)

1. **Model** — Start from `Qwen/Qwen3-0.6B-Base` (causal LM). Replace `lm_head` with a 2-logit linear layer (ham/spam).
2. **Labeling scheme** — For each input, forward once and read **the last token’s logits**. Apply softmax to get class probabilities.
3. **Training loop** — Simple supervised fine-tuning on cross-entropy of those logits versus integer labels. Mini evals every 50 steps; best-val symlink updated on improvement.

---

## Tips & troubleshooting

- **GPU vs CPU**: `device_map="auto"` will try GPU if available. If you only have CPU and see CUDA errors, ensure your PyTorch install matches your platform or set CUDA-related env vars off.
- **Model downloads**: Hugging Face will download `Qwen/Qwen3-0.6B-Base` on first run; ensure you have internet or have it cached.
- **Sequence length**: Very long inputs will be truncated to the training max length; very short inputs are padded. If desired, set a hard `max_length` in `SpamDataset`.
- **Reproducibility**: The script seeds a few things (`torch.manual_seed(123)`), but complete determinism is not guaranteed across hardware/backends.

---

## License

See the LICENSE file.

