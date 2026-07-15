# Magibu_HW1

A two-part homework repository: **(1)** training a BPE tokenizer from scratch on a Turkish novel, and **(2)** a small-scale, character-level implementation of the Qwen3.5 architecture (hybrid attention / Gated DeltaNet) trained on Turkish village names.

```
Magibu_HW1/
├── 1-BPE_Tokenizer_Suc_ve_Ceza/   # HW1 — BPE tokenizer training
└── 2-TinyModel/                   # HW2 — Qwen3.5-hybrid language model
```

---

## 1 — BPE Tokenizer: Suç ve Ceza (Crime and Punishment)

A Byte-Pair Encoding tokenizer is trained from scratch, using the `tokenizers` (HuggingFace) library, on the Turkish translation of Dostoevsky's _Crime and Punishment_.

|                    |                                                                                    |
| ------------------ | ---------------------------------------------------------------------------------- |
| **Input text**     | `suc_ve_ceza.txt` (163,253 words, ~1.2M characters)                                |
| **Pre-tokenizer**  | `Whitespace`                                                                       |
| **Vocab size**     | 15,000                                                                             |
| **Special tokens** | `[UNK]`, `[PAD]`, `[BOS]`, `[EOS]`                                                 |
| **Output**         | `tokenizer.json` (trained tokenizer, containing ~15,000 merge rules and the vocab) |

### Running it

```bash
cd 1-BPE_Tokenizer_Suc_ve_Ceza
pip install tokenizers
python3 tokenizer.py
```

The script reads the text, runs BPE training, sanity-checks the result by encoding/decoding a sample sentence (`"merhaba dünya"`), and saves the output as `tokenizer.json`.

### Reusing the saved tokenizer

```python
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer.json")
output = tok.encode("Raskolnikov sokakta yürüyordu.")
print(output.tokens)      # subword pieces
print(output.ids)         # token ids
print(tok.decode(output.ids))
```

### Notes

- `suc_ve_ceza.pdf` is the original PDF source; `suc_ve_ceza.txt` is the plain-text extraction of it.
- A vocab size of 15,000 is a reasonable balance for a ~163K-word corpus: large enough to learn root + suffix patterns, but not so large that it results in excessive, overly sparse subword fragmentation.

---

## 2 — TinyModel: Qwen3.5-hybrid mini language model

A small, teaching-oriented implementation of the Qwen3.5 architecture with no dependency beyond PyTorch. The model learns to generate **Turkish village names** character by character — designed to show end-to-end how a modern hybrid attention architecture (softmax attention + linear attention) works, at a scale small enough to read top to bottom.

### Architecture

The model in `qwen3_5/` builds on a standard Qwen3 skeleton (pre-norm, RMSNorm, RoPE, SwiGLU, GQA), but replaces softmax attention with **Gated DeltaNet** (linear attention) in most layers:

```
S_t = alpha_t * S_{t-1} + beta_t * (v_t - alpha_t * S_{t-1} k_t) k_t^T
```

- `alpha_t` — **decay gate**: how much of the old memory to keep
- `beta_t` — **write gate**: how strongly to write the correction
- `(v_t - prediction)` — **delta term**: only the part that was predicted wrong gets written to memory

With `config.full_attn_every = 4`, every 4th layer uses full (softmax) attention, and the rest use Gated DeltaNet.

| File                | Role                                                         |
| ------------------- | ------------------------------------------------------------ |
| `config.py`         | Model hyperparameters (`ModelConfig`)                        |
| `rms_norm.py`       | RMSNorm                                                      |
| `rotary.py`         | RoPE (cos/sin precomputation)                                |
| `gated_deltanet.py` | Linear-attention / gated-delta-rule layer                    |
| `attention.py`      | Full (softmax, GQA) attention layer                          |
| `mlp.py`            | SwiGLU MLP                                                   |
| `block.py`          | A single transformer block (attention type chosen per layer) |
| `model.py`          | `TinyQwen35` — embed → blocks → norm → tied lm_head          |
| `tokenizer.py`      | `CharTokenizer` — character-level tokenizer                  |
| `train.py`          | Training loop                                                |
| `generate.py`       | Name generation from a trained checkpoint                    |

### Data

`data/koy_isimleri.txt` — a list of Turkish village names (11,141 lines). The model learns from this list character by character, using `\n` as the start/end-of-name (EOS) marker.

### Running it

```bash
cd 2-TinyModel/qwen3_5
python3 train.py              # trains, prints loss, samples names, saves a checkpoint
python3 generate.py 20        # generates 20 new village names
python3 generate.py 20 0.7    # lower temperature = safer / more consistent names
```

At the end of training, the model weights are saved as `tiny_qwen35.pt`; `generate.py` loads this checkpoint and samples new names from it.

### Notes

- `demo.ipynb` walks through running the model step by step and inspecting the results.
- The goal is to demonstrate the **signature ideas** of Qwen3.5 (the gated delta rule, the hybrid layer split) in their simplest correct form, not to match its actual scale; the delta-rule update here is a plain Python loop over time (real Qwen3.5 parallelizes it in chunks).

---

## Requirements

```bash
pip install tokenizers torch
```

## Author

[@eraycancelik](https://github.com/eraycancelik)
