# Training — Input-Trio (HFAQE + HDPE)

## Prerequisites

```bash
# 1. Clone with submodules
git clone --recurse-submodules https://github.com/nexuss0781/Input-Trio.git
cd Input-Trio

# OR update existing clone:
git pull
git submodule update --init --recursive

# 2. Install Python dataset dependency
pip install datasets huggingface_hub

# 3. Build all targets (CMake auto-inits submodules at configure time)
cd master-input
mkdir -p build && cd build && cmake .. && make -j$(nproc)
```

## Build

```bash
cd master-input
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Builds all targets: `train_input_layer`, `validation`, `master_main`, `master_test`.

To build only one target:
```bash
make train_input_layer -j$(nproc)   # training only
make validation -j$(nproc)          # inference only
```

Or build a single target:
```bash
make train_input_layer -j$(nproc)
make validation -j$(nproc)
```

## Train on WikiText-2 (auto-download)

```bash
./train_input_layer --steps 2000 --d 256 --batch 16 --seq 128 --V 256 --lr 3e-4
```

On first run this downloads `Salesforce/wikitext-2-raw-v1` into `Data/` and trains.
After training, it automatically evaluates on validation and test splits.

### Expected results

| Metric | Random | Trained | Gain |
|--------|--------|---------|------|
| Train loss | 5.55 (ln 256) | ~4.64 | -0.91 |
| Val loss | 5.55 | ~4.69 | -0.86 |
| Val perplexity | 256 | ~108 | 2.4× better |
| Test perplexity | 256 | ~108 | 2.4× better |

Training takes ~2.5 min (2000 steps, ~78 ms/step on GPU).

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | 200 | Training steps |
| `--batch` | 4 | Batch size |
| `--seq` | 32 | Sequence length |
| `--lr` | 1e-3 | Learning rate |
| `--V` | 4096 | Vocabulary size (set `--V 256` for byte-level WikiText) |
| `--d` | 64 | Embedding dimension (256+ recommended) |
| `--data_dir` | `Data` | Path to folder with train/validation/test.txt |
| `--ckpt` | `checkpoints` | Checkpoint directory |
| `--resume` | — | Resume from latest checkpoint |

## Inference on sentences

```bash
# Single sentence
./validation --sentence "The future of AI begins here."

# Batch from file
./validation --file ../src/sentences.txt
```

Output shows all stages:
1. **Byte token IDs** — ASCII byte values
2. **Raw embeddings** — HFAQE output vectors
3. **Position-encoded** — after HDPE RoPE rotation
4. **Next-token predictions** — per-position top-5 probabilities
5. **Final next-token** — top-8 distribution with entropy
