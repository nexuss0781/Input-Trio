# Training — Input-Trio (HFAQE + HDPE)

## Prerequisites

```bash
# 1. Clone with submodules
git clone --recurse-submodules https://github.com/nexuss0781/Input-Trio.git
cd Input-Trio

# OR update existing clone without losing changes:
git pull
git submodule update --init --recursive

# 2. Install Python dataset dependency
pip install datasets huggingface_hub

# 3. Build all targets
cd master-input
mkdir -p build && cd build && cmake .. && make -j$(nproc)
```

> CMake auto-runs `git submodule update --init --recursive` at configure time,
> so a plain `cmake ..` also fetches submodules automatically.

## Quick start (auto-download WikiText-2)

```bash
./train_input_layer --steps 1000
```

On first run this downloads `Salesforce/wikitext-2-raw-v1` into `Data/` and trains.

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

## Manual data download

```bash
python3 dataset.py --data_dir /path/to/Data
./build/train_input_layer --data_dir /path/to/Data
```

## Training results (first run)

Default config (`d=64`, `V=4096`, `batch=4`, `seq=32`):
```
train loss 8.3 → 4.5   |   val loss stuck at ~6.0  |   val ppl ~420
```
Loss drops but validation plateaus — model is too small for byte-level patterns.

**Recommended** for WikiText-2 byte-level:
```bash
./train_input_layer --steps 2000 --d 256 --batch 16 --seq 128 --V 256 --lr 3e-4
```

Expected: val ppl < 50 within 1000 steps.

## Run inference on a sentence

After training, run the trained model on any text:

```bash
./build/validation --sentence "The future of AI begins here."
```

Shows each stage:
1. **Byte token IDs** — ASCII byte values
2. **Raw embeddings** — HFAQE output vectors (first dims)
3. **Position-encoded** — after HDPE RoPE rotation
4. **Next-token predictions** — per-position top-5 probabilities
5. **Final prediction** — top-10 next byte after the full sentence

## Training expected behaviour

```
loss=5.5 → loss=3.5 → loss=2.5 → ... → loss ~1.0-1.5
```

Perplexity should drop from ~200+ to ~10-20 within 1000 steps on WikiText-2 byte-level.
