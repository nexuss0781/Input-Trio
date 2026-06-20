# Training — Input-Trio (HFAQE + HDPE)

## Prerequisites

```bash
# 1. Clone with submodules
git clone --recurse-submodules https://github.com/nexuss0781/Input-Trio.git
cd Input-Trio

# OR if already cloned without submodules:
git submodule update --init --recursive

# 2. Install Python dataset dependency
pip install datasets huggingface_hub

# 3. Build
cd master-input
mkdir -p build && cd build && cmake .. && make train_input_layer -j$(nproc)
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
| `--V` | 4096 | Vocabulary size (synthetic only) |
| `--data_dir` | `Data` | Path to folder with train/validation/test.txt |
| `--ckpt` | `checkpoints` | Checkpoint directory |
| `--resume` | — | Resume from latest checkpoint |

## Manual data download

```bash
python3 dataset.py --data_dir /path/to/Data
./build/train_input_layer --data_dir /path/to/Data
```

## Training expected behaviour

```
loss=5.5 → loss=3.5 → loss=2.5 → ... → loss ~1.0-1.5
```

Perplexity should drop from ~200+ to ~10-20 within 1000 steps on WikiText-2 byte-level.
