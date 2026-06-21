# Training — Input-Trio (HFAQE + HDPE)

## Setup

```bash
git clone --recurse-submodules https://github.com/nexuss0781/Input-Trio.git
cd Input-Trio
pip install datasets huggingface_hub
```

Update existing: `git pull && git submodule update --init --recursive`

## Build

```bash
cd master-input && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
make train_input_layer -j$(nproc)
```

## Train

```bash
./train_input_layer --steps 2000 --d 256 --batch 16 --seq 128 --V 256 --lr 3e-4
```

Auto-downloads WikiText-2 on first run. After training: val + test perplexity eval.

| Metric | Before | After |
|--------|--------|-------|
| Loss | 5.55 | ~4.64 |
| Val ppl | 256 | ~108 |
| Test ppl | 256 | ~108 |

### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | 200 | Steps |
| `--batch` | 4 | Batch size |
| `--seq` | 32 | Sequence length |
| `--lr` | 1e-3 | Learning rate |
| `--V` | 4096 | Vocab size |
| `--d` | 64 | Embedding dim |
| `--data_dir` | Data | Data folder |
| `--ckpt` | checkpoints | Checkpoint dir |
| `--resume` | — | Resume latest checkpoint |
| `--push_to_hub` | — | HF repo ID to push to |
| `--token` | — | HF token (inline) |
| `--token_file` | HF_TOKEN | HF token file path |

## Push to Hub

Three ways to provide token (priority: `--token` > `$HF_TOKEN` > file):

```bash
# 1 — env var
HF_TOKEN="hf_xxx" ./train_input_layer --push_to_hub nexuss0781/Input-Trio --steps 2000 --d 256

# 2 — token file (auto-created with placeholder on first push)
echo 'TOKEN="hf_xxx"' > HF_TOKEN
./train_input_layer --push_to_hub nexuss0781/Input-Trio --steps 2000 --d 256

# 3 — inline flag
./train_input_layer --push_to_hub nexuss0781/Input-Trio --token hf_xxx --steps 2000 --d 256
```

`HF_TOKEN` file is gitignored.

## Inference

```bash
./validation --sentence "The future of AI begins here."
./validation --file ../src/sentences.txt
```

Output pipeline: byte tokens → HFAQE embeddings → HDPE positional encoding → next-token predictions → final distribution.
