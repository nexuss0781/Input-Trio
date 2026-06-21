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
make train_input_layer -j$(nproc)  # single target
```

## Train

```bash
./train_input_layer --steps 2000 --d 256 --batch 16 --seq 128 --V 256 --lr 3e-4
```

First run auto-downloads `Salesforce/wikitext-2-raw-v1` → `Data/{train,validation,test}.txt`.  
After training: auto-evaluates val + test perplexity (~2.5 min on GPU).

| Metric | Random | Trained |
|--------|--------|---------|
| Loss | 5.55 (ln 256) | ~4.64 |
| Val ppl | 256 | ~108 |
| Test ppl | 256 | ~108 |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | 200 | Steps |
| `--batch` | 4 | Batch size |
| `--seq` | 32 | Sequence length |
| `--lr` | 1e-3 | Learning rate |
| `--V` | 4096 | Vocab size |
| `--d` | 64 | Embedding dim |
| `--data_dir` | `Data` | Data folder |
| `--ckpt` | `checkpoints` | Checkpoint dir |
| `--resume` | — | Resume from latest |
| `--push_to_hub` | — | HF repo ID |
| `--token` | — | HF token string |
| `--token_file` | `HF_TOKEN` | Token file path |

### Push to Hugging Face Hub

Add `--push_to_hub` to upload checkpoints after training:

```bash
HF_TOKEN="hf_xxxxxxxxxx" ./train_input_layer --push_to_hub nexuss0781/Input-Trio --steps 2000 --d 256
```

Or via file (auto-created with placeholder on first push):

```bash
echo 'TOKEN="hf_xxxxxxxxxx"' > HF_TOKEN
./train_input_layer --push_to_hub nexuss0781/Input-Trio --steps 2000 --d 256
```

Or via `--token` flag:

```bash
./train_input_layer --push_to_hub nexuss0781/Input-Trio --token hf_xxxxxxxxxx --steps 2000 --d 256
```

Priority: `--token` > `$HF_TOKEN` > `HF_TOKEN` file. The `HF_TOKEN` file is gitignored.

## Inference

```bash
./validation --sentence "The future of AI begins here."
./validation --file ../src/sentences.txt
```

Output: byte tokens → HFAQE embeddings → HDPE position-encode → next-token probs → final distribution.
