# Training — Input-Trio (HFAQE + HDPE)

## 1. Setup

```bash
git clone --recurse-submodules https://github.com/nexuss0781/Input-Trio.git
cd Input-Trio
pip install datasets huggingface_hub
```

Update existing clone:
```bash
git pull && git submodule update --init --recursive
```

## 2. Build

```bash
cd master-input && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```

Targets: `train_input_layer`, `validation`, `master_main`, `master_test`.

Build one target: `make train_input_layer -j$(nproc)`

## 3. Train

```bash
./train_input_layer --steps 2000 --d 256 --batch 16 --seq 128 --V 256 --lr 3e-4
```

On first run: auto-downloads `Salesforce/wikitext-2-raw-v1` → `Data/{train,validation,test}.txt`.  
After training: auto-evaluates val + test perplexity (~2.5 min, ~78 ms/step on GPU).

### Expected results

| Metric | Random | Trained | Gain |
|--------|--------|---------|------|
| Train loss | 5.55 (ln 256) | ~4.64 | -0.91 |
| Val loss | 5.55 | ~4.69 | -0.86 |
| Val ppl | 256 | ~108 | 2.4× |
| Test ppl | 256 | ~108 | 2.4× |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | 200 | Steps |
| `--batch` | 4 | Batch size |
| `--seq` | 32 | Sequence length |
| `--lr` | 1e-3 | Learning rate |
| `--V` | 4096 | Vocab size (`--V 256` for byte-level) |
| `--d` | 64 | Embedding dim (256+ recommended) |
| `--data_dir` | `Data` | Path to train/val/test.txt |
| `--ckpt` | `checkpoints` | Checkpoint dir |
| `--resume` | — | Resume from latest checkpoint |
| `--push_to_hub` | — | Repo ID to push (e.g. `nexuss0781/Input-Trio`) |

## 4. Push to Hugging Face Hub

```bash
./train_input_layer --push_to_hub nexuss0781/Input-Trio --steps 2000 --d 256 --batch 16 --seq 128 --V 256 --lr 3e-4
```

First run auto-creates `HF_TOKEN` with `TOKEN=""`. Edit it:
```
TOKEN="hf_your_token_here"
```
Or set env var `HF_TOKEN`. File is gitignored.

## 5. Inference

```bash
./validation --sentence "The future of AI begins here."
./validation --file ../src/sentences.txt
```

Output stages: byte tokens → HFAQE embeddings → HDPE position-encode → next-token predictions → final distribution.
