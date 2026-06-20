# Input-Trio

**Phase-1 Input Layer** for the [Nexus Transformer](https://github.com/nexuss0781) — a unified C++17 pipeline that transforms raw token IDs into position-aware, trainable embeddings.

```
Token IDs → [Tokenizer] → [HFAQE Embed] → [HDPE Position-Encode] → Training
```

Three components, one header-only orchestrator, zero dependencies beyond a C++17 compiler.

---

## Architecture

```
Input-Trio/
├── Component-1.1_Tokenizer/           # Tokenizer bridge (Python stub / extendable)
├── Component-1.2_Token-Embedding/     # HFAQE — Hierarchical Frequency-Adaptive
│                                      #   Quantized Embedding (submodule)
├── Component-1.3_Positional-Encoding/ # HDPE — Hierarchical Digit Positional
│                                      #   Encoding with RoPE (submodule)
├── master-input/                      # Unified orchestrator (C++17 header-only)
│   ├── src/
│   │   ├── input_engine.cpp           # InputEngine — tokenize → embed → rotate
│   │   ├── training.cpp               # InputTrainingPipeline — full training loop
│   │   ├── train_input_layer.cpp      # Standalone trainer CLI entry point
│   │   └── main.cpp                   # Demo / smoke test
│   ├── tests/
│   │   ├── test.cpp                   # 16 integration tests
│   │   └── test_train_numerical.cpp   # 28 numerical correctness tests
│   └── CMakeLists.txt                 # 4 targets, CTest integration
├── .gitmodules
└── README.md
```

### 1. Tokenizer (`Component-1.1_Tokenizer`)

Bridge to your tokenizer of choice. Ships with a `pybind11` stub that falls back gracefully when Python bindings are unavailable. Produces `std::vector<int>` token IDs for the embedding stage.

### 2. HFAQE Embedding (`Component-1.2_Token-Embedding`)

Hierarchical Frequency-Adaptive Quantized Embedding — a production-grade embedding layer with:

- **Two-tier architecture**: frequently-used tokens stored in a hot `int8` cache; rare tokens reconstructed via low-rank cold SVD (`Basis · α`)
- **STE gradients** through quantisation for end-to-end training
- **AdamW optimizer** with tier-specific LR multipliers (hot ×1.0, cold ×2.0)
- **Automatic tier reallocation** based on token frequency shifts
- **Auxiliary losses**: semantic alignment, quantisation error, orthogonality regularisation

### 3. HDPE Position Encoding (`Component-1.3_Positional-Encoding`)

Hierarchical Digit Positional Encoding — Rotary Position Embedding (RoPE) with:

- Multi-resolution encoding via hierarchical digit decomposition
- Per-head rotation: each head's `d_k` dimensions rotated independently
- **Exact inverse rotation** (`R^T`) for correct gradient backpropagation through position encoding
- O(1) cos/sin lookup per position

### 4. Orchestrator (`master-input`)

The orchestrator stitches the three components together into `InputEngine` (inference) and `InputTrainingPipeline` (training). All code is **single-translation-unit** — include `training.cpp` and everything compiles in one TU.

---

## Quick Start

```bash
# Clone with submodules
git clone --recursive https://github.com/nexuss0781/Input-Trio.git
cd Input-Trio/master-input

# Build (Release recommended — HFAQE cold SVD is 10-20s even in Release)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run demo
./build/master_main

# Run all tests
./build/master_test
./build/test_train_numerical

# Run via CTest
ctest --test-dir build -V
```

**Build times**: Release ~5-10 minutes (HFAQE + Storage.cpp are template-heavy). Debug may exceed 15 minutes.

---

## Training

### Standalone Trainer

```bash
./build/train_input_layer --steps 500 --V 4096 --d 64 --r 16 --K 256 \
  --batch 4 --seq 32 --lr 1e-3
```

| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | 200 | Total training steps |
| `--V` | 4096 | Vocabulary size |
| `--d` | 64 | Embedding dimension |
| `--r` | 16 | Cold-tier SVD rank |
| `--K` | 256 | Hot-tier cache size |
| `--batch` | 4 | Sequences per step |
| `--seq` | 32 | Tokens per sequence |
| `--lr` | 1e-3 | Learning rate |
| `--ckpt` | checkpoints | Checkpoint directory |
| `--resume` | — | Resume from latest checkpoint |

### Training Loop

Each `train_step` executes:

```
zero_grad               →  reset STE gradient buffer
forward(token_ids)      →  HFAQE embed: raw embeddings
apply_rotation          →  HDPE RoPE (per-head d_k rotation)
compute_loss            →  LM head projection → CE loss → dL/dX_rot
apply_inverse_rotation  →  R^T · dL/dX_rot (gradient through position encoding)
backward(dL_dX)         →  HFAQE STE backward → accumulate dW
apply_gradients(lr)     →  AdamW + aux losses + gradient compression
```

Position encoding has **no trainable parameters**, but gradients must still pass through it correctly. The pipeline uses `apply_inverse_rope_inplace` (R^T rotation) to achieve exact gradient backpropagation through RoPE, verified by §4 of the numerical test suite.

### Checkpoint Format

Checkpoints use the **Nex** binary format (from `Storage.cpp`):
- Versioned sections with CRC32C checksums
- Delta-compressed hot-tier weights
- Full AdamW optimizer state (momentum + velocity)
- Token frequency histogram for tier reallocation after resume

```bash
ls checkpoints/
# input_trio_step_0000100.nex  input_trio_step_0000200.nex  input_trio_final.nex
```

Resume with `--resume` — loads the latest checkpoint and continues from the saved step count.

---

## Test Suites

### Integration Tests (`master_test` — 16 tests)

Cover the full pipeline life cycle:

| § | Test | What it verifies |
|---|------|------------------|
| 1 | Initialisation | Engine init returns true, config preserved |
| 2 | Forward Pass | Token → embedding output shape and finite values |
| 3 | Position Encoding | Norm preservation (RoPE is orthogonal) |
| 4 | Raw Embedding | Embed without position encoding |
| 5 | Norm Preservation | ‖embed‖₂ ≈ ‖rotated‖₂ within 1e-4 |
| 6 | LM Head | Logits finite, correct shape, tied-weight symmetry |
| 7 | Training Pipeline | Init, gradient flow, training steps produce valid loss |
| 8 | Validation Loss | Held-out CE / perplexity computation |

### Numerical Correctness Tests (`test_train_numerical` — 28 tests)

Targeted numerical validation:

| § | Test | What it verifies |
|---|------|------------------|
| 1 | Gradient Flow | Finite loss + non-zero grad norm after one step |
| 2 | Loss Convergence | CE loss decreases over 20 steps |
| 3 | Checkpoint Round-Trip | Save → load → step count preserved → loss identical |
| 4 | Position Encoding Gradient | RoPE forward changes values, inv(RoPE(x)) == x |
| 5 | HDPE Integration | Position 0 identity, different positions differ |

---

## Key Design Decisions

**Single-TU architecture**: All components are included as `.cpp` files (not headers) with include guards. This avoids:
- ODR violations from template instantiations
- Linker complexity with 3+ separate libraries
- CMake dependency chains across submodule boundaries

**Gradient through RoPE**: Since position encoding has no learnable parameters, the backward pass applies `R^T` rotation to `dL/dX_rot`. This is mathematically exact — verified by the §4.3 test (`inv(RoPE(forward(x))) == x` within 1e-5).

**HFAQE training API**: The model exposes `forward`, `backward`, `zero_grad`, `apply_gradients`, and `lm_head` directly — no Python wrappers or DSL. The training pipeline calls these directly in a straightforward loop.

**Checkpoint fidelity**: Full AdamW state (momentum `m`, velocity `v`, step counter) is serialised and restored. Without this, the optimiser would lose momentum state after resume, causing a step mismatch in repeated runs. Verified by §3.4 (loss matches within 1e-4 after round-trip).

---

## Performance Notes

| Operation | d=64 | d=256 | d=512 |
|-----------|------|-------|-------|
| Init (cold SVD) | ~2s | ~18s | ~80s |
| Forward (1K tokens) | ~15ms | ~60ms | ~200ms |
| Train step (128 tok) | ~250ms | ~1s | ~3s |

- **Cold SVD** scales O(V·d·r) during init. Use smaller `r` and `d` for faster iteration.
- **Hot-tier** gradient accumulation is O(K·d) per step — K=1024, d=256 → 262K params, negligible.
- **AdamW** state is O(V·d) × 2 fp32 vectors — 16K·256 → 32 MB for full vocab.
- Compile time is dominated by `Storage.cpp` (NexWriter/NexReader templates, ~1400 lines).

---

## Submodules

```bash
git submodule update --init --recursive
```

- [Component-1.2_Token-Embedding](https://github.com/nexuss0781/Nexuss_Embedding) — HFAQE quantised embedding
- [Component-1.3_Positional-Encoding](https://github.com/nexuss0781/Positional-Encoding) — HDPE rotary position encoding

---

## Licence

MIT — see individual submodules for their licence terms.
