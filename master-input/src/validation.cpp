// =============================================================================
// validation — Load trained model and run inference on a sentence
//   Shows each stage: token IDs → raw embeddings → position-encoded → logits
//   Usage: ./build/validation [--sentence "your text here"] [--ckpt checkpoints]
// =============================================================================
#include "training.cpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

// ---------------------------------------------------------------------------
// Softmax helper
// ---------------------------------------------------------------------------
static void softmax(fp32* logits, int V) {
    fp32 mx = -1e30f;
    for (int t = 0; t < V; ++t)
        if (logits[t] > mx) mx = logits[t];
    fp32 sum = 0.0f;
    for (int t = 0; t < V; ++t) {
        logits[t] = std::exp(logits[t] - mx);
        sum += logits[t];
    }
    fp32 inv = 1.0f / (sum + 1e-10f);
    for (int t = 0; t < V; ++t) logits[t] *= inv;
}

// ---------------------------------------------------------------------------
// Print a short vector summary (first few dims + ...)
// ---------------------------------------------------------------------------
static void print_vec(const fp32* v, int d, int show = 6, const char* label = "") {
    std::printf("  %s [%d]:", label, d);
    int n = std::min(show, d);
    for (int i = 0; i < n; ++i) std::printf(" %+.4f", v[i]);
    if (n < d) std::printf(" ...");
    std::printf("\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string sentence  = "The future of AI begins here.";
    std::string ckpt_dir  = "checkpoints";
    std::string ckpt_name = "input_trio";

    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--sentence")) sentence = argv[++i];
        else if (!std::strcmp(argv[i], "--ckpt"))     ckpt_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--name"))     ckpt_name = argv[++i];
    }

    std::printf("============================================================\n");
    std::printf("  Inference — Input Layer Validation\n");
    std::printf("  Sentence: \"%s\"\n", sentence.c_str());
    std::printf("============================================================\n\n");

    // ---- 1. Byte tokenise ----
    std::printf("--- Step 1: Byte Tokenisation ---\n");
    auto ids = byte_tokenise(sentence);
    int n = (int)ids.size();
    std::printf("  %d bytes:\n", n);
    std::printf("  IDs:  ");
    for (int i = 0; i < n; ++i) std::printf("%3d ", ids[i]);
    std::printf("\n");
    std::printf("  Chars:");
    for (int i = 0; i < n; ++i) std::printf("  %c ", (char)ids[i]);
    std::printf("\n\n");

    // ---- 2. Init engine and load checkpoint ----
    std::printf("--- Step 2: Load Model ---\n");
    InputEngineConfig engine_cfg;
    // Values from the final training run (must match checkpoint)
    engine_cfg.V = 256;
    engine_cfg.d = 256;
    engine_cfg.r = 16;
    engine_cfg.K = 255;   // V-1 per clamping
    engine_cfg.B = 64;
    engine_cfg.L = 4;
    engine_cfg.h = 4;
    engine_cfg.base = 10000.0f;

    InputTrainingConfig train_cfg;
    train_cfg.batch_size  = 1;
    train_cfg.seq_len     = n;
    train_cfg.ckpt_dir    = ckpt_dir;
    train_cfg.ckpt_name   = ckpt_name;

    InputTrainingPipeline trainer;
    if (!trainer.init(engine_cfg, train_cfg)) {
        std::fprintf(stderr, "FATAL: pipeline init failed\n");
        return 1;
    }
    if (!trainer.load_checkpoint()) {
        std::fprintf(stderr, "FATAL: no checkpoint found in %s/\n", ckpt_dir.c_str());
        return 1;
    }
    std::printf("  Model loaded from %s/ (step %d)\n\n", ckpt_dir.c_str(), trainer.step());

    // ---- 3. Embed ----
    std::printf("--- Step 3: Raw Embedding (HFAQE forward) ---\n");
    auto t0 = Clock::now();
    int d = engine_cfg.d;
    int d_k = engine_cfg.d_k();
    int h_ = engine_cfg.h;
    std::vector<fp32> X(static_cast<size_t>(n) * d);

    HFAQE* model = trainer.model();
    model->forward(ids.data(), n, X.data());

    for (int i = 0; i < n && i < 3; ++i) {
        print_vec(X.data() + i * d, d, 6,
                  (std::string("token[") + std::to_string(i) + "] raw").c_str());
    }
    auto t1 = Clock::now();
    double embed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("  Embed time: %.2f ms\n\n", embed_ms);

    // ---- 4. Position-encode (RoPE) ----
    std::printf("--- Step 4: Position Encoding (HDPE RoPE) ---\n");
    t0 = Clock::now();

    const HDPE& hdpe = trainer.engine().hdpe();
    int pairs = engine_cfg.pairs();
    std::vector<fp32> cos_buf(static_cast<size_t>(n) * pairs);
    std::vector<fp32> sin_buf(static_cast<size_t>(n) * pairs);

    // Precompute cos/sin for all positions
    for (int i = 0; i < n; ++i) {
        hdpe.encode_position(i, cos_buf.data() + i * pairs,
                                sin_buf.data() + i * pairs);
    }

    // Apply RoPE in-place (modifies X)
    for (int i = 0; i < n; ++i) {
        fp32* row = X.data() + i * d;
        for (int head = 0; head < h_; ++head) {
            apply_rope_inplace(row + head * d_k, d_k,
                               cos_buf.data() + i * pairs,
                               sin_buf.data() + i * pairs);
        }
    }

    for (int i = 0; i < n && i < 3; ++i) {
        print_vec(X.data() + i * d, d, 6,
                  (std::string("token[") + std::to_string(i) + "] rot").c_str());
    }
    t1 = Clock::now();
    double rope_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("  RoPE time: %.2f ms\n\n", rope_ms);

    // ---- 5. LM head — next-token prediction at each position ----
    std::printf("--- Step 5: Next-Token Predictions ---\n");
    t0 = Clock::now();

    int V = model->cfg.V;
    std::vector<fp32> logits(static_cast<size_t>(V));
    int top_k = 5;

    for (int i = 0; i < n - 1; ++i) {
        const fp32* hi = X.data() + static_cast<ptrdiff_t>(i) * d;
        model->lm_head(hi, logits.data());
        softmax(logits.data(), V);

        int target = ids[i + 1];
        fp32 p_target = std::max(logits[target], 1e-10f);

        // Find top-K
        int top_ids[5];
        fp32 top_vals[5];
        for (int k = 0; k < top_k; ++k) top_vals[k] = -1e30f;
        for (int t = 0; t < V; ++t) {
            if (logits[t] <= top_vals[top_k - 1]) continue;
            int pos = top_k - 1;
            while (pos > 0 && logits[t] > top_vals[pos - 1]) --pos;
            for (int r = top_k - 1; r > pos; --r) {
                top_ids[r] = top_ids[r - 1];
                top_vals[r] = top_vals[r - 1];
            }
            top_ids[pos] = t;
            top_vals[pos] = logits[t];
        }

        std::printf("  pos %2d: "
                    "true='%c'(%3d) p=%.2f  |  top:",
                    i, (char)target, target, p_target);
        for (int k = 0; k < top_k; ++k)
            std::printf(" '%c'(%d)=%.2f",
                    (char)top_ids[k], top_ids[k], top_vals[k]);
        std::printf("\n");
    }

    // ---- 6. Final next-token prediction (after last byte) ----
    std::printf("\n--- Step 6: Next Token After Entire Input ---\n");
    {
        const fp32* h_last = X.data() + static_cast<ptrdiff_t>(n - 1) * d;
        model->lm_head(h_last, logits.data());
        softmax(logits.data(), V);

        int top_ids[10];
        fp32 top_vals[10];
        int show_top = 10;
        for (int k = 0; k < show_top; ++k) top_vals[k] = -1e30f;
        for (int t = 0; t < V; ++t) {
            if (logits[t] <= top_vals[show_top - 1]) continue;
            int pos = show_top - 1;
            while (pos > 0 && logits[t] > top_vals[pos - 1]) --pos;
            for (int r = show_top - 1; r > pos; --r) {
                top_ids[r] = top_ids[r - 1];
                top_vals[r] = top_vals[r - 1];
            }
            top_ids[pos] = t;
            top_vals[pos] = logits[t];
        }

        std::printf("  Next-token distribution (top %d):\n", show_top);
        for (int k = 0; k < show_top; ++k) {
            char c = (top_ids[k] >= 32 && top_ids[k] <= 126)
                     ? (char)top_ids[k] : '?';
            std::printf("    %2d. '%c' (byte %3d)  p=%.4f\n",
                        k + 1, c, top_ids[k], top_vals[k]);
        }

        // Entropy of distribution
        fp32 H = 0.0f;
        for (int t = 0; t < V; ++t) {
            if (logits[t] > 1e-10f)
                H -= logits[t] * std::log(logits[t]);
        }
        std::printf("  Distribution entropy: %.3f nats  (max=%.3f)\n",
                    H, std::log((fp32)V));
    }

    t1 = Clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::printf("\n============================================================\n");
    std::printf("  Total inference time: %.2f ms\n", embed_ms + rope_ms + total_ms);
    std::printf("============================================================\n");

    return 0;
}
