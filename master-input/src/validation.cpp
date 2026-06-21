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
    std::string file_path;

    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--sentence")) sentence = argv[++i];
        else if (!std::strcmp(argv[i], "--file"))     file_path = argv[++i];
        else if (!std::strcmp(argv[i], "--ckpt"))     ckpt_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--name"))     ckpt_name = argv[++i];
    }

    // Load sentences from file if --file is given
    std::vector<std::string> sentences;
    if (!file_path.empty()) {
        sentences = load_text_file(file_path);
        if (sentences.empty()) {
            std::fprintf(stderr, "FATAL: no sentences in %s\n", file_path.c_str());
            return 1;
        }
    } else {
        sentences.push_back(sentence);
    }

    std::printf("============================================================\n");
    std::printf("  Inference — Input Layer Validation\n");
    std::printf("  Sentences: %zu\n", sentences.size());
    if (file_path.empty())
        std::printf("  \"%s\"\n", sentence.c_str());
    else
        std::printf("  File: %s\n", file_path.c_str());
    std::printf("============================================================\n\n");

    // ---- 2. Init engine and load checkpoint (once) ----
    std::printf("--- Step 2: Load Model ---\n");
    InputEngineConfig engine_cfg;
    engine_cfg.V    = 256;
    engine_cfg.d    = 256;
    engine_cfg.r    = 16;
    engine_cfg.K    = 255;
    engine_cfg.B    = 64;
    engine_cfg.L    = 4;
    engine_cfg.h    = 4;
    engine_cfg.base = 10000.0f;

    InputTrainingConfig train_cfg;
    train_cfg.batch_size  = 1;
    train_cfg.ckpt_dir    = ckpt_dir;
    train_cfg.ckpt_name   = ckpt_name;

    InputTrainingPipeline trainer;
    if (!trainer.init(engine_cfg, train_cfg)) { std::fprintf(stderr, "FATAL: init\n"); return 1; }
    if (!trainer.load_checkpoint()) { std::fprintf(stderr, "FATAL: no ckpt\n"); return 1; }
    std::printf("  Model loaded from %s/ (step %d)\n\n", ckpt_dir.c_str(), trainer.step());

    int d       = engine_cfg.d;
    int d_k     = engine_cfg.d_k();
    int h_      = engine_cfg.h;
    int pairs   = engine_cfg.pairs();
    int V       = trainer.model()->cfg.V;
    HFAQE* model = trainer.model();
    HDPE& hdpe   = trainer.engine().hdpe();
    std::vector<fp32> logits(static_cast<size_t>(V));

    // ---- Process each sentence ----
    for (size_t si = 0; si < sentences.size(); ++si) {
        const std::string& text = sentences[si];

        std::printf("\n============================================================\n");
        std::printf("  Sentence %zu/%zu: \"%s\"\n", si + 1, sentences.size(), text.c_str());
        std::printf("============================================================\n");

        // Byte tokenise
        auto ids = byte_tokenise(text);
        int n = (int)ids.size();
        std::printf("\n--- Token IDs ---\n  ");
        for (int i = 0; i < n; ++i) std::printf("%3d ", ids[i]);
        std::printf("\n  ");
        for (int i = 0; i < n; ++i) std::printf("  %c ", (char)ids[i]);
        std::printf("\n");

        // Embed
        std::vector<fp32> X(static_cast<size_t>(n) * d);
        auto t0 = Clock::now();
        model->forward(ids.data(), n, X.data());
        auto t1 = Clock::now();

        std::printf("\n--- Raw Embeddings (first 6 dims) ---\n");
        for (int i = 0; i < std::min(n, 3); ++i)
            print_vec(X.data() + i * d, d, 6, ("token[" + std::to_string(i) + "]").c_str());

        // Position-encode (RoPE)
        t0 = Clock::now();
        std::vector<fp32> cos_buf(static_cast<size_t>(n) * pairs);
        std::vector<fp32> sin_buf(static_cast<size_t>(n) * pairs);
        for (int i = 0; i < n; ++i)
            hdpe.encode_position(i, cos_buf.data() + i * pairs, sin_buf.data() + i * pairs);
        for (int i = 0; i < n; ++i) {
            fp32* row = X.data() + i * d;
            for (int head = 0; head < h_; ++head)
                apply_rope_inplace(row + head * d_k, d_k,
                                   cos_buf.data() + i * pairs,
                                   sin_buf.data() + i * pairs);
        }
        t1 = Clock::now();

        std::printf("\n--- Position-Encoded (first 6 dims) ---\n");
        for (int i = 0; i < std::min(n, 3); ++i)
            print_vec(X.data() + i * d, d, 6, ("token[" + std::to_string(i) + "]").c_str());

        // Per-position next-token predictions
        std::printf("\n--- Next-Token Predictions ---\n");
        int top_k = 5;
        fp32 total_loss = 0.0f;
        for (int i = 0; i < n - 1; ++i) {
            const fp32* hi = X.data() + static_cast<ptrdiff_t>(i) * d;
            model->lm_head(hi, logits.data());
            softmax(logits.data(), V);

            int target = ids[i + 1];
            fp32 p_target = std::max(logits[target], 1e-10f);
            total_loss -= std::log(p_target);

            int top_ids[5];
            fp32 top_vals[5];
            for (int k = 0; k < top_k; ++k) top_vals[k] = -1e30f;
            for (int t = 0; t < V; ++t) {
                if (logits[t] <= top_vals[top_k - 1]) continue;
                int pos = top_k - 1;
                while (pos > 0 && logits[t] > top_vals[pos - 1]) --pos;
                for (int r = top_k - 1; r > pos; --r) { top_ids[r] = top_ids[r-1]; top_vals[r] = top_vals[r-1]; }
                top_ids[pos] = t; top_vals[pos] = logits[t];
            }

            std::printf("  pos %2d: true='%c'(%3d) p=%.2f  |",
                        i, (char)target, target, p_target);
            for (int k = 0; k < top_k; ++k)
                std::printf(" '%c'(%d)=%.2f", (char)top_ids[k], top_ids[k], top_vals[k]);
            std::printf("\n");
        }

        // Final next-token prediction
        std::printf("\n--- Next Token After Input ---\n");
        {
            const fp32* h_last = X.data() + static_cast<ptrdiff_t>(n - 1) * d;
            model->lm_head(h_last, logits.data());
            softmax(logits.data(), V);

            int show_top = 8;
            int top_ids[8];
            fp32 top_vals[8];
            for (int k = 0; k < show_top; ++k) top_vals[k] = -1e30f;
            for (int t = 0; t < V; ++t) {
                if (logits[t] <= top_vals[show_top - 1]) continue;
                int pos = show_top - 1;
                while (pos > 0 && logits[t] > top_vals[pos - 1]) --pos;
                for (int r = show_top - 1; r > pos; --r) { top_ids[r] = top_ids[r-1]; top_vals[r] = top_vals[r-1]; }
                top_ids[pos] = t; top_vals[pos] = logits[t];
            }
            for (int k = 0; k < show_top; ++k) {
                char c = (top_ids[k] >= 32 && top_ids[k] <= 126) ? (char)top_ids[k] : '?';
                std::printf("  %2d. '%c' (byte %3d)  p=%.4f\n", k+1, c, top_ids[k], top_vals[k]);
            }

            fp32 H = 0;
            for (int t = 0; t < V; ++t)
                if (logits[t] > 1e-10f) H -= logits[t] * std::log(logits[t]);
            std::printf("  entropy=%.3f / max=%.3f  |  avg_loss=%.4f\n",
                        H, std::log((fp32)V), total_loss / std::max(1, n-1));
        }
    }

    return 0;
}
