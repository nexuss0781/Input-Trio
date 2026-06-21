// =============================================================================
// InputEngine — Nexus Unified Input Layer
//   Tokenize → Embed (HFAQE) → Position-Encode (HDPE)
//   Single end-to-end entry point for the complete input pipeline.
// =============================================================================
#ifndef INPUT_ENGINE_CPP
#define INPUT_ENGINE_CPP

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <random>
#include <chrono>

using Clock = std::chrono::high_resolution_clock;

// ---------------------------------------------------------------------------
// 1. Embedding — HFAQE (hierarchical frequency-adaptive quantized embedding)
//    Transitively includes tokenizer bridge (Input.cpp → Core.cpp).
// ---------------------------------------------------------------------------
#include "../../Component-1.2_Token-Embedding/src/Output.cpp"

// ---------------------------------------------------------------------------
// 2. Positional Encoding — HDPE (hierarchical digit positional encoding)
//    Types already defined by HFAQE — skip redefinition.
// ---------------------------------------------------------------------------
#define NEXUS_TYPES
#include "../../Component-1.3_Positional-Encoding/src/core.cpp"

// Inverse RoPE rotation for backward pass: R^T · grad
// Forward:  q_rot[2i]   = q[2i]*c - q[2i+1]*s
//           q_rot[2i+1] = q[2i]*s + q[2i+1]*c
// Inverse:  grad[2i]   = grad[2i]*c + grad[2i+1]*s
//           grad[2i+1] = grad[2i+1]*c - grad[2i]*s
inline void apply_inverse_rope_inplace(fp32* grad, int d_k,
                                        const fp32* cos_arr, const fp32* sin_arr) {
    int pairs = d_k / 2;
    for (int i = 0; i < pairs; ++i) {
        fp32 g0 = grad[2 * i];
        fp32 g1 = grad[2 * i + 1];
        fp32 c  = cos_arr[i];
        fp32 s  = sin_arr[i];
        grad[2 * i]     = g0 * c + g1 * s;
        grad[2 * i + 1] = g1 * c - g0 * s;
    }
}

// =============================================================================
// §1 — Configuration
// =============================================================================

struct InputEngineConfig {
    // Tokenizer
    std::string tokenizer_path = "";

    // Embedding (HFAQE)
    int V  = 16000;   // vocabulary size
    int d  = 512;     // model dimension
    int r  = 64;      // cold-tier rank
    int K  = 1024;    // number of hot tokens

    // Positional encoding (HDPE)
    int B    = 64;    // hierarchical base
    int L    = 4;     // number of levels
    int h    = 8;     // heads (d_k = d / h)
    fp32 base = 10000.0f;

    // Derived
    int d_k()   const { return d / h; }
    int pairs() const { return d_k() / 2; }
    int64_t range() const {
        int64_t r = 1;
        for (int i = 0; i < L; ++i) r *= B;
        return r;
    }
};

// =============================================================================
// §2 — Input Engine (tokenize → embed → position-encode)
// =============================================================================

class InputEngine {
public:
    InputEngine() = default;
    ~InputEngine() = default;

    // ---- Init ----
    bool init(const InputEngineConfig& cfg) {
        cfg_ = cfg;

        // 2a. Init token embedder
        //     Falls back to stub mode if pybind11 unavailable (tokenizer_path = "")
        //     Clamp K to V-1 so cold tier has at least 1 slot (required by HFAQE).
        int K_eff = std::min(cfg.K, std::max(1, cfg.V - 1));
        embedder_ = HFAQEOutput::from_config(cfg.V, cfg.d, cfg.r, K_eff,
                                              /*B=*/64, /*seed=*/42, cfg.tokenizer_path);

        // 2b. Init HDPE tables
        hdpe_cfg_.B    = cfg.B;
        hdpe_cfg_.L    = cfg.L;
        hdpe_cfg_.d    = cfg.d;
        hdpe_cfg_.pairs = cfg.pairs();
        hdpe_cfg_.base = cfg.base;
        hdpe_.init(hdpe_cfg_);

        return true;
    }

    // ---- Full pipeline: text → position-aware embeddings ----
    // Returns number of tokens, or 0 on failure.
    int forward_text(const std::string& text, fp32* out, int max_tokens = 0) {
        auto result = embedder_->embed_text(text);
        if (result.n == 0) return 0;

        int n = result.n;
        if (max_tokens > 0 && n > max_tokens) n = max_tokens;

        apply_position_encoding(result.data.data(), n, out);
        return n;
    }

    // ---- Token IDs → position-aware embeddings ----
    void forward(const int* token_ids, int n, fp32* out) {
        embedder_->embed_tokens(token_ids, n, out);
        apply_position_encoding(out, n, out);
    }

    // ---- Token IDs + explicit positions ----
    void forward(const int* token_ids, const int* positions, int n, fp32* out) {
        embedder_->embed_tokens(token_ids, n, out);
        apply_position_encoding_at(out, positions, n, out);
    }

    // ---- Sub-operations ----
    std::vector<int> tokenize(const std::string& text) {
        auto result = embedder_->embed_text(text);
        return result.token_ids;
    }

    void embed(const int* token_ids, int n, fp32* out) {
        embedder_->embed_tokens(token_ids, n, out);
    }

    // ---- Position encoding: forward rotation ----
    // Apply RoPE to each head's d_k dimensions using precomputed cos/sin for
    // natural positions 0..n-1.
    void apply_rotation(fp32* data, int n) {
        int pairs_ = cfg_.pairs();
        int d_k_   = cfg_.d_k();
        int h_     = cfg_.h;
        int d      = cfg_.d;
        std::vector<fp32> cos_buf(pairs_), sin_buf(pairs_);
        for (int i = 0; i < n; ++i) {
            hdpe_.encode_position(i, cos_buf.data(), sin_buf.data());
            fp32* row = data + i * d;
            for (int head = 0; head < h_; ++head)
                apply_rope_inplace(row + head * d_k_, d_k_, cos_buf.data(), sin_buf.data());
        }
    }

    // ---- Position encoding: inverse rotation (for backward pass) ----
    // Applies R^T to each head's d_k dimensions, where R is the RoPE rotation.
    void apply_inverse_rotation(fp32* grad, int n) {
        int pairs_ = cfg_.pairs();
        int d_k_   = cfg_.d_k();
        int h_     = cfg_.h;
        int d      = cfg_.d;
        std::vector<fp32> cos_buf(pairs_), sin_buf(pairs_);
        for (int i = 0; i < n; ++i) {
            hdpe_.encode_position(i, cos_buf.data(), sin_buf.data());
            fp32* row = grad + i * d;
            for (int head = 0; head < h_; ++head)
                apply_inverse_rope_inplace(row + head * d_k_, d_k_, cos_buf.data(), sin_buf.data());
        }
    }

    // ---- Position encoding: inverse rotation at explicit positions ----
    void apply_inverse_rotation_at(fp32* grad, const int* positions, int n) {
        int pairs_ = cfg_.pairs();
        int d_k_   = cfg_.d_k();
        int h_     = cfg_.h;
        int d      = cfg_.d;
        std::vector<fp32> cos_buf(pairs_), sin_buf(pairs_);
        for (int i = 0; i < n; ++i) {
            hdpe_.encode_position(positions[i], cos_buf.data(), sin_buf.data());
            fp32* row = grad + i * d;
            for (int head = 0; head < h_; ++head)
                apply_inverse_rope_inplace(row + head * d_k_, d_k_, cos_buf.data(), sin_buf.data());
        }
    }

    void encode_positions(int n, fp32* out) {
        int pairs_ = cfg_.pairs();
        std::vector<fp32> cos_buf(pairs_), sin_buf(pairs_);
        for (int i = 0; i < n; ++i) {
            hdpe_.encode_position(i, cos_buf.data(), sin_buf.data());
            for (int j = 0; j < pairs_; ++j) {
                out[i * 2 * pairs_ + 2 * j]     = cos_buf[j];
                out[i * 2 * pairs_ + 2 * j + 1] = sin_buf[j];
            }
        }
    }

    // ---- Accessors ----
    const InputEngineConfig& config() const { return cfg_; }
    HDPE& hdpe() { return hdpe_; }
    const HDPE& hdpe() const { return hdpe_; }
    HFAQEOutput* embedder() { return embedder_.get(); }
    HFAQE* raw_model() { return embedder_ ? embedder_->raw_model() : nullptr; }
    const HFAQE* raw_model() const { return embedder_ ? embedder_->raw_model() : nullptr; }

private:
    void apply_position_encoding(fp32* data, int n, fp32* out) {
        int pairs_ = cfg_.pairs();
        int d_k_   = cfg_.d_k();
        int h_     = cfg_.h;
        int d      = cfg_.d;
        std::vector<fp32> cos_buf(pairs_), sin_buf(pairs_);

        for (int i = 0; i < n; ++i) {
            hdpe_.encode_position(i, cos_buf.data(), sin_buf.data());
            fp32* row = data + i * d;
            for (int head = 0; head < h_; ++head) {
                apply_rope_inplace(row + head * d_k_, d_k_,
                                   cos_buf.data(), sin_buf.data());
            }
        }
        if (out != data) {
            std::memcpy(out, data, static_cast<size_t>(n) * d * sizeof(fp32));
        }
    }

    void apply_position_encoding_at(fp32* data, const int* positions, int n, fp32* out) {
        int pairs_ = cfg_.pairs();
        int d_k_   = cfg_.d_k();
        int h_     = cfg_.h;
        int d      = cfg_.d;
        std::vector<fp32> cos_buf(pairs_), sin_buf(pairs_);

        for (int i = 0; i < n; ++i) {
            hdpe_.encode_position(positions[i], cos_buf.data(), sin_buf.data());
            fp32* row = data + i * d;
            for (int head = 0; head < h_; ++head) {
                apply_rope_inplace(row + head * d_k_, d_k_,
                                   cos_buf.data(), sin_buf.data());
            }
        }
        if (out != data) {
            std::memcpy(out, data, static_cast<size_t>(n) * d * sizeof(fp32));
        }
    }

    InputEngineConfig cfg_;
    std::unique_ptr<HFAQEOutput> embedder_;
    HDPEConfig hdpe_cfg_;
    HDPE hdpe_;
};

#endif // INPUT_ENGINE_CPP
