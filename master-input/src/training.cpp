// =============================================================================
// Training — Nexus Unified Input Layer Training Pipeline
//   Production-grade: HFAQE embedding training with HDPE positional encoding.
//   Integrated LM head (tied), next-token prediction, checkpoint save/load.
// =============================================================================
#ifndef INPUT_TRAINING_CPP
#define INPUT_TRAINING_CPP

#include "input_engine.cpp"
#include "../../Component-1.2_Token-Embedding/src/Storage.cpp"

// =============================================================================
// §1 — Training Configuration
// =============================================================================

struct InputTrainingConfig {
    // Optimizer
    fp32 lr          = 1e-3f;
    fp32 lr_min      = 1e-5f;
    fp32 beta        = 0.3f;
    int  T_realloc   = 300;
    int  T_ortho     = 100;
    int  warmup      = 100;

    // Loss weights (HFAQE auxiliary losses)
    fp32 lambda_semantic = 1.0f;
    fp32 lambda_align    = 0.1f;
    fp32 lambda_quant    = 0.01f;
    fp32 lambda_ortho    = 0.001f;

    // Data
    int  batch_size  = 8;
    int  seq_len     = 128;
    int  total_steps = 1000;

    // Logging / checkpoint
    int  log_every   = 10;
    int  val_every   = 100;
    int  save_every  = 500;

    // Checkpoint
    std::string ckpt_dir  = "checkpoints";
    std::string ckpt_name = "input_trio";
};

// =============================================================================
// §2 — Training Metrics
// =============================================================================

struct TrainingMetrics {
    fp32 loss          = 0.0f;
    fp32 loss_semantic = 0.0f;
    fp32 loss_align    = 0.0f;
    fp32 loss_quant    = 0.0f;
    fp32 grad_norm     = 0.0f;
    fp32 lr_current    = 0.0f;
    int   step         = 0;
    int   n_tokens     = 0;
    double ms          = 0.0;

    void print() const {
        std::printf("  step=%4d  loss=%.4f  (sem=%.4f align=%.4f quant=%.4f)"
                    "  grad=%.2e  lr=%.2e  %d tok  %.1f ms\n",
                    step, loss, loss_semantic, loss_align, loss_quant,
                    grad_norm, lr_current, n_tokens, ms);
    }
};

// =============================================================================
// §3 — LR Schedule (linear warmup → cosine decay)
// =============================================================================
static fp32 compute_lr_schedule(int step, int warmup, int total_steps,
                                 fp32 lr_max, fp32 lr_min) {
    if (step < warmup)
        return lr_max * static_cast<fp32>(step + 1) / static_cast<fp32>(warmup);
    fp32 progress = static_cast<fp32>(step - warmup)
                  / static_cast<fp32>(std::max(1, total_steps - warmup));
    fp32 cosine = 0.5f * (1.0f + std::cos(static_cast<fp32>(M_PI) * progress));
    return lr_min + (lr_max - lr_min) * cosine;
}

// =============================================================================
// §4 — Loss Computation
//   Forward: X_rot → lm_head → logits → softmax CE (next-token prediction)
//   Gradient: dL/dX_rot via hot + cold backward through LM head
// =============================================================================

struct LossResult {
    fp32             loss;
    std::vector<fp32> dL_dX;
    int              n_toks;
};

static LossResult compute_loss(const HFAQE& model,
                                const std::vector<fp32>& X_rot,
                                const std::vector<int>& ids) {
    int n = static_cast<int>(ids.size());
    int d = model.cfg.d;
    int V = model.cfg.V;
    LossResult out;
    out.dL_dX.assign(static_cast<size_t>(n) * d, 0.0f);
    out.n_toks = 0;

    if (n < 2) { out.loss = 0.0f; return out; }

    std::vector<fp32> logits(V);
    double total_nll = 0.0;
    int n_steps = n - 1;

    for (int i = 0; i < n_steps; ++i) {
        const fp32* hi = X_rot.data() + static_cast<ptrdiff_t>(i) * d;
        int target     = ids[i + 1];

        model.lm_head(hi, logits.data());

        fp32 mx = -1e30f;
        for (int t = 0; t < V; ++t)
            if (logits[t] > mx) mx = logits[t];

        fp32 sum_exp = 0.0f;
        for (int t = 0; t < V; ++t) {
            logits[t] = std::exp(logits[t] - mx);
            sum_exp  += logits[t];
        }
        fp32 inv = 1.0f / (sum_exp + 1e-10f);
        for (int t = 0; t < V; ++t) logits[t] *= inv;

        fp32 p_target = std::max(logits[target], 1e-10f);
        total_nll -= std::log(p_target);

        fp32 inv_n = 1.0f / static_cast<fp32>(n_steps);
        fp32* dhi  = out.dL_dX.data() + static_cast<ptrdiff_t>(i) * d;

        // Hot contribution
        for (int slot = 0; slot < model.hot.K; ++slot) {
            int gid  = model.hot.global_ids[slot];
            fp32 dg  = (logits[gid] - (gid == target ? 1.0f : 0.0f)) * inv_n;
            if (std::abs(dg) < 1e-12f) continue;
            const int8_t* qr = model.hot.row_q(slot);
            const fp32*   sr = model.hot.row_s(slot);
            int m_blk        = model.cfg.m();
            for (int b = 0; b < m_blk; ++b) {
                int start = b * model.cfg.B;
                int end   = std::min(start + model.cfg.B, d);
                fp32 s    = sr[b];
                for (int j = start; j < end; ++j)
                    dhi[j] += dg * s * static_cast<fp32>(qr[j]);
            }
        }

        // Cold contribution
        std::vector<fp32> z_c(model.cfg.r, 0.0f);
        for (int cs = 0; cs < model.cold.Vc; ++cs) {
            int gid  = model.cold.global_ids[cs];
            fp32 dg  = (logits[gid] - (gid == target ? 1.0f : 0.0f)) * inv_n;
            if (std::abs(dg) < 1e-12f) continue;
            const fp16* ar = model.cold.row_a(cs);
            for (int k = 0; k < model.cfg.r; ++k)
                z_c[k] += dg * bf16_to_f32(ar[k]);
        }
        for (int k = 0; k < model.cfg.r; ++k) {
            if (std::abs(z_c[k]) < 1e-12f) continue;
            const fp16* bk = model.cold.basis_col(k);
            for (int j = 0; j < d; ++j)
                dhi[j] += z_c[k] * bf16_to_f32(bk[j]);
        }
    }

    out.loss   = static_cast<fp32>(total_nll / n_steps);
    out.n_toks = n_steps;
    return out;
}

// =============================================================================
// §5 — Input Training Pipeline
//   Production-grade: HFAQE training + HDPE position encoding + checkpointing
// =============================================================================

class InputTrainingPipeline {
public:
    InputTrainingPipeline() = default;
    ~InputTrainingPipeline() = default;

    // ---- Init ----
    bool init(const InputEngineConfig& engine_cfg, const InputTrainingConfig& train_cfg) {
        engine_cfg_ = engine_cfg;
        train_cfg_  = train_cfg;

        // Init InputEngine (creates HFAQE model + HDPE tables)
        if (!engine_.init(engine_cfg_)) {
            std::fprintf(stderr, "[Training] InputEngine init failed\n");
            return false;
        }

        // Access the HFAQE model and set up training state
        model_ = engine_.raw_model();
        if (!model_) {
            std::fprintf(stderr, "[Training] No HFAQE model available\n");
            return false;
        }

        // Init HDPE tables for the backward (inverse rotation)
        hdpe_cfg_.B     = engine_cfg.B;
        hdpe_cfg_.L     = engine_cfg.L;
        hdpe_cfg_.d     = engine_cfg.d;
        hdpe_cfg_.pairs = engine_cfg.pairs();
        hdpe_cfg_.base  = engine_cfg.base;
        hdpe_.init(hdpe_cfg_);

        // Allocate per-position cos/sin buffers
        int max_pos = std::max(train_cfg.seq_len, train_cfg.batch_size * train_cfg.seq_len);
        cos_buf_ = std::vector<fp32>(static_cast<size_t>(max_pos) * engine_cfg.pairs());
        sin_buf_ = std::vector<fp32>(static_cast<size_t>(max_pos) * engine_cfg.pairs());

        // Init HFAQE training (AdamW, master latent, tier alloc)
        model_->setup_training(train_cfg.beta, train_cfg.T_realloc, train_cfg.T_ortho);

        // Set up checkpoint manager
        CheckpointManager::Config ccfg;
        ccfg.ckpt_dir  = train_cfg.ckpt_dir;
        ccfg.base_name = train_cfg.ckpt_name;
        ccfg.compress  = false;
        ccfg.checksums = true;
        ccfg.keep_last = 3;
        ckpt_mgr_ = std::make_unique<CheckpointManager>(ccfg);

        return true;
    }

    // ---- Single training step ----
    //   zero_grad → embed → rotate → compute_loss → inverse_rotate → backward → apply_gradients
    TrainingMetrics train_step(const std::vector<int>& token_ids) {
        TrainingMetrics m;
        m.step   = step_;
        m.n_tokens = static_cast<int>(token_ids.size());
        auto t0 = Clock::now();

        int n = static_cast<int>(token_ids.size());
        int d = engine_cfg_.d;
        int pairs_ = engine_cfg_.pairs();
        int d_k_   = engine_cfg_.d_k();
        int h_     = engine_cfg_.h;

        // 1. Zero gradients
        model_->zero_grad();

        // 2. Embed (raw, no position encoding)
        std::vector<fp32> X(static_cast<size_t>(n) * d);
        model_->forward(token_ids.data(), n, X.data());

        // 3. Position-encode (RoPE at natural positions 0..n-1)
        for (int i = 0; i < n; ++i) {
            hdpe_.encode_position(i, cos_buf_.data(), sin_buf_.data());
            fp32* row = X.data() + i * d;
            for (int head = 0; head < h_; ++head)
                apply_rope_inplace(row + head * d_k_, d_k_, cos_buf_.data(), sin_buf_.data());
        }

        // 4. Compute LM head + CE loss + dL/dX_rot
        auto lr = compute_loss(*model_, X, token_ids);

        // 5. Inverse-rotate gradient before feeding to HFAQE backward
        for (int i = 0; i < n; ++i) {
            hdpe_.encode_position(i, cos_buf_.data(), sin_buf_.data());
            fp32* row = lr.dL_dX.data() + i * d;
            for (int head = 0; head < h_; ++head)
                apply_inverse_rope_inplace(row + head * d_k_, d_k_, cos_buf_.data(), sin_buf_.data());
        }

        // 6. HFAQE backward (STE gradient accumulation)
        model_->backward(lr.dL_dX.data(), token_ids.data(), n);

        // 7. Capture grad norm BEFORE apply_gradients (which zeroes dW)
        m.grad_norm  = model_->master.grad_norm_master();

        // 8. Apply gradients (AdamW + auxiliary losses + compress)
        fp32 current_lr = compute_lr_schedule(step_, train_cfg_.warmup,
                                              train_cfg_.total_steps,
                                              train_cfg_.lr, train_cfg_.lr_min);
        model_->apply_gradients(current_lr);

        auto t1 = Clock::now();

        // 9. Populate rest of metrics
        m.loss       = lr.loss;
        m.lr_current = current_lr;
        m.ms         = std::chrono::duration<double, std::milli>(t1 - t0).count();

        step_++;
        return m;
    }

    // ---- Validation ----
    // Compute mean CE loss + perplexity on held-out token sequences.
    float validate(const std::vector<int>& token_ids) {
        int n = static_cast<int>(token_ids.size());
        int d = engine_cfg_.d;
        int pairs_ = engine_cfg_.pairs();
        int d_k_   = engine_cfg_.d_k();
        int h_     = engine_cfg_.h;

        std::vector<fp32> X(static_cast<size_t>(n) * d);
        model_->forward(token_ids.data(), n, X.data());

        for (int i = 0; i < n; ++i) {
            hdpe_.encode_position(i, cos_buf_.data(), sin_buf_.data());
            fp32* row = X.data() + i * d;
            for (int head = 0; head < h_; ++head)
                apply_rope_inplace(row + head * d_k_, d_k_, cos_buf_.data(), sin_buf_.data());
        }

        auto lr = compute_loss(*model_, X, token_ids);
        return lr.loss;
    }

    // ---- Checkpoint save ----
    bool save_checkpoint(const std::string& tag) {
        if (!ckpt_mgr_) return false;
        std::vector<fp32> freq;
        if (model_->token_frequencies.size() == static_cast<size_t>(model_->cfg.V))
            freq = model_->token_frequencies;

        // Build AdamW state
        NexAdamState adam;
        adam.m_master    = model_->adam_W.m;
        adam.v_master    = model_->adam_W.v;
        adam.step_master = model_->adam_W.step;

        NexCheckpointMeta meta;
        meta.global_step   = step_;
        meta.best_val_loss = best_val_loss_;
        meta.best_val_ppl  = std::exp(std::min(best_val_loss_, 20.0f));

        const std::vector<fp32>* freq_ptr = freq.empty() ? nullptr : &freq;
        auto path = ckpt_mgr_->save(*model_, meta, tag, &adam, freq_ptr);
        return !path.empty();
    }

    // ---- Checkpoint load ----
    bool load_checkpoint(const std::string& path = "") {
        if (!ckpt_mgr_) return false;

        if (path.empty()) {
            NexCheckpointMeta meta;
            if (!ckpt_mgr_->load(*model_, meta, nullptr)) return false;
            step_         = meta.global_step;
            best_val_loss_ = meta.best_val_loss;
            return true;
        }

        try {
            auto r = NexReader::open(path);
            HFAQEConfig fc = r.config();
            if (fc.V != model_->cfg.V || fc.d != model_->cfg.d ||
                fc.r != model_->cfg.r || fc.K != model_->cfg.K) {
                std::fprintf(stderr, "[Training] checkpoint config mismatch\n");
                return false;
            }
            r.load(*model_);
            // Load Adam optimizer state (momentum + velocity)
            if (r.has_adam()) {
                NexAdamState adam;
                r.load_adam(adam);
                model_->adam_W.m    = std::move(adam.m_master);
                model_->adam_W.v    = std::move(adam.v_master);
                model_->adam_W.step = adam.step_master;
            }
            auto meta = r.meta();
            step_         = meta.global_step;
            best_val_loss_ = meta.best_val_loss;
            r.close();

            // Rebuild lookup maps
            model_->hot.idx.clear();
            for (int s = 0; s < model_->hot.K; ++s)
                model_->hot.idx[model_->hot.global_ids[s]] = s;
            model_->cold.idx.clear();
            for (int s = 0; s < model_->cold.Vc; ++s)
                model_->cold.idx[model_->cold.global_ids[s]] = s;

            return true;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[Training] load failed: %s\n", e.what());
            return false;
        }
    }

    // ---- Validate gradient flow ----
    // End-to-end structural check: forward → loss → backward produces finite
    // gradients with non-zero norm.
    bool validate_gradient_flow() {
        std::printf("[Training] Validating gradient flow...\n");

        std::mt19937_64 rng(42);
        std::uniform_int_distribution<int> dist(0, engine_cfg_.V - 1);
        std::vector<int> ids(static_cast<size_t>(train_cfg_.batch_size * train_cfg_.seq_len));
        for (auto& id : ids) id = dist(rng);

        auto m = train_step(ids);

        bool loss_finite  = std::isfinite(m.loss);
        bool grad_finite  = std::isfinite(m.grad_norm);
        bool grad_nonzero = m.grad_norm > 1e-10f;

        if (!loss_finite)  std::fprintf(stderr, "  [FAIL] Loss is NaN/Inf\n");
        if (!grad_finite)  std::fprintf(stderr, "  [FAIL] Grad norm is NaN/Inf\n");
        if (!grad_nonzero) std::fprintf(stderr, "  [FAIL] Grad norm is zero\n");

        bool ok = loss_finite && grad_finite && grad_nonzero;
        std::printf("  [%s] Gradient flow OK (loss=%.4f, grad_norm=%.2e)\n",
                    ok ? "PASS" : "FAIL", m.loss, m.grad_norm);
        return ok;
    }

    // ---- Accessors ----
    InputEngine& engine() { return engine_; }
    HFAQE* model() { return model_; }
    int step() const { return step_; }

private:
    InputEngineConfig   engine_cfg_;
    InputTrainingConfig train_cfg_;
    InputEngine         engine_;
    HFAQE*              model_ = nullptr;

    // HDPE for backward (inverse rotation)
    HDPE       hdpe_;
    HDPEConfig hdpe_cfg_;
    std::vector<fp32> cos_buf_, sin_buf_;

    // Checkpoint manager
    std::unique_ptr<CheckpointManager> ckpt_mgr_;

    // State
    int   step_          = 0;
    fp32  best_val_loss_ = 1e30f;
};

#endif // INPUT_TRAINING_CPP
