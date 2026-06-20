// =============================================================================
// Master Input — Nexus Unified Input Layer Demo
//   End-to-end: init → tokenize → embed → position-encode → train loop
// =============================================================================
#include "training.cpp"

#include <cstdio>
#include <cmath>

static const char* SEP =
    "================================================================\n";

int main() {
    std::printf("%s", SEP);
    std::printf("  Nexus Master Input — Unified Input Layer\n");
    std::printf("  Tokenize  →  Embed (HFAQE)  →  Position-Encode (HDPE)\n");
    std::printf("%s\n", SEP);

    // ---- Config ----
    // NOTE: HFAQE cold-tier SVD scales O(V·d·r) — expect 15-20s init at d=256.
    //       d=512 would be ~4x slower (d·V·r cold SVD).
    InputEngineConfig cfg;
    cfg.V  = 16000;
    cfg.d  = 256;
    cfg.r  = 64;
    cfg.K  = 1024;
    cfg.B  = 64;
    cfg.L  = 4;
    cfg.h  = 8;
    cfg.base = 10000.0f;

    int d = cfg.d;

    std::printf("  Config: V=%d  d=%d  r=%d  K=%d  B=%d  L=%d  h=%d  base=%.0f\n",
                cfg.V, cfg.d, cfg.r, cfg.K, cfg.B, cfg.L, cfg.h, cfg.base);
    std::printf("  d_k=%d  pairs=%d  range=%lld\n",
                cfg.d_k(), cfg.pairs(),
                static_cast<long long>(cfg.range()));

    // ---- Init ----
    auto t0 = Clock::now();
    InputEngine engine;
    engine.init(cfg);
    auto t1 = Clock::now();
    std::printf("\n[Init] %.0f ms\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count());

    // ---- Forward: token IDs → position-aware embeddings ----
    std::vector<int> token_ids = {101, 2056, 789, 42, 999, 333, 777, 1, 5000, 12345};
    int n = static_cast<int>(token_ids.size());
    std::vector<fp32> output(static_cast<size_t>(n) * d);

    auto t2 = Clock::now();
    engine.forward(token_ids.data(), n, output.data());
    auto t3 = Clock::now();
    double fwd_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::printf("[Forward %d tokens] %.3f ms (%.0f tok/s)\n", n, fwd_ms, n / (fwd_ms / 1000.0));

    // ---- Sample output ----
    std::printf("\n--- Sample: first 8 dims of each position-encoded token ---\n");
    for (int i = 0; i < n && i < 5; ++i) {
        std::printf("  tok[%d] (id=%d):", i, token_ids[i]);
        for (int j = 0; j < 8 && j < d; ++j)
            std::printf(" %+.4f", output[static_cast<size_t>(i) * d + j]);
        std::printf("\n");
    }

    // ---- Norm check ----
    std::vector<fp32> raw(static_cast<size_t>(n) * d);
    engine.embed(token_ids.data(), n, raw.data());
    std::printf("\n--- Norm preservation ---\n");
    bool norm_ok = true;
    for (int i = 0; i < n; ++i) {
        fp32 nr = 0.0f, ne = 0.0f;
        for (int j = 0; j < d; ++j) {
            nr += raw[static_cast<size_t>(i) * d + j] * raw[static_cast<size_t>(i) * d + j];
            ne += output[static_cast<size_t>(i) * d + j] * output[static_cast<size_t>(i) * d + j];
        }
        nr = std::sqrt(nr); ne = std::sqrt(ne);
        fp32 diff = std::abs(nr - ne);
        if (diff > 1e-4f) norm_ok = false;
        if (i < 4 || !norm_ok)
            std::printf("  pos %d: raw=%.6f  encoded=%.6f  diff=%.2e  %s\n",
                        i, nr, ne, diff, diff < 1e-4f ? "OK" : "FAIL");
    }
    std::printf("  Norm: %s\n", norm_ok ? "ALL OK" : "FAIL");

    // ---- Compact position encoding output ----
    std::printf("\n--- Position encoding (cos/sin for positions 0..4, pair 0..3) ---\n");
    int pairs = cfg.pairs();
    std::vector<fp32> pos_enc(static_cast<size_t>(n) * 2 * pairs);
    engine.encode_positions(n, pos_enc.data());
    for (int p = 0; p < 5; ++p) {
        std::printf("  pos %d:", p);
        for (int i = 0; i < 4 && i < pairs; ++i)
            std::printf(" (%.4f,%.4f)", pos_enc[static_cast<size_t>(p) * 2 * pairs + 2 * i],
                                         pos_enc[static_cast<size_t>(p) * 2 * pairs + 2 * i + 1]);
        std::printf("\n");
    }

    // ---- Training validation ----
    std::printf("\n--- Training Pipeline ---\n");
    InputTrainingConfig train_cfg;
    train_cfg.batch_size  = 2;
    train_cfg.seq_len     = 16;
    train_cfg.total_steps = 5;
    train_cfg.lr          = 1e-4f;

    InputTrainingPipeline trainer;
    trainer.init(cfg, train_cfg);

    // Validate gradient flow
    bool grad_ok = trainer.validate_gradient_flow();

    // Run a few steps
    std::mt19937_64 rng(123);
    std::uniform_int_distribution<int> tok_dist(0, cfg.V - 1);
    std::printf("\n--- Training steps (synthetic data) ---\n");
    int good_steps = 0;
    for (int s = 0; s < 3; ++s) {
        std::vector<int> batch(static_cast<size_t>(train_cfg.batch_size * train_cfg.seq_len));
        for (auto& id : batch) id = tok_dist(rng);
        auto m = trainer.train_step(batch);
        m.print();
        if (std::isfinite(m.loss)) good_steps++;
    }

    // ---- Verdict ----
    std::printf("\n%s", SEP);
    bool all_ok = norm_ok && grad_ok && good_steps == 3;
    std::printf("  VERDICT: %s\n", all_ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    std::printf("%s", SEP);

    return all_ok ? 0 : 1;
}
