// =============================================================================
// test_train_numerical — Numerical Correctness Validation Suite
//   Verifies the training pipeline produces correct gradients, converging loss,
//   and survives checkpoint round-trip.
// =============================================================================
#include "../src/training.cpp"

#include <cstdio>
#include <cmath>
#include <chrono>

using Clock = std::chrono::high_resolution_clock;

static int g_passed = 0;
static int g_failed = 0;

#define EXPECT_TRUE(cond, msg) do { \
    if (!(cond)) { \
        std::printf("  [FAIL] %s  (line %d)\n", msg, __LINE__); \
        g_failed++; \
    } else { \
        std::printf("  [PASS] %s\n", msg); \
        g_passed++; \
    } \
} while(0)

#define EXPECT_NEAR(a, b, tol, msg) do { \
    fp32 _diff = std::abs((a) - (b)); \
    if (_diff > tol) { \
        std::printf("  [FAIL] %s  (%.6f vs %.6f, diff=%.2e > %.2e)  line %d\n", \
                    msg, (double)(a), (double)(b), (double)_diff, (double)tol, __LINE__); \
        g_failed++; \
    } else { \
        std::printf("  [PASS] %s  (%.6f ≈ %.6f)\n", msg, (double)(a), (double)(b)); \
        g_passed++; \
    } \
} while(0)

static void print_section(const char* name) {
    std::printf("\n=== %s ===\n", name);
}

static InputEngineConfig small_cfg() {
    InputEngineConfig cfg;
    cfg.V = 512;
    cfg.d = 32;
    cfg.r = 8;
    cfg.K = 128;
    cfg.B = 64;
    cfg.L = 2;
    cfg.h = 4;
    cfg.base = 10000.0f;
    return cfg;
}

// ---- §1 Gradient flow validation ----
static void test_gradient_flow() {
    print_section("§1 Gradient Flow");
    auto engine_cfg = small_cfg();
    InputTrainingConfig train_cfg;
    train_cfg.total_steps = 1;
    train_cfg.batch_size  = 2;
    train_cfg.seq_len     = 8;
    train_cfg.lr          = 1e-4f;
    train_cfg.ckpt_dir    = "/tmp/ckpt_test";

    InputTrainingPipeline trainer;
    bool init_ok = trainer.init(engine_cfg, train_cfg);
    EXPECT_TRUE(init_ok, "§1.1 Training pipeline init");

    // Gradient flow validation runs one train_step and checks finite loss + grad
    bool grad_ok = trainer.validate_gradient_flow();
    EXPECT_TRUE(grad_ok, "§1.2 Gradient flow (finite loss, finite non-zero grad)");
}

// ---- §2 Loss convergence ----
static void test_loss_convergence() {
    print_section("§2 Loss Convergence");
    auto engine_cfg = small_cfg();
    InputTrainingConfig train_cfg;
    train_cfg.total_steps = 20;
    train_cfg.batch_size  = 2;
    train_cfg.seq_len     = 8;
    train_cfg.lr          = 1e-2f;
    train_cfg.log_every   = 100;
    train_cfg.val_every   = 100;
    train_cfg.save_every  = 100;
    train_cfg.ckpt_dir    = "/tmp/ckpt_test";

    InputTrainingPipeline trainer;
    trainer.init(engine_cfg, train_cfg);

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int> dist(0, engine_cfg.V - 1);

    fp32 loss_first = 1e10f;
    fp32 loss_last  = 1e10f;

    for (int s = 0; s < 20; ++s) {
        std::vector<int> batch(static_cast<size_t>(train_cfg.batch_size * train_cfg.seq_len));
        for (auto& id : batch) id = dist(rng);
        auto m = trainer.train_step(batch);
        if (s == 0)  loss_first = m.loss;
        if (s == 19) loss_last  = m.loss;
    }

    EXPECT_TRUE(std::isfinite(loss_first), "§2.1 Initial loss is finite");
    EXPECT_TRUE(std::isfinite(loss_last),  "§2.2 Final loss is finite");
    // Loss should decrease (next-token prediction on synthetic data)
    bool converged = loss_last < loss_first;
    EXPECT_TRUE(converged, "§2.3 Loss decreases over training");
    if (!converged)
        std::printf("    First=%.4f  Last=%.4f  ratio=%.4f\n",
                    (double)loss_first, (double)loss_last,
                    (double)(loss_last / loss_first));
}

// ---- §3 Checkpoint round-trip ----
static void test_checkpoint_roundtrip() {
    print_section("§3 Checkpoint Round-Trip");

    auto engine_cfg = small_cfg();
    InputTrainingConfig train_cfg;
    train_cfg.total_steps = 5;
    train_cfg.batch_size  = 2;
    train_cfg.seq_len     = 8;
    train_cfg.lr          = 1e-3f;
    train_cfg.log_every   = 100;
    train_cfg.val_every   = 100;
    train_cfg.save_every  = 100;
    train_cfg.ckpt_dir    = "/tmp/ckpt_test";
    train_cfg.ckpt_name   = "test_roundtrip";

    // Train pipeline A
    InputTrainingPipeline trainer_a;
    trainer_a.init(engine_cfg, train_cfg);

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int> dist(0, engine_cfg.V - 1);

    // Train a few steps
    for (int s = 0; s < 3; ++s) {
        std::vector<int> batch(static_cast<size_t>(train_cfg.batch_size * train_cfg.seq_len));
        for (auto& id : batch) id = dist(rng);
        trainer_a.train_step(batch);
    }

    // Save checkpoint
    bool saved = trainer_a.save_checkpoint("test_roundtrip");
    EXPECT_TRUE(saved, "§3.1 Checkpoint saved successfully");

    // Create pipeline B (same config) and load
    InputTrainingPipeline trainer_b;
    trainer_b.init(engine_cfg, train_cfg);
    bool loaded = trainer_b.load_checkpoint("/tmp/ckpt_test/test_roundtrip_test_roundtrip.nex");
    EXPECT_TRUE(loaded, "§3.2 Checkpoint loaded successfully");

    // Compare step count
    EXPECT_TRUE(trainer_b.step() == 3, "§3.3 Step count preserved after load");

    // Run one more step on both and compare loss
    std::vector<int> batch(static_cast<size_t>(train_cfg.batch_size * train_cfg.seq_len));
    for (auto& id : batch) id = dist(rng);

    auto ma = trainer_a.train_step(batch);
    auto mb = trainer_b.train_step(batch);
    EXPECT_NEAR(ma.loss, mb.loss, 1e-4f, "§3.4 Loss matches after checkpoint round-trip");
}

// ---- §4 Position encoding gradient ----
static void test_position_encoding_gradient() {
    print_section("§4 Position Encoding Gradient (Inverse Rotation)");

    // Verify that forward(+) then inverse(forward(x)) = x for random data
    int d_k = 8;
    int pairs = d_k / 2;

    std::mt19937_64 rng(42);
    std::normal_distribution<fp32> dist(0.0f, 1.0f);

    std::vector<fp32> x(d_k), cos_(pairs), sin_(pairs);
    for (int i = 0; i < d_k; ++i) x[i] = dist(rng);
    for (int i = 0; i < pairs; ++i) {
        fp32 angle = dist(rng);
        cos_[i] = std::cos(angle);
        sin_[i] = std::sin(angle);
    }

    // Save copy
    std::vector<fp32> x_copy = x;

    // Forward RoPE
    apply_rope(x.data(), d_k, cos_.data(), sin_.data(), x.data());

    // Verify forward changed something
    bool changed = false;
    for (int i = 0; i < d_k; ++i)
        if (std::abs(x[i] - x_copy[i]) > 1e-6f) { changed = true; break; }
    EXPECT_TRUE(changed, "§4.1 RoPE forward changes values");

    // Now forward again onto a different buffer to verify we get same result
    std::vector<fp32> x_rot(d_k);
    apply_rope(x_copy.data(), d_k, cos_.data(), sin_.data(), x_rot.data());
    for (int i = 0; i < d_k; ++i)
        EXPECT_NEAR(x[i], x_rot[i], 1e-6f, "§4.2 In-place forward matches out-of-place");

    // Inverse rotation
    apply_inverse_rope_inplace(x.data(), d_k, cos_.data(), sin_.data());

    // Should be back to original
    for (int i = 0; i < d_k; ++i)
        EXPECT_NEAR(x[i], x_copy[i], 1e-5f, "§4.3 Inv(RoPE(forward(x))) == x");
}

// ---- §5 HDPE position encoding ----
static void test_hdpe_integration() {
    print_section("§5 HDPE Position Encoding Integration");

    auto cfg = small_cfg();
    InputEngine engine;
    engine.init(cfg);

    int n = 5;
    int pairs = cfg.pairs();
    std::vector<fp32> pos_enc(static_cast<size_t>(n) * 2 * pairs);
    engine.encode_positions(n, pos_enc.data());

    // Position 0: cos=1, sin=0
    bool pos0_identity = true;
    for (int i = 0; i < pairs; ++i) {
        if (std::abs(pos_enc[2 * i] - 1.0f) > 1e-6f ||
            std::abs(pos_enc[2 * i + 1]) > 1e-6f) {
            pos0_identity = false;
            break;
        }
    }
    EXPECT_TRUE(pos0_identity, "§5.1 Position 0 → cos=1, sin=0");

    // Different positions → different (cos, sin) pairs
    bool different = false;
    for (int i = 0; i < pairs; ++i) {
        if (std::abs(pos_enc[2 * i] - pos_enc[2 * i + 2 * pairs]) > 1e-6f ||
            std::abs(pos_enc[2 * i + 1] - pos_enc[2 * i + 1 + 2 * pairs]) > 1e-6f) {
            different = true; break;
        }
    }
    EXPECT_TRUE(different, "§5.2 Positions 0 and 1 have different encodings");
}

// ---- Runner ----
int main() {
    std::printf("============================================\n");
    std::printf("  Numerical Correctness — Training Pipeline\n");
    std::printf("============================================\n");

    test_gradient_flow();
    test_loss_convergence();
    test_checkpoint_roundtrip();
    test_position_encoding_gradient();
    test_hdpe_integration();

    std::printf("\n============================================\n");
    std::printf("  Results: %d PASSED, %d FAILED\n", g_passed, g_failed);
    std::printf("============================================\n");

    return g_failed > 0 ? 1 : 0;
}
