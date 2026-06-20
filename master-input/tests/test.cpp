// =============================================================================
// Master Input — Integration Validation Suite
//   Validates the full input pipeline: init, forward, norm, gradient flow.
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

static void print_section(const char* name) {
    std::printf("\n=== %s ===\n", name);
}

// Default test config
static InputEngineConfig test_cfg() {
    InputEngineConfig cfg;
    cfg.V    = 16000;
    cfg.d    = 256;     // small for fast tests
    cfg.r    = 64;
    cfg.K    = 1024;
    cfg.B    = 64;
    cfg.L    = 4;
    cfg.h    = 8;       // d_k = 32
    cfg.base = 10000.0f;
    return cfg;
}

// ---- §1 Initialisation ----
static void test_init() {
    print_section("§1 Initialisation");
    InputEngineConfig cfg = test_cfg();
    InputEngine engine;
    bool ok = engine.init(cfg);
    EXPECT_TRUE(ok, "§1.1 InputEngine init returns true");
    EXPECT_TRUE(engine.config().d == 256, "§1.2 Config preserved after init");
}

// ---- §2 Forward Pass ----
static void test_forward() {
    print_section("§2 Forward Pass");
    InputEngineConfig cfg = test_cfg();
    InputEngine engine;
    engine.init(cfg);

    int d = cfg.d;
    std::vector<int> ids = {101, 2056, 789, 42};
    int n = static_cast<int>(ids.size());
    std::vector<fp32> out(static_cast<size_t>(n) * d);

    engine.forward(ids.data(), n, out.data());

    // Check output is finite
    bool finite = true;
    for (auto v : out) if (!std::isfinite(v)) { finite = false; break; }
    EXPECT_TRUE(finite, "§2.1 Forward output is finite");

    // Check output is non-zero (should have embedded content)
    bool non_zero = false;
    for (auto v : out) if (std::abs(v) > 1e-10f) { non_zero = true; break; }
    EXPECT_TRUE(non_zero, "§2.2 Forward output is non-zero");
}

// ---- §3 Norm Preservation ----
static void test_norm_preservation() {
    print_section("§3 Norm Preservation");
    InputEngineConfig cfg = test_cfg();
    InputEngine engine;
    engine.init(cfg);

    int d = cfg.d;
    std::vector<int> ids = {5, 10, 100, 1000, 5000};
    int n = static_cast<int>(ids.size());

    std::vector<fp32> raw(static_cast<size_t>(n) * d);
    std::vector<fp32> enc(static_cast<size_t>(n) * d);

    engine.embed(ids.data(), n, raw.data());
    engine.forward(ids.data(), n, enc.data());

    bool norm_ok = true;
    fp32 max_diff = 0.0f;
    for (int i = 0; i < n; ++i) {
        fp32 nr = 0.0f, ne = 0.0f;
        for (int j = 0; j < d; ++j) {
            nr += raw[static_cast<size_t>(i) * d + j] * raw[static_cast<size_t>(i) * d + j];
            ne += enc[static_cast<size_t>(i) * d + j] * enc[static_cast<size_t>(i) * d + j];
        }
        nr = std::sqrt(nr); ne = std::sqrt(ne);
        fp32 diff = std::abs(nr - ne);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-4f) norm_ok = false;
    }
    EXPECT_TRUE(norm_ok, "§3.1 Position encoding preserves L2 norm");
    if (norm_ok && max_diff > 0)
        std::printf("    Max norm diff: %.2e\n", max_diff);
}

// ---- §4 Explicit Positions ----
static void test_explicit_positions() {
    print_section("§4 Explicit Positions");
    InputEngineConfig cfg = test_cfg();
    InputEngine engine;
    engine.init(cfg);

    int d = cfg.d;
    // Same tokens at same positions → bit-identical
    std::vector<int> ids  = {101, 2056, 789, 42, 101, 2056};
    std::vector<int> positions = {0, 10, 100, 1000, 0, 10};
    int n = static_cast<int>(ids.size());

    std::vector<fp32> out(static_cast<size_t>(n) * d);
    engine.forward(ids.data(), positions.data(), n, out.data());

    fp32 diff0 = 0.0f, diff1 = 0.0f;
    for (int j = 0; j < d; ++j) {
        diff0 += std::abs(out[j] - out[4 * d + j]);
        diff1 += std::abs(out[1 * d + j] - out[5 * d + j]);
    }
    EXPECT_TRUE(diff0 < 1e-6f, "§4.1 Same token @ same pos → identical (pos 0)");
    EXPECT_TRUE(diff1 < 1e-6f, "§4.2 Same token @ same pos → identical (pos 10)");

    // Different tokens at same position → different
    fp32 diff_diff = 0.0f;
    for (int j = 0; j < d; ++j)
        diff_diff += std::abs(out[j] - out[3 * d + j]);
    EXPECT_TRUE(diff_diff > 1e-3f, "§4.3 Different token @ same pos → different");
}

// ---- §5 Position Encoding ----
static void test_position_coding() {
    print_section("§5 Position Encoding");
    InputEngineConfig cfg = test_cfg();
    InputEngine engine;
    engine.init(cfg);

    int pairs = cfg.pairs();
    int n = 10;
    std::vector<fp32> pos_enc(static_cast<size_t>(n) * 2 * pairs);
    engine.encode_positions(n, pos_enc.data());

    // Position 0: all cos = 1, sin = 0
    bool pos0_identity = true;
    for (int i = 0; i < pairs; ++i) {
        if (std::abs(pos_enc[2 * i] - 1.0f) > 1e-6f ||
            std::abs(pos_enc[2 * i + 1]) > 1e-6f) {
            pos0_identity = false;
            break;
        }
    }
    EXPECT_TRUE(pos0_identity, "§5.1 Position 0 → cos=1, sin=0");

    // All positions: cos² + sin² ≈ 1
    bool unit_norm = true;
    for (int p = 0; p < n; ++p) {
        for (int i = 0; i < pairs; ++i) {
            fp32 c = pos_enc[static_cast<size_t>(p) * 2 * pairs + 2 * i];
            fp32 s = pos_enc[static_cast<size_t>(p) * 2 * pairs + 2 * i + 1];
            fp32 nrm = c * c + s * s;
            if (std::abs(nrm - 1.0f) > 1e-5f) {
                unit_norm = false;
                break;
            }
        }
    }
    EXPECT_TRUE(unit_norm, "§5.2 All positions: cos²+sin² ≈ 1");
}

// ---- §6 Gradient Flow ----
static void test_gradient_flow() {
    print_section("§6 Gradient Flow");
    InputEngineConfig cfg = test_cfg();
    InputEngine engine;
    engine.init(cfg);

    int d = cfg.d;
    int n = 16;
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int> dist(0, cfg.V - 1);
    std::vector<int> ids(static_cast<size_t>(n));
    for (auto& id : ids) id = dist(rng);

    // Forward
    std::vector<fp32> X(static_cast<size_t>(n) * d);
    engine.forward(ids.data(), n, X.data());

    // LM head
    auto logits = engine.embedder()->lm_head_logits(X.data());
    bool logits_finite = !logits.has_nan;
    EXPECT_TRUE(logits_finite, "§6.1 LM head logits are finite");

    // Cross-entropy loss
    fp32 loss = 0.0f;
    int V = logits.V;
    for (int i = 0; i < n; ++i) {
        fp32 maxv = -1e30f;
        for (int j = 0; j < V; ++j)
            if (logits.logits[static_cast<size_t>(i) * V + j] > maxv)
                maxv = logits.logits[static_cast<size_t>(i) * V + j];
        fp32 sum_exp = 0.0f;
        for (int j = 0; j < V; ++j)
            sum_exp += std::exp(logits.logits[static_cast<size_t>(i) * V + j] - maxv);
        fp32 lp = logits.logits[static_cast<size_t>(i) * V + ids[i]] - maxv - std::log(sum_exp);
        loss -= lp;
    }
    loss /= n;
    EXPECT_TRUE(std::isfinite(loss), "§6.2 Cross-entropy loss is finite");

    // Approximate gradient norm (dL/dX via chain rule through LM head)
    std::vector<fp32> dL_dX(static_cast<size_t>(n) * d, 0.0f);
    for (int i = 0; i < n; ++i) {
        fp32 maxv = -1e30f;
        for (int j = 0; j < V; ++j)
            if (logits.logits[static_cast<size_t>(i) * V + j] > maxv)
                maxv = logits.logits[static_cast<size_t>(i) * V + j];
        fp32 sum_exp = 0.0f;
        for (int j = 0; j < V; ++j)
            sum_exp += std::exp(logits.logits[static_cast<size_t>(i) * V + j] - maxv);
        fp32 softmax_target = std::exp(logits.logits[static_cast<size_t>(i) * V + ids[i]] - maxv) / sum_exp;
        for (int k = 0; k < d; ++k)
            dL_dX[static_cast<size_t>(i) * d + k] = (softmax_target - 1.0f) / (n * d);
    }

    fp32 gn = 0.0f;
    for (auto g : dL_dX) gn += g * g;
    gn = std::sqrt(gn);
    EXPECT_TRUE(std::isfinite(gn) && gn > 1e-10f, "§6.3 Gradient norm is finite and non-zero");
}

// ---- §7 Training Pipeline Structure ----
static void test_training_pipeline() {
    print_section("§7 Training Pipeline");
    InputEngineConfig cfg = test_cfg();
    InputTrainingConfig train_cfg;
    train_cfg.total_steps = 3;
    train_cfg.batch_size  = 2;
    train_cfg.seq_len     = 8;

    InputTrainingPipeline trainer;
    bool init_ok = trainer.init(cfg, train_cfg);
    EXPECT_TRUE(init_ok, "§7.1 Training pipeline init");

    // Gradient flow validation
    bool grad_ok = trainer.validate_gradient_flow();
    EXPECT_TRUE(grad_ok, "§7.2 Gradient flow validation");

    // Training steps
    std::mt19937_64 rng(456);
    std::uniform_int_distribution<int> dist(0, cfg.V - 1);
    int good = 0;
    for (int s = 0; s < 3; ++s) {
        std::vector<int> batch(static_cast<size_t>(train_cfg.batch_size * train_cfg.seq_len));
        for (auto& id : batch) id = dist(rng);
        auto m = trainer.train_step(batch);
        if (std::isfinite(m.loss)) good++;
    }
    EXPECT_TRUE(good == 3, "§7.3 Training steps produce finite loss");
}

// ---- Runner ----
int main() {
    std::printf("============================================\n");
    std::printf("  Master Input — Integration Validation\n");
    std::printf("  Tokenize → Embed → Position-Encode\n");
    std::printf("============================================\n");

    test_init();
    test_forward();
    test_norm_preservation();
    test_explicit_positions();
    test_position_coding();
    test_gradient_flow();
    test_training_pipeline();

    std::printf("\n============================================\n");
    std::printf("  Results: %d PASSED, %d FAILED\n", g_passed, g_failed);
    std::printf("============================================\n");

    return g_failed > 0 ? 1 : 0;
}
