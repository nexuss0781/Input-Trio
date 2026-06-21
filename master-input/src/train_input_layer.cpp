// =============================================================================
// train_input_layer — Standalone Training Entry Point
//   Single command: trains the full input layer end-to-end on synthetic data.
//   Usage: ./build/train_input_layer [--steps N] [--V N] [--d N] ...
// =============================================================================
#include "training.cpp"
#include "repo.cpp"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>

using Clock = std::chrono::high_resolution_clock;

static const char* SEP =
    "================================================================\n";

// ---- Config from args ----
struct CmdConfig {
    int    V            = 4096;    // vocab (small for fast test)
    int    d            = 64;      // model dimension
    int    r            = 16;      // cold rank
    int    K            = 256;     // hot tokens
    int    B            = 64;      // block size
    int    L            = 4;       // HDPE levels
    int    h            = 4;       // heads
    fp32   base         = 10000.0f;
    int    steps        = 200;
    int    batch_size   = 4;
    int    seq_len      = 32;
    fp32   lr           = 1e-3f;
    int    log_every    = 10;
    int    val_every    = 50;
    int    save_every   = 100;
    bool   resume       = false;
    std::string ckpt_dir = "checkpoints";
    std::string data_dir = "Data";
    std::string push_to_hub = "";   // repo_id, e.g. "nexuss0781/Input-Trio"
    std::string token = "";
    std::string token_file = "HF_TOKEN";
};

static CmdConfig parse_args(int argc, char** argv) {
    CmdConfig cfg;
    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--steps"))    cfg.steps    = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--V"))        cfg.V        = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--d"))        cfg.d        = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--r"))        cfg.r        = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--K"))        cfg.K        = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--batch"))    cfg.batch_size = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--seq"))      cfg.seq_len  = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--lr"))       cfg.lr       = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--data_dir")) cfg.data_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--ckpt"))     cfg.ckpt_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--resume"))   cfg.resume   = true;
        else if (!std::strcmp(argv[i], "--push_to_hub")) cfg.push_to_hub = argv[++i];
        else if (!std::strcmp(argv[i], "--token"))      cfg.token      = argv[++i];
        else if (!std::strcmp(argv[i], "--token_file")) cfg.token_file = argv[++i];
    }
    return cfg;
}

// ---- Generate synthetic batch ----
static std::vector<int> make_synthetic_batch(int V, int batch_size, int seq_len,
                                              std::mt19937_64& rng) {
    std::uniform_int_distribution<int> dist(0, V - 1);
    int n = batch_size * seq_len;
    std::vector<int> ids(n);
    for (int i = 0; i < n - 1; ++i)
        ids[i] = dist(rng);
    // Make last token distinct for proper next-token boundary
    ids[n - 1] = dist(rng);
    return ids;
}

// ---- Main ----
int main(int argc, char** argv) {
    CmdConfig cmd = parse_args(argc, argv);

    std::printf("%s", SEP);
    std::printf("  Train Input Layer — End-to-End Training\n");
    std::printf("  Embedding (HFAQE) + Position Encoding (HDPE)\n");
    std::printf("%s\n", SEP);

    std::printf("  Config: V=%d  d=%d  r=%d  K=%d  B=%d  L=%d  h=%d\n",
                cmd.V, cmd.d, cmd.r, cmd.K, cmd.B, cmd.L, cmd.h);
    std::printf("  Steps=%d  batch=%d  seq=%d  lr=%.1e  ckpt=%s  data=%s\n",
                cmd.steps, cmd.batch_size, cmd.seq_len,
                cmd.lr, cmd.ckpt_dir.c_str(),
                cmd.data_dir.empty() ? "synthetic" : cmd.data_dir.c_str());
    std::printf("\n");

    // ---- Engine config ----
    InputEngineConfig engine_cfg;
    engine_cfg.V    = cmd.V;
    engine_cfg.d    = cmd.d;
    engine_cfg.r    = cmd.r;
    engine_cfg.K    = cmd.K;
    engine_cfg.B    = cmd.B;
    engine_cfg.L    = cmd.L;
    engine_cfg.h    = cmd.h;
    engine_cfg.base = cmd.base;

    // ---- Training config ----
    InputTrainingConfig train_cfg;
    train_cfg.total_steps = cmd.steps;
    train_cfg.batch_size  = cmd.batch_size;
    train_cfg.seq_len     = cmd.seq_len;
    train_cfg.lr          = cmd.lr;
    train_cfg.log_every   = cmd.log_every;
    train_cfg.val_every   = cmd.val_every;
    train_cfg.save_every  = cmd.save_every;
    train_cfg.ckpt_dir    = cmd.ckpt_dir;
    train_cfg.ckpt_name   = "input_trio";
    train_cfg.data_dir    = cmd.data_dir;

    // ---- Init pipeline ----
    std::printf("[init] Building HFAQE + HDPE...\n");
    auto t0 = Clock::now();

    InputTrainingPipeline trainer;
    if (!trainer.init(engine_cfg, train_cfg)) {
        std::fprintf(stderr, "FATAL: pipeline init failed\n");
        return 1;
    }

    auto t1 = Clock::now();
    double init_s = std::chrono::duration<double>(t1 - t0).count();
    std::printf("[init] Done (%.1f s)\n", init_s);

    // ---- Resume from checkpoint ----
    if (cmd.resume) {
        if (trainer.load_checkpoint())
            std::printf("[resume] Resumed at step %d\n", trainer.step());
        else
            std::printf("[resume] No checkpoint found, starting fresh\n");
    }

    // ---- Load dataset (real or synthetic) ----
    TextDataset dataset;
    bool use_real_data = !cmd.data_dir.empty();
    if (use_real_data) {
        std::string train_path = cmd.data_dir + "/train.txt";

        // Auto-download if missing
        {
            std::ifstream probe(train_path);
            if (!probe) {
                std::printf("[data] %s not found — running dataset.py to download WikiText ...\n",
                            train_path.c_str());
                std::string dld = "python3 ../dataset.py --data_dir " + cmd.data_dir;
                int rc = std::system(dld.c_str());
                if (rc != 0) {
                    std::fprintf(stderr, "FATAL: dataset.py failed (rc=%d). "
                                "Install datasets: pip install datasets huggingface_hub\n", rc);
                    return 1;
                }
            }
        }

        std::printf("[data] Loading %s ...\n", train_path.c_str());
        if (!dataset.load(train_path)) {
            std::fprintf(stderr, "FATAL: could not load %s\n", train_path.c_str());
            return 1;
        }
        std::printf("[data] Loaded %d lines, %d byte tokens (vocab=256)\n",
                    (int)dataset.lines.size(), dataset.size());
    } else {
        std::printf("[data] Using synthetic data (V=%d)\n", cmd.V);
    }

    // ---- Training loop ----
    std::printf("\n--- Training ---\n");
    fp32 best_loss = 1e30f;
    int train_start_step = trainer.step();

    for (int s = train_start_step; s < cmd.steps; ++s) {
        int seed = s + 42;

        std::vector<int> ids;
        if (use_real_data) {
            ids = dataset.get_batch(cmd.batch_size, cmd.seq_len, 0, seed);
        } else {
            std::mt19937_64 rng(static_cast<uint64_t>(seed));
            ids = make_synthetic_batch(cmd.V, cmd.batch_size, cmd.seq_len, rng);
        }

        auto m = trainer.train_step(ids);

        if (m.loss < best_loss) best_loss = m.loss;

        // Log
        if ((s + 1) % cmd.log_every == 0) {
            std::printf("  step=%4d/%d  loss=%.4f  grad=%.2e  lr=%.2e  %d tok  %.1f ms\n",
                        s + 1, cmd.steps, m.loss, m.grad_norm, m.lr_current,
                        m.n_tokens, m.ms);
        }

        // Validation (on real or synthetic held-out)
        if ((s + 1) % cmd.val_every == 0) {
            fp32 val_loss;
            if (use_real_data) {
                auto val_ids = dataset.get_val_batch(cmd.batch_size, cmd.seq_len);
                val_loss = trainer.validate(val_ids);
            } else {
                std::mt19937_64 val_rng(static_cast<uint64_t>(s + 9999));
                auto val_ids = make_synthetic_batch(cmd.V, cmd.batch_size, cmd.seq_len, val_rng);
                val_loss = trainer.validate(val_ids);
            }
            fp32 val_ppl  = std::exp(std::min(val_loss, 20.0f));
            std::printf("  [val]  step=%4d  val_loss=%.4f  val_ppl=%.2f\n",
                        s + 1, val_loss, val_ppl);
        }

        // Save checkpoint
        if ((s + 1) % cmd.save_every == 0) {
            char tag[32];
            std::snprintf(tag, sizeof(tag), "step_%07d", s + 1);
            trainer.save_checkpoint(tag);
        }
    }

    // ---- Final save ----
    trainer.save_checkpoint("final");

    // ---- Evaluation ----
    auto t2 = Clock::now();
    double total_s = std::chrono::duration<double>(t2 - t1).count();

    std::printf("\n%s", SEP);
    std::printf("  Training complete\n");
    std::printf("  Steps: %d  |  Best loss: %.4f\n", cmd.steps, best_loss);
    std::printf("  Wall time: %.1f s  |  %.2f ms/step\n",
                total_s, total_s * 1000.0 / std::max(1, cmd.steps - train_start_step));
    std::printf("  Checkpoints: %s/\n", cmd.ckpt_dir.c_str());

    // ---- Validation split perplexity ----
    std::printf("\n--- Evaluation ---\n");
    bool all_checks_ok = true;

    // 1. Pipeline sanity checks
    std::printf("[sanity] Checking tokenize→embed→position-encode pipeline...\n");
    {
        std::string msg;
        bool ok = check_pipeline_sanity(trainer, "Hello World!", &msg);
        std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", msg.c_str());
        all_checks_ok = all_checks_ok && ok;
    }
    {
        std::string msg;
        bool ok = check_pipeline_sanity(trainer,
            "The quick brown fox jumps over the lazy dog. 0123456789.", &msg);
        std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", msg.c_str());
        all_checks_ok = all_checks_ok && ok;
    }

    // 2. Full validation-set perplexity (if real data available)
    if (use_real_data) {
        std::string val_path = cmd.data_dir + "/validation.txt";
        TextDataset val_set;
        if (val_set.load(val_path)) {
            std::printf("[eval] Loading %s (%d lines, %d tokens)...\n",
                        val_path.c_str(), (int)val_set.lines.size(), val_set.size());
            auto pr = eval_perplexity(trainer, val_set.token_ids,
                                      cmd.batch_size, cmd.seq_len);
            std::printf("  [val-set] loss=%.4f  ppl=%.2f  (%d tokens evaluated)\n",
                        pr.loss, pr.ppl, pr.n_tok);
        } else {
            std::printf("  [val-set] Could not load %s — skipped\n", val_path.c_str());
        }

        // 3. Test-set perplexity
        std::string test_path = cmd.data_dir + "/test.txt";
        TextDataset test_set;
        if (test_set.load(test_path)) {
            auto pr = eval_perplexity(trainer, test_set.token_ids,
                                      cmd.batch_size, cmd.seq_len);
            std::printf("  [test-set] loss=%.4f  ppl=%.2f  (%d tokens evaluated)\n",
                        pr.loss, pr.ppl, pr.n_tok);
        } else {
            std::printf("  [test-set] Could not load %s — skipped\n", test_path.c_str());
        }
    }

    // ---- Push to Hugging Face Hub ----
    if (!cmd.push_to_hub.empty()) {
        RepoConfig rcfg;
        rcfg.repo_id         = cmd.push_to_hub;
        rcfg.local_path      = cmd.ckpt_dir;
        rcfg.token           = cmd.token;
        rcfg.token_file      = cmd.token_file;
        rcfg.create_if_missing = true;
        char msg[128];
        std::snprintf(msg, sizeof(msg), "Input-Trio checkpoint (step %d)", trainer.step());
        rcfg.message = msg;
        repo_push(rcfg);
    }

    std::printf("\n%s", SEP);

    if (all_checks_ok && best_loss < 5.0f)
        std::printf("  ** Model capable — ready for attention layers **\n");
    else if (all_checks_ok)
        std::printf("  ** Model functional but weak (loss=%.4f); consider larger d/seq **\n", best_loss);
    else
        std::printf("  ** Pipeline checks FAILED — review architecture **\n");
    std::printf("%s\n", SEP);

    return all_checks_ok ? 0 : 1;
}
