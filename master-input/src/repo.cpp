// =============================================================================
// repo.cpp — Hugging Face Hub Repository Push
//   Clean C++ API: RepoConfig + repo_push().
//   Delegates to hub_push.py (Python) via std::system().
//   Token from HF_TOKEN file (created auto if missing).
// =============================================================================
#ifndef INPUT_REPO_CPP
#define INPUT_REPO_CPP

#include <string>
#include <cstdio>
#include <cstdlib>

// =============================================================================
// RepoConfig — parameters for pushing to Hugging Face Hub
// =============================================================================
struct RepoConfig {
    std::string repo_id;        // e.g. "nexuss0781/Input-Trio"
    std::string local_path;     // directory to upload (e.g. "checkpoints")
    std::string message;        // commit message
    bool create_if_missing = true;
};

// =============================================================================
// repo_push — push local_path to HF Hub repo_id
//   Returns true on success.
//   Token read from HF_TOKEN file (auto-created if missing) or HF_TOKEN env.
// =============================================================================
static bool repo_push(const RepoConfig& cfg) {
    std::string cmd = "python3 hub_push.py"
                    + std::string(" --repo_id ") + cfg.repo_id
                    + std::string(" --local_path ") + cfg.local_path;

    if (cfg.create_if_missing)
        cmd += " --create";
    if (!cfg.message.empty())
        cmd += " --message \"" + cfg.message + "\"";

    std::printf("[repo] Pushing %s → https://huggingface.co/%s ...\n",
                cfg.local_path.c_str(), cfg.repo_id.c_str());
    std::fflush(stdout);

    int rc = std::system(cmd.c_str());
    if (rc == 0) {
        std::printf("[repo] Done.\n");
        return true;
    } else {
        std::fprintf(stderr, "[repo] FAILED (rc=%d).\n", rc);
        std::fprintf(stderr, "  1. Edit HF_TOKEN with your token from https://huggingface.co/settings/tokens\n");
        std::fprintf(stderr, "  2. Or set HF_TOKEN environment variable\n");
        return false;
    }
}

#endif // INPUT_REPO_CPP
