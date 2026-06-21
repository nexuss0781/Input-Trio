#!/usr/bin/env python3
"""
hub_push.py — Push local directory to Hugging Face Hub.
Used by repo.cpp via std::system().

Usage:
    python3 hub_push.py --repo_id user/repo --local_path ./checkpoints [--token hf_...] [--create]
"""
import argparse
import os
import sys

def _read_token_file(path="HF_TOKEN"):
    """Read token from HF_TOKEN file (TOKEN=... format). Create if missing."""
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("# Paste your Hugging Face token here:\n")
            f.write('# Get one at https://huggingface.co/settings/tokens\n')
            f.write('TOKEN=""\n')
        print(f"[hub_push] Created {path} — please edit it with your HF token.")
        return ""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("TOKEN="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                return val
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--local_path", required=True)
    parser.add_argument("--token", default="")
    parser.add_argument("--token_file", default="HF_TOKEN")
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--message", default="Upload from Input-Trio")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[hub_push] installing huggingface_hub ...")
        os.system(f"{sys.executable} -m pip install huggingface_hub -q")
        from huggingface_hub import HfApi

    token = args.token or os.environ.get("HF_TOKEN", "")
    if not token:
        token = _read_token_file(args.token_file)
    if not token:
        print("[hub_push] ERROR: no HF token found.")
        print(f"[hub_push] Edit {args.token_file} with your token, or set HF_TOKEN env var.")
        return 1

    api = HfApi(token=token)

    if args.create:
        print(f"[hub_push] Creating repo {args.repo_id} (exist_ok=True) ...")
        api.create_repo(repo_id=args.repo_id, exist_ok=True, private=False)

    print(f"[hub_push] Uploading {args.local_path} → {args.repo_id} ...")
    api.upload_folder(
        repo_id=args.repo_id,
        folder_path=args.local_path,
        commit_message=args.message,
        ignore_patterns=["__pycache__/*", "*.pyc", ".git/*"],
    )
    print(f"[hub_push] Done — https://huggingface.co/{args.repo_id}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
