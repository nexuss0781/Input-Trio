Traceback (most recent call last):
  File "/content/Input-Trio/master-input/hub_push.py", line 42, in main
    from huggingface_hub import HfApi, Repository
ImportError: cannot import name 'Repository' from 'huggingface_hub' (/usr/local/lib/python3.12/dist-packages/huggingface_hub/__init__.py)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/content/Input-Trio/master-input/hub_push.py", line 73, in <module>
    sys.exit(main())
             ^^^^^^
  File "/content/Input-Trio/master-input/hub_push.py", line 46, in main
    from huggingface_hub import HfApi, Repository
ImportError: cannot import name 'Repository' from 'huggingface_hub' (/usr/local/lib/python3.12/dist-packages/huggingface_hub/__init__.py)
[repo] FAILED (rc=256).
  1. Edit HF_TOKEN with your token from https://huggingface.co/settings/tokens
  2. Or set HF_TOKEN environment variable

================================================================
  ** Model capable — ready for attention layers **
================================================================

