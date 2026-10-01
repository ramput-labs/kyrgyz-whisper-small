"""Upload the local model folder and model card to the Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import LocalTokenNotFoundError

from kyrgyz_whisper_small.config import DEFAULT_MODEL_DIR, MODEL_ID

CARD = Path(__file__).with_name("model_card.md")
WEIGHTS_LICENSE = Path(__file__).with_name("weights_license.txt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=MODEL_ID)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    if not (args.model_dir / "model.safetensors").exists():
        raise SystemExit(f"No model in {args.model_dir}. Run `make download` first.")

    api = HfApi()
    try:
        print(f"Logged in as {api.whoami()['name']}")
    except LocalTokenNotFoundError:
        raise SystemExit("Not logged in. Run `.venv/bin/hf auth login` with a Write token from "
                         "https://huggingface.co/settings/tokens, then retry.")
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo,
        folder_path=args.model_dir,
        ignore_patterns=[".cache/*", "README.md"],
        commit_message="Upload model weights",
    )
    api.upload_file(
        repo_id=args.repo,
        path_or_fileobj=CARD,
        path_in_repo="README.md",
        commit_message="Update model card",
    )
    api.upload_file(
        repo_id=args.repo,
        path_or_fileobj=WEIGHTS_LICENSE,
        path_in_repo="LICENSE",
        commit_message="Add Apache-2.0 license",
    )
    print(f"Done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
