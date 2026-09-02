"""Fail fast with the real install command — never boot a half-broken env."""

from __future__ import annotations

import inspect
import sys


def check() -> None:
    print(f"python {sys.executable}")

    try:
        import torch
        import transformers
    except ImportError as e:
        raise SystemExit(f"missing {e.name} — run ./start.sh") from e

    try:
        from transformers import MimiModel  # noqa: F401
    except Exception as e:
        cause = e.__cause__ or e
        raise SystemExit(
            f"MimiModel import failed (transformers {transformers.__version__})\n"
            f"  {type(cause).__name__}: {cause}\n"
            "  pip install -U 'transformers>=4.49,<5' accelerate\n"
            "  then ./start.sh"
        ) from e

    from kupe import ThinkSpark

    if "subfolder" not in inspect.signature(ThinkSpark.__init__).parameters:
        raise SystemExit(
            "kupe is too old (no subfolder=).\n"
            "  pip install -U 'kupe[thinkspark] @ git+https://github.com/kupe-ai/kupe-sdk.git@main'"
        )

    print(
        f"preflight ok  torch={torch.__version__} cuda={torch.cuda.is_available()}  "
        f"transformers={transformers.__version__}  kupe=subfolder+"
    )


if __name__ == "__main__":
    check()
