"""Environment loading helpers for evaluation commands and clients."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def candidate_env_paths(project_root: str | Path = _PROJECT_ROOT) -> tuple[Path, ...]:
    """Returns .env paths in lookup order.

    The do-um project .env has priority, and the parent workspace .env fills
    missing values for users who keep shared API keys at the workspace root.
    """
    root = Path(project_root)
    return (root / ".env", root.parent / ".env")


def load_evaluation_env(project_root: str | Path = _PROJECT_ROOT) -> list[Path]:
    """Loads available evaluation .env files without printing secret values."""
    loaded: list[Path] = []
    for path in candidate_env_paths(project_root):
        if path.exists():
            load_dotenv(path, override=False)
            loaded.append(path)
    return loaded
